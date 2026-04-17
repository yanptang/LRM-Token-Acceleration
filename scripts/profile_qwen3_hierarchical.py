'''
更新日期：2026.04.17

这个版本是用于 Qwen3-1.7B 的分层 profiling 脚本，目标是：
1. 先将单条样本的生成过程拆分为 Prefill / Decode 两大阶段。
2. 再将 Decode 内部进一步拆分为：
   - Transformer
   - LM Head
   - Sampling
3. 使用“自然生成”方式：
   - 正常情况下，遇到 EOS 就停止
   - 同时保留一个很大的 HARD_MAX_NEW_TOKENS 作为保险上限，避免异常情况下无限生成
4. 导出轻量级 trace：
   - Prefill 全部导出 trace
   - Decode 只导出前若干步（默认 3 步）的 trace，避免 json 文件过大
5. 输出最终的 json 结果，包含：
   - 端到端时间
   - Prefill / Decode 时间
   - Decode 内 Transformer / LM Head / Sampling 时间
   - 各部分占比
   - 自然停止信息
   - 生成文本
'''

# ======================================
# 0. 导入必要的库
# ======================================
import json
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================================
# 1. 配置部分
# ======================================
# 使用本地模型路径
MODEL_PATH = "/data/users/tongf/master_thesis_tang/models/Qwen3-1.7B"

# prompt 文件路径
PROMPT_FILE = "prompts/perf_prompts_100_qwen3.json"

# 本次只测一条样本
PROFILE_PROMPT_IDS = ["P050"]

# 输出目录
OUTPUT_DIR = Path("results")
TRACE_DIR = OUTPUT_DIR / "profiler_traces"
OUTPUT_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)

# 输出 json 文件
OUTPUT_JSON = OUTPUT_DIR / "qwen3_hierarchical_profile_p050.json"

# 模型精度
DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = False

# “自然生成”为主，但保留一个极大的保险上限，防止异常情况下无限生成
HARD_MAX_NEW_TOKENS = 4096

# Decode trace 只记录前几步，避免 trace 文件过大
TRACE_DECODE_STEPS = 3

# warm-up 次数
WARMUP_RUNS = 2


# ======================================
# 2. 一些辅助函数
# ======================================
def cuda_sync():
    """如果有 CUDA，则强制同步，保证计时更准确。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_messages(user_prompt: str):
    """
    构造 chat template 所需的 messages。
    这里保留 system + user 的格式，和 Qwen3 的聊天模板一致。
    """
    return [
        {
            "role": "system",
            "content": "You are a careful reasoning assistant. Provide a clear step-by-step answer.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def prepare_inputs(tokenizer, user_prompt, device):
    """
    将用户 prompt 应用 chat template，转成模型输入张量。
    """
    messages = build_messages(user_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return inputs.to(device)


def load_selected_prompts(prompt_file, selected_ids):
    """
    从 prompt 文件中加载指定 id 的 prompt。
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompt_map = {item["id"]: item for item in prompts}
    selected = []
    for pid in selected_ids:
        if pid not in prompt_map:
            raise ValueError(f"Prompt id {pid} not found in {prompt_file}")
        selected.append(prompt_map[pid])
    return selected


def warmup_model(model, tokenizer, device, warmup_text="Please briefly introduce yourself."):
    """
    预热模型，避免第一次推理时间异常偏大。
    这里使用一个简短 prompt，跑固定次数，不导出 trace，也不写入最终结果。
    """
    print(f"Running warm-up for {WARMUP_RUNS} time(s)...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": warmup_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    for i in range(WARMUP_RUNS):
        cuda_sync()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=64,   # warm-up 用一个小值即可
                do_sample=False,
            )
        cuda_sync()
        print(f"  warm-up run {i + 1}/{WARMUP_RUNS} done.")


def safe_div(a, b):
    """安全除法，避免除零错误。"""
    return a / b if b not in [0, None] else None


# ======================================
# 3. 核心 profiling 逻辑
# ======================================
def profile_single_prompt(model, tokenizer, item):
    """
    对单条 prompt 进行分层 profiling：
    1. Prefill 单独计时，并导出完整 prefill trace
    2. Decode 总时间单独计时
    3. Decode 内部进一步拆成：
       - Transformer
       - LM Head
       - Sampling
    4. Decode 的 trace 只记录前 TRACE_DECODE_STEPS 步
    """
    prompt_id = item["id"]
    category = item.get("category", "unknown")
    prompt = item["prompt"]

    inputs = prepare_inputs(tokenizer, prompt, model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[1]

    eos_token_id = tokenizer.eos_token_id

    # ==================================================
    # 3.1 Prefill profiling
    # ==================================================
    # Prefill 的目标：
    # - 一次性处理完整输入 prompt
    # - 构建 past_key_values (KV cache)
    # - 得到最后一个 hidden state，供 lm_head 生成第一个 token
    cuda_sync()
    prefill_start = time.perf_counter()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof_prefill:
        with record_function("prefill_total"):
            with torch.no_grad():
                # 这里不要直接调用 model(...) 拿 logits
                # 而是显式调用 model.model(...)，这样后面可以把 lm_head 单独拆出来
                with record_function("prefill_transformer"):
                    backbone_outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )

                hidden_states = backbone_outputs.last_hidden_state
                past_key_values = backbone_outputs.past_key_values

                # prefill 的最后一步，需要用最后一个 hidden state 过 lm_head，
                # 得到“第一个要生成的 token”的 logits
                with record_function("prefill_lm_head"):
                    logits = model.lm_head(hidden_states[:, -1:, :])

                with record_function("prefill_sampling"):
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    cuda_sync()
    prefill_end = time.perf_counter()
    prefill_time_s = prefill_end - prefill_start

    prefill_trace_path = TRACE_DIR / f"{prompt_id}_prefill_trace_hierarchical.json"
    prof_prefill.export_chrome_trace(str(prefill_trace_path))

    # 第一个 token 是 prefill 末尾生成出来的
    generated_token_ids = [next_token]
    natural_stop = False
    hit_hard_cap = False

    # 如果第一个 token 就已经是 eos，则可视为自然停止
    if eos_token_id is not None and next_token.item() == eos_token_id:
        natural_stop = True

    # ==================================================
    # 3.2 Decode profiling（完整计时）
    # ==================================================
    # decode 总时间
    decode_time_s = 0.0

    # decode 内部分项时间
    transformer_time_s = 0.0
    lm_head_time_s = 0.0
    sampling_time_s = 0.0

    # 逐步统计
    decode_steps = 0
    per_step_stats = []

    # 轻量级 decode trace：只记录前 TRACE_DECODE_STEPS 步
    # 注意：这里只是导出局部 trace，不影响完整 decode 时间统计
    trace_enabled = TRACE_DECODE_STEPS > 0
    prof_decode_trace = None

    if trace_enabled:
        prof_decode_trace = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof_decode_trace.__enter__()

    try:
        while True:
            # 如果 prefill 已经生成 eos，则不进入 decode
            if natural_stop:
                break

            # 达到保险上限时停止，防止异常情况下无限生成
            if len(generated_token_ids) >= HARD_MAX_NEW_TOKENS:
                hit_hard_cap = True
                break

            # 当前 decode step 的输入，就是上一步生成出的单个 token
            current_input_ids = next_token

            # attention_mask 需要在末尾补一个 1，表示新 token 被加入序列
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )

            cuda_sync()
            decode_step_start = time.perf_counter()

            # ---------------------------
            # (1) Transformer 部分
            # ---------------------------
            cuda_sync()
            t1 = time.perf_counter()

            if trace_enabled and decode_steps < TRACE_DECODE_STEPS:
                with record_function("decode_transformer"):
                    with torch.no_grad():
                        backbone_outputs = model.model(
                            input_ids=current_input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
            else:
                with torch.no_grad():
                    backbone_outputs = model.model(
                        input_ids=current_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

            cuda_sync()
            t2 = time.perf_counter()
            transformer_step_s = t2 - t1
            transformer_time_s += transformer_step_s

            hidden_states = backbone_outputs.last_hidden_state
            past_key_values = backbone_outputs.past_key_values

            # ---------------------------
            # (2) LM Head 部分
            # ---------------------------
            cuda_sync()
            t3 = time.perf_counter()

            if trace_enabled and decode_steps < TRACE_DECODE_STEPS:
                with record_function("decode_lm_head"):
                    with torch.no_grad():
                        logits = model.lm_head(hidden_states[:, -1:, :])
            else:
                with torch.no_grad():
                    logits = model.lm_head(hidden_states[:, -1:, :])

            cuda_sync()
            t4 = time.perf_counter()
            lm_head_step_s = t4 - t3
            lm_head_time_s += lm_head_step_s

            # ---------------------------
            # (3) Sampling 部分
            # ---------------------------
            cuda_sync()
            t5 = time.perf_counter()

            if trace_enabled and decode_steps < TRACE_DECODE_STEPS:
                with record_function("decode_sampling"):
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            cuda_sync()
            t6 = time.perf_counter()
            sampling_step_s = t6 - t5
            sampling_time_s += sampling_step_s

            cuda_sync()
            decode_step_end = time.perf_counter()
            decode_step_total_s = decode_step_end - decode_step_start
            decode_time_s += decode_step_total_s

            decode_steps += 1
            generated_token_ids.append(next_token)

            per_step_stats.append(
                {
                    "step_idx": decode_steps,
                    "transformer_time_ms": transformer_step_s * 1000,
                    "lm_head_time_ms": lm_head_step_s * 1000,
                    "sampling_time_ms": sampling_step_s * 1000,
                    "decode_step_total_ms": decode_step_total_s * 1000,
                }
            )

            # 如果当前生成 token 是 eos，则自然停止
            if eos_token_id is not None and next_token.item() == eos_token_id:
                natural_stop = True
                break

    finally:
        if trace_enabled:
            prof_decode_trace.__exit__(None, None, None)

    # 导出 decode 前若干步的轻量 trace
    decode_trace_path = TRACE_DIR / f"{prompt_id}_decode_first{TRACE_DECODE_STEPS}steps_trace.json"
    if trace_enabled:
        prof_decode_trace.export_chrome_trace(str(decode_trace_path))

    # ==================================================
    # 3.3 结果整理
    # ==================================================
    generated_ids = torch.cat(generated_token_ids, dim=1)
    output_len = generated_ids.shape[1]
    total_len = input_len + output_len
    total_latency_s = prefill_time_s + decode_time_s

    # 端到端 TPOT：总延迟 / 输出 token 数
    e2e_tpot_ms = safe_div(total_latency_s, output_len)
    if e2e_tpot_ms is not None:
        e2e_tpot_ms *= 1000

    # decode-only TPOT：decode 总时间 / 输出 token 数
    # 注意：这里 output_len 包括 prefill 生成出的第一个 token
    # 从严格意义上说，decode_steps 只对应后续 token
    # 这里保留这个口径，便于整体分析；如果你后面想更严谨，也可以改成除以 decode_steps
    decode_tpot_ms = safe_div(decode_time_s, output_len)
    if decode_tpot_ms is not None:
        decode_tpot_ms *= 1000

    # 平均每 step 时间（decode loop 内部）
    avg_transformer_ms_per_step = safe_div(transformer_time_s * 1000, decode_steps)
    avg_lm_head_ms_per_step = safe_div(lm_head_time_s * 1000, decode_steps)
    avg_sampling_ms_per_step = safe_div(sampling_time_s * 1000, decode_steps)

    # decode 内部占比
    transformer_ratio_in_decode = safe_div(transformer_time_s, decode_time_s)
    lm_head_ratio_in_decode = safe_div(lm_head_time_s, decode_time_s)
    sampling_ratio_in_decode = safe_div(sampling_time_s, decode_time_s)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    result = {
        "id": prompt_id,
        "category": category,
        "prompt": prompt,

        "input_tokens": input_len,
        "output_tokens": output_len,
        "total_tokens": total_len,

        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "total_latency_s": total_latency_s,

        "e2e_tpot_ms": e2e_tpot_ms,
        "decode_tpot_ms": decode_tpot_ms,

        "decode_steps_measured": decode_steps,

        "transformer_time_s": transformer_time_s,
        "lm_head_time_s": lm_head_time_s,
        "sampling_time_s": sampling_time_s,

        "transformer_ratio_in_decode": transformer_ratio_in_decode,
        "lm_head_ratio_in_decode": lm_head_ratio_in_decode,
        "sampling_ratio_in_decode": sampling_ratio_in_decode,

        "avg_transformer_ms_per_step": avg_transformer_ms_per_step,
        "avg_lm_head_ms_per_step": avg_lm_head_ms_per_step,
        "avg_sampling_ms_per_step": avg_sampling_ms_per_step,

        "natural_stop": natural_stop,
        "hit_hard_cap": hit_hard_cap,
        "hard_max_new_tokens": HARD_MAX_NEW_TOKENS,

        "prefill_trace_file": str(prefill_trace_path),
        "decode_trace_file": str(decode_trace_path),

        "generated_text": generated_text,

        # 保存每一步的细粒度时间，后面画图会很有用
        "per_step_stats": per_step_stats,
    }

    return result


# ======================================
# 4. Summary 汇总
# ======================================
def summarize_results(results):
    """
    对多条结果取平均。
    目前虽然只测 P050 一条，但后面扩展到多条时可以直接复用。
    """
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "num_profiled_prompts": len(results),
        "avg_input_tokens": avg("input_tokens"),
        "avg_output_tokens": avg("output_tokens"),
        "avg_prefill_time_s": avg("prefill_time_s"),
        "avg_decode_time_s": avg("decode_time_s"),
        "avg_total_latency_s": avg("total_latency_s"),
        "avg_e2e_tpot_ms": avg("e2e_tpot_ms"),
        "avg_decode_tpot_ms": avg("decode_tpot_ms"),
        "avg_transformer_time_s": avg("transformer_time_s"),
        "avg_lm_head_time_s": avg("lm_head_time_s"),
        "avg_sampling_time_s": avg("sampling_time_s"),
        "avg_transformer_ratio_in_decode": avg("transformer_ratio_in_decode"),
        "avg_lm_head_ratio_in_decode": avg("lm_head_ratio_in_decode"),
        "avg_sampling_ratio_in_decode": avg("sampling_ratio_in_decode"),
    }


# ======================================
# 5. 主函数
# ======================================
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU node.")

    print("GPU:", torch.cuda.get_device_name(0))

    # 加载要 profile 的 prompt
    selected_prompts = load_selected_prompts(PROMPT_FILE, PROFILE_PROMPT_IDS)

    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.eval()

    # 某些 tokenizer 没有 pad_token，这里补上
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 预热模型
    warmup_model(model, tokenizer, model.device)

    # 逐条 profiling
    results = []
    for idx, item in enumerate(selected_prompts, start=1):
        print(f"\n[{idx}/{len(selected_prompts)}] Profiling {item['id']} ({item.get('category', 'unknown')})")

        result = profile_single_prompt(model, tokenizer, item)
        results.append(result)

        print(f"input_tokens              = {result['input_tokens']}")
        print(f"output_tokens             = {result['output_tokens']}")
        print(f"prefill_time_s            = {result['prefill_time_s']:.6f}")
        print(f"decode_time_s             = {result['decode_time_s']:.6f}")
        print(f"total_latency_s           = {result['total_latency_s']:.6f}")
        print(f"e2e_tpot_ms               = {result['e2e_tpot_ms']:.6f}" if result["e2e_tpot_ms"] is not None else "e2e_tpot_ms = None")
        print(f"decode_tpot_ms            = {result['decode_tpot_ms']:.6f}" if result["decode_tpot_ms"] is not None else "decode_tpot_ms = None")

        print(f"transformer_time_s        = {result['transformer_time_s']:.6f}")
        print(f"lm_head_time_s            = {result['lm_head_time_s']:.6f}")
        print(f"sampling_time_s           = {result['sampling_time_s']:.6f}")

        print(f"transformer_ratio_decode  = {result['transformer_ratio_in_decode']:.4f}" if result["transformer_ratio_in_decode"] is not None else "transformer_ratio_decode = None")
        print(f"lm_head_ratio_decode      = {result['lm_head_ratio_in_decode']:.4f}" if result["lm_head_ratio_in_decode"] is not None else "lm_head_ratio_decode = None")
        print(f"sampling_ratio_decode     = {result['sampling_ratio_in_decode']:.4f}" if result["sampling_ratio_in_decode"] is not None else "sampling_ratio_decode = None")

        print(f"natural_stop              = {result['natural_stop']}")
        print(f"hit_hard_cap              = {result['hit_hard_cap']}")

    summary = summarize_results(results)

    payload = {
        "config": {
            "model_path": MODEL_PATH,
            "prompt_file": PROMPT_FILE,
            "profile_prompt_ids": PROFILE_PROMPT_IDS,
            "dtype": str(DTYPE),
            "hard_max_new_tokens": HARD_MAX_NEW_TOKENS,
            "trace_decode_steps": TRACE_DECODE_STEPS,
            "warmup_runs": WARMUP_RUNS,
        },
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n===== Overall Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to: {OUTPUT_JSON}")
    print(f"Saved traces to: {TRACE_DIR}")


if __name__ == "__main__":
    main()