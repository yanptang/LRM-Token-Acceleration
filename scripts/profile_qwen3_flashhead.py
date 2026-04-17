# profile_qwen3_flashhead.py
# 更新日期：2026.04.17
#
# 目的：
# 1. 基于你现有的 qwen3 分层 profiling 脚本，集成 FlashHead
# 2. Prefill 阶段保持 dense lm_head，不改
# 3. Decode 阶段将 dense lm_head + sampling 替换为 FlashHead.get_next_token()
# 4. 继续输出：
#    - 端到端时间
#    - Prefill / Decode 时间
#    - Decode 内 Transformer / LM Head / Sampling 时间
#    - trace 文件
#
# 说明：
# - 这是“最稳的第一版”：
#   * greedy only（do_sample=False）
#   * prefill dense
#   * decode flashhead
# - FlashHead 模式下：
#   * decode_lm_head = FlashHead 全部开销
#   * decode_sampling = 0
#
# 依赖：
# - 同目录下存在 flash_head.py（原作者提供）
# - 本地模型目录或 HF cache 中可找到 clustering_cache.safetensors

# ======================================
# 0. 导入必要库
# ======================================
import json
import time
import random
import sys
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer, AutoModelForCausalLM

# 如果 flash_head.py 与本脚本同目录，确保可导入
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from flash_head import FlashHead, get_flash_head_parameters


# ======================================
# 1. 配置部分
# ======================================
# 本地模型路径
MODEL_PATH = "/data/users/tongf/master_thesis_tang/models/Qwen3-1.7B"

# prompt 文件路径
PROMPT_FILE = "/data/users/tongf/master_thesis_tang/prompts/perf_prompts_100_qwen3.json"

# 本次测试多条样本：默认 30 条
random.seed(42)
PROFILE_PROMPT_IDS = sorted(random.sample([f"P{i:03d}" for i in range(1, 101)], 3))

# 输出目录
OUTPUT_DIR = Path("results")
TRACE_DIR = OUTPUT_DIR / "profiler_traces_flashhead"
OUTPUT_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)

# 输出 json 文件
OUTPUT_JSON = OUTPUT_DIR / "qwen3_flashhead_profile_30prompts.json"

# 模型精度
DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = False

# “自然生成”为主，但保留一个极大的保险上限
HARD_MAX_NEW_TOKENS = 4096

# Decode trace 只记录前几步
TRACE_DECODE_STEPS = 3

# warm-up 次数
WARMUP_RUNS = 2

# ========== FlashHead 相关配置 ==========
USE_FLASHHEAD = True

# 这里填“聚类缓存所在目录”，相对于 model_or_dir 的 cache_dir
# 举例：
#   如果 clustering_cache.safetensors 在
#   /data/users/tongf/master_thesis_tang/models/Qwen3-1.7B/flash_head_cache/clustering_cache.safetensors
#   那么这里填 "flash_head_cache"
FLASHHEAD_CACHE_DIR = "flash_head_cache"

# 这里通常直接等于 MODEL_PATH；如果你想从 HF repo id 加载，也可以填 repo 名
FLASHHEAD_MODEL_OR_DIR = MODEL_PATH

# 先扫 probes，cluster 数体现在你加载的 clustering cache 中
FLASHHEAD_N_PROBES = 256

# 第一版先不额外传 special tokens
FLASHHEAD_SPECIAL_TOKEN_IDS = None

# greedy only
FLASHHEAD_DO_SAMPLE = False


# ======================================
# 2. 辅助函数
# ======================================
def cuda_sync():
    """如果有 CUDA，则强制同步，保证计时更准确。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_div(a, b):
    """安全除法，避免除零。"""
    return a / b if b not in [0, None] else None


def build_messages(user_prompt: str):
    """
    构造 chat template 所需 messages。
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
    应用 chat template，转成模型输入张量。
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
    从 prompt 文件加载指定 id 的 prompt。
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
    预热模型。
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
                max_new_tokens=64,
                do_sample=False,
            )
        cuda_sync()
        print(f"  warm-up run {i + 1}/{WARMUP_RUNS} done.")


def build_flash_head(model):
    """
    根据作者提供接口构造 FlashHead。
    """
    
    params = get_flash_head_parameters(
        lm_head=model.lm_head,
        cache_dir=FLASHHEAD_CACHE_DIR,
        model_or_dir=FLASHHEAD_MODEL_OR_DIR,
    )

    flash_head = FlashHead(
        lm_head=model.lm_head,
        centroids=params["centroids"],
        vocab_maps_tensor=params["vocab_maps_tensor"],
        n_probes=FLASHHEAD_N_PROBES,
        special_token_ids=FLASHHEAD_SPECIAL_TOKEN_IDS,
    )
    print("lm_head.weight.shape =", tuple(model.lm_head.weight.shape))
    print("flash_head.centroids.shape =", tuple(flash_head.centroids.shape))
    print("flash_head.cluster_linear.weight.shape =", tuple(flash_head.cluster_linear.weight.shape))
    flash_head.eval()
    flash_head.to(model.device)
    return flash_head


# ======================================
# 3. 核心 profiling 逻辑
# ======================================
def profile_single_prompt(model, tokenizer, item, flash_head=None):
    """
    对单条 prompt 进行 profiling：

    1. Prefill：
       - transformer / lm_head / sampling
       - 保持 dense lm_head，不改
    2. Decode：
       - transformer
       - 若 flash_head is None：dense lm_head + argmax
       - 若 flash_head is not None：FlashHead.get_next_token()
    3. FlashHead 模式下：
       - decode_lm_head = FlashHead 整体耗时
       - decode_sampling = 0
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
    # 3.1 Prefill profiling（保持 dense）
    # ==================================================
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
                with record_function("prefill_transformer"):
                    backbone_outputs = model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )

                hidden_states = backbone_outputs.last_hidden_state
                past_key_values = backbone_outputs.past_key_values

                with record_function("prefill_lm_head"):
                    logits = model.lm_head(hidden_states[:, -1:, :])

                with record_function("prefill_sampling"):
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    cuda_sync()
    prefill_end = time.perf_counter()
    prefill_time_s = prefill_end - prefill_start

    prefill_trace_path = TRACE_DIR / f"{prompt_id}_prefill_trace_flashhead.json"
    prof_prefill.export_chrome_trace(str(prefill_trace_path))

    # 第一个 token 是 prefill 末尾生成的
    generated_token_ids = [next_token]
    natural_stop = False
    hit_hard_cap = False

    if eos_token_id is not None and next_token.item() == eos_token_id:
        natural_stop = True

    # ==================================================
    # 3.2 Decode profiling
    # ==================================================
    decode_time_s = 0.0
    transformer_time_s = 0.0
    lm_head_time_s = 0.0
    sampling_time_s = 0.0

    decode_steps = 0

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
            if natural_stop:
                break

            if len(generated_token_ids) >= HARD_MAX_NEW_TOKENS:
                hit_hard_cap = True
                break

            current_input_ids = next_token

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
            # (1) Transformer
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
            # (2) LM Head / FlashHead
            # ---------------------------
            cuda_sync()
            t3 = time.perf_counter()

            if flash_head is None:
                # ===== Baseline 路径：dense lm_head =====
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
                # (3) Sampling
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

            else:
                # ===== FlashHead 路径：直接返回 next_token =====
                if trace_enabled and decode_steps < TRACE_DECODE_STEPS:
                    with record_function("decode_lm_head"):
                        with torch.no_grad():
                            next_token = flash_head.get_next_token(
                                hidden_states[:, -1:, :],
                                do_sample=FLASHHEAD_DO_SAMPLE,
                            )
                else:
                    with torch.no_grad():
                        next_token = flash_head.get_next_token(
                            hidden_states[:, -1:, :],
                            do_sample=FLASHHEAD_DO_SAMPLE,
                        )

                cuda_sync()
                t4 = time.perf_counter()
                lm_head_step_s = t4 - t3
                lm_head_time_s += lm_head_step_s

                # FlashHead 已完成 token selection
                # 为保持和 baseline 表结构一致，这里 sampling 记 0
                sampling_step_s = 0.0
                sampling_time_s += sampling_step_s

            cuda_sync()
            decode_step_end = time.perf_counter()
            decode_step_total_s = decode_step_end - decode_step_start
            decode_time_s += decode_step_total_s

            decode_steps += 1
            generated_token_ids.append(next_token)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                natural_stop = True
                break

    finally:
        if trace_enabled:
            prof_decode_trace.__exit__(None, None, None)

    decode_trace_path = TRACE_DIR / f"{prompt_id}_decode_first{TRACE_DECODE_STEPS}steps_trace_flashhead.json"
    if trace_enabled:
        prof_decode_trace.export_chrome_trace(str(decode_trace_path))

    # ==================================================
    # 3.3 结果整理
    # ==================================================
    generated_ids = torch.cat(generated_token_ids, dim=1)
    output_len = generated_ids.shape[1]
    total_len = input_len + output_len
    total_latency_s = prefill_time_s + decode_time_s

    e2e_tpot_ms = safe_div(total_latency_s, output_len)
    if e2e_tpot_ms is not None:
        e2e_tpot_ms *= 1000

    decode_tpot_ms = safe_div(decode_time_s, output_len)
    if decode_tpot_ms is not None:
        decode_tpot_ms *= 1000

    avg_transformer_ms_per_step = safe_div(transformer_time_s * 1000, decode_steps)
    avg_lm_head_ms_per_step = safe_div(lm_head_time_s * 1000, decode_steps)
    avg_sampling_ms_per_step = safe_div(sampling_time_s * 1000, decode_steps)

    transformer_ratio_in_decode = safe_div(transformer_time_s, decode_time_s)
    lm_head_ratio_in_decode = safe_div(lm_head_time_s, decode_time_s)
    sampling_ratio_in_decode = safe_div(sampling_time_s, decode_time_s)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    result = {
        "id": prompt_id,
        "category": category,
        "prompt": prompt,

        "mode": "flashhead" if flash_head is not None else "baseline",
        "use_flashhead": flash_head is not None,
        "flashhead_n_probes": FLASHHEAD_N_PROBES if flash_head is not None else None,
        "flashhead_cache_dir": FLASHHEAD_CACHE_DIR if flash_head is not None else None,
        "flashhead_model_or_dir": FLASHHEAD_MODEL_OR_DIR if flash_head is not None else None,

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
    }

    return result


# ======================================
# 4. Summary 汇总
# ======================================
def summarize_results(results):
    """
    对多条结果取平均。
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
        "avg_decode_steps_measured": avg("decode_steps_measured"),
        "natural_stop_count": sum(1 for r in results if r.get("natural_stop") is True),
        "hit_hard_cap_count": sum(1 for r in results if r.get("hit_hard_cap") is True),
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

    # 加载 prompts
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

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建 FlashHead
    flash_head = None
    if USE_FLASHHEAD:
        print("Building FlashHead...")
        flash_head = build_flash_head(model)
        print("FlashHead ready.")
        print(f"  cache_dir = {FLASHHEAD_CACHE_DIR}")
        print(f"  n_probes  = {FLASHHEAD_N_PROBES}")

    # 预热
    warmup_model(model, tokenizer, model.device)

    # 逐条 profiling
    results = []
    for idx, item in enumerate(selected_prompts, start=1):
        print(f"\n[{idx}/{len(selected_prompts)}] Profiling {item['id']} ({item.get('category', 'unknown')})")

        result = profile_single_prompt(
            model=model,
            tokenizer=tokenizer,
            item=item,
            flash_head=flash_head,
        )
        results.append(result)

        print(f"mode                      = {result['mode']}")
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
            "use_flashhead": USE_FLASHHEAD,
            "flashhead_cache_dir": FLASHHEAD_CACHE_DIR if USE_FLASHHEAD else None,
            "flashhead_model_or_dir": FLASHHEAD_MODEL_OR_DIR if USE_FLASHHEAD else None,
            "flashhead_n_probes": FLASHHEAD_N_PROBES if USE_FLASHHEAD else None,
            "flashhead_do_sample": FLASHHEAD_DO_SAMPLE if USE_FLASHHEAD else None,
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