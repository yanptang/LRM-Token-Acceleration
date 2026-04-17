'''
简单测试使用 PyTorch Profiler 来分析 Qwen3 模型在特定 prompt 上的 prefill 和 decode 阶段的性能表现。
没有做任何拆分优化，目的是先看看原始模型在这些 prompt 上的性能瓶颈在哪里，以便后续针对性地进行优化。
'''
import json
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================================
# 1. Configuration
# ======================================
MODEL_PATH = "/data/users/tongf/master_thesis_tang/models/Qwen3-1.7B"
PROMPT_FILE = "prompts/perf_prompts_100_qwen3.json"

OUTPUT_DIR = Path("results")
TRACE_DIR = OUTPUT_DIR / "profiler_traces"
OUTPUT_DIR.mkdir(exist_ok=True)
TRACE_DIR.mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "qwen3_torchprofiler_summary.json"

# 先只挑少量代表性 prompt
PROFILE_PROMPT_IDS = ["P050"]

DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = False
MAX_NEW_TOKENS = 128   # profiler 阶段建议先别太大，不然 trace 非常重
DO_SAMPLE = False

# profiler 里展示的 top ops 数量
TOP_K_OP = 20

# ======================================
# 2. Utilities
# ======================================
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# 构建符合模型输入格式的 messages 列表
def build_messages(user_prompt: str):
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


# 将 user prompt 转换成模型输入的 input_ids 和 attention_mask
def prepare_inputs(tokenizer, user_prompt, device):
    messages = build_messages(user_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return inputs.to(device)


def load_selected_prompts(prompt_file, selected_ids):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    prompt_map = {item["id"]: item for item in prompts}
    selected = []
    for pid in selected_ids:
        if pid not in prompt_map:
            raise ValueError(f"Prompt id {pid} not found in {prompt_file}")
        selected.append(prompt_map[pid])
    return selected


# 从 profiler 对象中提取关键信息，返回一个表格字符串和一个包含 top ops 的列表
def summarize_profiler(prof, topk=20):
    # 不同 PyTorch 版本里 profiler 字段名可能不同：
    # 有的版本用 cuda_time_total，有的版本用 device_time_total
    key_averages = prof.key_averages()

    sample_evt = key_averages[0] if len(key_averages) > 0 else None

    if sample_evt is not None and hasattr(sample_evt, "cuda_time_total"):
        sort_key = "cuda_time_total"
        get_device_time = lambda evt: evt.cuda_time_total
    elif sample_evt is not None and hasattr(sample_evt, "device_time_total"):
        sort_key = "device_time_total"
        get_device_time = lambda evt: evt.device_time_total
    else:
        # 实在没有设备时间字段，就退化成按 cpu_time_total 排序
        sort_key = "cpu_time_total"
        get_device_time = lambda evt: 0.0

    table = key_averages.table(
        sort_by=sort_key,
        row_limit=topk,
    )

    events = []
    #记录算子的名字key、CPU时间、设备时间、调用次数等信息，方便后续分析
    for evt in key_averages:
        events.append({
            "key": evt.key, #算子名字
            "cpu_time_total_us": evt.cpu_time_total, #算子在 CPU 上的总时间，单位是微秒
            "device_time_total_us": get_device_time(evt),#算子在设备（GPU）上的总时间，单位是微秒
            "count": evt.count, #算子被调用的次数
            "self_cpu_time_total_us": evt.self_cpu_time_total,#算子自身在 CPU 上的总时间，不包括子调用的时间，单位是微秒
        })

    events = sorted(events, key=lambda x: x["device_time_total_us"], reverse=True)
    return table, events[:topk]


# 预热模型，跑几次生成，目的是让模型预热，避免第一次推理时间过长，以及让 GPU 进入稳定状态
# 默认参数是生成一个简单的文本，生成长度不超过 16，跑 2 次
def warmup_model(
    model,
    tokenizer,
    device,
    warmup_text="Hello, please briefly introduce yourself.",
    warmup_new_tokens=16,
    warmup_runs=2,
):
    print(f"Running warm-up for {warmup_runs} time(s)...")

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

    for i in range(warmup_runs):
        cuda_sync()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=warmup_new_tokens,
                do_sample=DO_SAMPLE,
            )
        cuda_sync()
        print(f"  warm-up run {i+1}/{warmup_runs} done.")


# ======================================
# 3. Single prompt profiling
# ======================================
def profile_single_prompt(model, tokenizer, item):
    prompt_id = item["id"]
    category = item.get("category", "unknown")
    prompt = item["prompt"]

    inputs = prepare_inputs(tokenizer, prompt, model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_len = input_ids.shape[1]

    # -------------------------
    # Prefill profiling
    # -------------------------
    cuda_sync()
    prefill_start = time.perf_counter()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof_prefill:
        with record_function("prefill_forward"):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

    cuda_sync()
    prefill_end = time.perf_counter()
    prefill_time_s = prefill_end - prefill_start

    prefill_trace_path = TRACE_DIR / f"{prompt_id}_prefill_trace.json"
    prof_prefill.export_chrome_trace(str(prefill_trace_path))

    prefill_table, prefill_top_ops = summarize_profiler(prof_prefill, topk=TOP_K_OP)

    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    # generated_ids 只保存“模型新生成出来的 token”，不包含原始输入 prompt
    generated_ids = [next_token]

    # -------------------------
    # Decode profiling
    # -------------------------
    decode_time_s = 0.0
    decode_steps = 0
    eos_token_id = tokenizer.eos_token_id

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof_decode:
        while decode_steps < MAX_NEW_TOKENS - 1:
            if eos_token_id is not None and next_token.item() == eos_token_id:
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
            step_start = time.perf_counter()

            with record_function("decode_step"):
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

            cuda_sync()
            step_end = time.perf_counter()

            decode_time_s += (step_end - step_start)
            decode_steps += 1

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            generated_ids.append(next_token)

    decode_trace_path = TRACE_DIR / f"{prompt_id}_decode_trace.json"
    prof_decode.export_chrome_trace(str(decode_trace_path))

    decode_table, decode_top_ops = summarize_profiler(prof_decode, topk=TOP_K_OP)

    generated_ids = torch.cat(generated_ids, dim=1)
    output_len = generated_ids.shape[1]
    total_len = input_len + output_len
    total_latency_s = prefill_time_s + decode_time_s

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

        # 这里只统计 decode while 循环中真正执行了多少步
        # 不把 prefill 后拿到的第一个 token 算进 decode_steps_measured 里
        "decode_steps_measured": decode_steps,

        "hit_max_new_tokens": output_len >= MAX_NEW_TOKENS,
        "generated_text": generated_text,

        "prefill_trace_file": str(prefill_trace_path),
        "decode_trace_file": str(decode_trace_path),

        "prefill_top_ops": prefill_top_ops,
        "decode_top_ops": decode_top_ops,
        "prefill_table_text": prefill_table,
        "decode_table_text": decode_table,
    }
    return result


# ======================================
# 4. Summary
# ======================================
def summarize_results(results):
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
    }


# ======================================
# 5. Main
# ======================================
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU node.")

    print("GPU:", torch.cuda.get_device_name(0))

    # 选取要 profiling 的 prompt
    selected_prompts = load_selected_prompts(PROMPT_FILE, PROFILE_PROMPT_IDS)

    # 加载模型和 tokenizer
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

    # 确保 tokenizer 有 pad_token，否则在生成时可能会有问题
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 进行模型预热，避免第一次推理时间过长，以及让 GPU 进入稳定状态
    # 注意：warm-up 不计入 profiler 统计
    warmup_model(model, tokenizer, model.device)

    # 逐个 prompt 进行 profiling
    results = []
    for idx, item in enumerate(selected_prompts, start=1):
        print(f"\n[{idx}/{len(selected_prompts)}] Profiling {item['id']} ({item.get('category', 'unknown')})")
        result = profile_single_prompt(model, tokenizer, item)
        results.append(result)

        print(f"prefill_time_s = {result['prefill_time_s']:.4f}")
        print(f"decode_time_s  = {result['decode_time_s']:.4f}")
        print("\n--- Prefill Top Ops ---")
        print(result["prefill_table_text"])
        print("\n--- Decode Top Ops ---")
        print(result["decode_table_text"])

    summary = summarize_results(results)

    payload = {
        "config": {
            "model_path": MODEL_PATH,
            "prompt_file": PROMPT_FILE,
            "profile_prompt_ids": PROFILE_PROMPT_IDS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "dtype": str(DTYPE),
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