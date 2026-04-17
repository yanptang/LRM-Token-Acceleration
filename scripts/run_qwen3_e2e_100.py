import json
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ========================
# 1. Configuration
# ========================
MODEL_PATH = "/data/users/tongf/master_thesis_tang/models/Qwen3-1.7B"
PROMPT_FILE = "prompts/qwen3_1.7b_e2e_100.json"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "qwen3_1_7b_e2e_100_results.json"

DTYPE = torch.bfloat16
WARMUP_PROMPTS = 3

# Natural generation with a high safety cap
MAX_NEW_TOKENS = 9999
DO_SAMPLE = False

# Optional: set True if the model requires remote code
TRUST_REMOTE_CODE = False


# ========================
# 2. Helper functions
# ========================
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

def prepare_inputs(tokenizer, user_prompt, device):
    messages = build_messages(user_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return inputs.to(device)


def summarize_results(results):
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    input_tokens = [r["input_tokens"] for r in results if r["input_tokens"] is not None]
    output_tokens = [r["output_tokens"] for r in results if r["output_tokens"] is not None]
    total_tokens = [r["total_tokens"] for r in results if r["total_tokens"] is not None]
    tpot_values = [r["tpot_ms"] for r in results if r["tpot_ms"] is not None]

    return {
        "num_prompts": len(results),
        "avg_latency_s": mean(latencies) if latencies else None,
        "avg_input_tokens": mean(input_tokens) if input_tokens else None,
        "avg_output_tokens": mean(output_tokens) if output_tokens else None,
        "avg_total_tokens": mean(total_tokens) if total_tokens else None,
        "avg_tpot_ms": mean(tpot_values) if tpot_values else None,
        "min_latency_s": min(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "min_output_tokens": min(output_tokens) if output_tokens else None,
        "max_output_tokens": max(output_tokens) if output_tokens else None,
    }


# ========================
# 3. Main
# ========================
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU node.")

    print("GPU:", torch.cuda.get_device_name(0))

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)

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

    # ------------------------
    # Warm-up
    # 拿 prompt 文件里的前 3 条做 warm-up
    # 目的是让模型预热，避免第一次推理时间过长，以及让 GPU 进入稳定状态
    # ------------------------
    print(f"Running {WARMUP_PROMPTS} warm-up prompts...")
    for item in prompts[:WARMUP_PROMPTS]:
        inputs = prepare_inputs(tokenizer, item["prompt"], model.device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=64, #只要跑起来就可以了，并不需要生成很长的文本，所以这里设置一个较小的 max_new_tokens
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )
    torch.cuda.synchronize()
    print("Warm-up finished.")

    # ------------------------
    # Full evaluation
    # ------------------------
    results = []

    for idx, item in enumerate(prompts, start=1):
        prompt_id = item["id"]
        category = item.get("category", "unknown")
        prompt = item["prompt"]

        print(f"[{idx:03d}/{len(prompts)}] Running {prompt_id} ({category})")

        inputs = prepare_inputs(tokenizer, prompt, model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        output_ids = outputs[0]
        total_len = output_ids.shape[0]
        generated_len = total_len - input_len
        latency_s = end - start
        tpot_ms = (latency_s / generated_len) * 1000 if generated_len > 0 else None

        generated_text = tokenizer.decode(
            output_ids[input_len:],
            skip_special_tokens=True,
        )

        result = {
            "id": prompt_id,
            "category": category,
            "prompt": prompt,
            "input_tokens": input_len,
            "output_tokens": generated_len,
            "total_tokens": total_len,
            "latency_s": latency_s,
            "tpot_ms": tpot_ms,
            "generated_text": generated_text,
            "hit_max_new_tokens": generated_len >= MAX_NEW_TOKENS,
        }
        results.append(result)

    summary = summarize_results(results)

    payload = {
        "config": {
            "model_path": MODEL_PATH,
            "prompt_file": PROMPT_FILE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "warmup_prompts": WARMUP_PROMPTS,
            "dtype": str(DTYPE),
        },
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n===== Average Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
