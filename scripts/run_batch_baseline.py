'''
基础测试，batch版本
1/读取 data/perf_prompts.json
2/加载一次模型
3/对每条 prompt 循环生成
4/记录每条结果
5/保存到 results/batch_baseline_results.json
'''

#-------------0.导入必要的库----------------
import json
import time
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#----------------1.配置部分----------------
MODEL_PATH = "/data/users/tongf/master_thesis_tang/models/qwen2.5-1.5b"
TRUST_REMOTE_CODE = False
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512

PROMPT_FILE = Path("prompts/perf_prompts.json")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_inputs(tokenizer, prompt_text, device):
    """
    优先使用 chat template；如果不可用，则退回普通文本 tokenize。
    """
    try:
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return {"input_ids": input_ids.to(device)}
    except Exception:
        return tokenizer(prompt_text, return_tensors="pt").to(device)


def run_one_prompt(model, tokenizer, prompt_text):
    inputs = build_inputs(tokenizer, prompt_text, model.device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
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
        skip_special_tokens=True
    )

    return {
        "input_tokens": input_len,
        "output_tokens": generated_len,
        "total_tokens": total_len,
        "latency_s": latency_s,
        "tpot_ms": tpot_ms,
        "generated_text": generated_text,
    }


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU node and environment.")

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

    # ---------------- warm-up ----------------
    warmup_prompt = "What is 1 + 1?"
    print("\nRunning warm-up...")
    _ = run_one_prompt(model, tokenizer, warmup_prompt)
    print("Warm-up finished.")

    # ---------------- formal runs ----------------
    all_results = []

    for item in prompts:
        prompt_id = item["id"]
        category = item["category"]
        prompt_text = item["text"]

        print(f"\nRunning {prompt_id} ({category})...")

        result_core = run_one_prompt(model, tokenizer, prompt_text)

        result = {
            "id": prompt_id,
            "category": category,
            "prompt": prompt_text,
            **result_core,
        }

        all_results.append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    output_path = OUTPUT_DIR / "batch_baseline_results_v2_512_tokens_limit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()