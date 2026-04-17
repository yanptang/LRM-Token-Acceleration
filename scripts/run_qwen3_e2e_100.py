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
PROMPT_FILE = "prompts/perf_prompts_100_qwen3.json"
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

# 将 user prompt 转换成模型输入的 input_ids 和 attention_mask
# 这一步统一完成了 prompt 的格式化和 tokenization，方便后续的推理过程
def prepare_inputs(tokenizer, user_prompt, device):
    messages = build_messages(user_prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    return inputs.to(device)

#统计函数，计算完所有的prompt结果之后，调用这个函数来计算平均 latency、平均输入输出 token 数量等统计信息
#返回一个包含这些统计信息的字典
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
        #打印处理的信息，可以反馈在.out文件上，方便查看当前进度和正在处理的 prompt 信息
        print(f"[{idx:03d}/{len(prompts)}] Running {prompt_id} ({category})")

        #---------------------------
        #1.准备输入
        #---------------------------
        inputs = prepare_inputs(tokenizer, prompt, model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        start = time.perf_counter()

        #--------------------------------------------------------------------------------
        #2.文本生成
        # 进行文本生成，生成的文本长度不受限制，直到模型认为结束或者达到 max_new_tokens 的限制
        # 推理阶段不需要计算梯度，所以使用 torch.no_grad() 来节省内存和加速计算
        # 这里输出的是生成的 token ids，后续会解码成文本
        #--------------------------------------------------------------------------------
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=DO_SAMPLE, #是否贪心，False 表示贪心生成，True 表示使用采样生成
                pad_token_id=tokenizer.pad_token_id, #生成时使用的 pad_token_id，确保生成过程中不会因为缺少 pad_token 而出错
            )

        torch.cuda.synchronize() #等待 GPU 完成所有计算，确保计时的准确性
        end = time.perf_counter() 

        output_ids = outputs[0]  #生成的 token ids，包含了输入部分和新生成的部分
        total_len = output_ids.shape[0]
        generated_len = total_len - input_len
        latency_s = end - start
        tpot_ms = (latency_s / generated_len) * 1000 if generated_len > 0 else None

        #----------------
        #3.解码生成的文本
        # 将生成的 token ids 转换回文本，跳过输入部分的 token，只解码新生成的部分
        #----------------
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

    #生成结果的摘要保存到 summary 字段中，包含平均 latency、平均输入输出 token 数量等统计信息
    summary = summarize_results(results)

    #环境和配置信息也保存到 payload 中
    #这就是最终保存信息的结构
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

    #保存结果到 JSON 文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        #json.dump负责将 Python 对象转换成 JSON 格式的字符串，并写入到文件中。参数 indent=2 表示使用 2 个空格进行缩进，使输出的 JSON 文件更易读。参数 ensure_ascii=False 表示允许输出非 ASCII 字符（如中文），而不是将它们转义成 Unicode 字符。
        json.dump(payload, f, indent=2, ensure_ascii=False) 

    print("\n===== Average Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
