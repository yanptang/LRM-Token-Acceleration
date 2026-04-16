'''
更新日期：2026.04.16
最基础版本的单次测试推理脚本，主要功能是：
1. 加载指定路径的预训练语言模型和对应的tokenizer。
2. 对输入的文本prompt进行tokenize，并将输入数据移动到GPU上。
3. 使用模型生成文本，并测量生成过程的时间。
4. 计算输入token数量、生成token数量、总token数量、推理时间和每token的平均推理时间（TPOT）。
5. 将生成的文本和相关统计信息以JSON格式保存到results目录下的baseline_single_run.json文件中，方便后续分析和比较不同模型或配置的性能表现。
'''

#-------------0.导入必要的库----------------
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#----------------1.配置部分----------------
#使用已下载的模型路径
MODEL_PATH = "/data/users/tongf/master_thesis_tang/models/qwen2.5-1.5b"
DTYPE = torch.bfloat16

#测试的输入prompt
PROMPT = "Solve step by step: If a train travels 60 km in 1.5 hours, what is its average speed in km/h?"
#最多生成128个新token，仅测试使用
MAX_NEW_TOKENS = 128

#创建结果输出目录，如果不存在则创建
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

#----------------2.主代码----------------
def main():
    #获取环境下的torch版本和CUDA可用性，确保GPU环境正确配置
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check GPU node and environment.")

    print("GPU:", torch.cuda.get_device_name(0))

    #加载model和tokenizer，设置local_files_only=True确保从本地加载
    #trust_remote_code=True允许加载模型定义中的自定义代码（如果有）
    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    #输入文本进行tokenize，并将输入移动到模型所在的设备（GPU）
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    #确保GPU上所有之前的操作完成后再开始计时，避免前面加载模型等操作的时间干扰推理时间的测量
    torch.cuda.synchronize()
    start = time.perf_counter()

    #使用模型生成文本，设置最大新token数量，并关闭采样以获得确定性输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    #从输出中提取生成的token ID，并计算输入长度、生成长度、总长度、推理时间和每token的平均推理时间（TPOT）
    output_ids = outputs[0]
    total_len = output_ids.shape[0]
    generated_len = total_len - input_len
    latency_s = end - start
    tpot_ms = (latency_s / generated_len) * 1000 if generated_len > 0 else None

    generated_text = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)

    result = {
        "model_path": MODEL_PATH,
        "prompt": PROMPT,
        "input_tokens": input_len, #输入token数量
        "output_tokens": generated_len,#生成token数量
        "total_tokens": total_len, #总token数量
        "latency_s": latency_s, #推理时间（秒）
        "tpot_ms": tpot_ms, #每token的平均推理时间（毫秒），time per token
        "generated_text": generated_text, #生成的文本内容
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))#打印结果
    
    #将结果以JSON格式保存到results目录下的baseline_single_run.json文件中，方便后续分析和比较
    with open(OUTPUT_DIR / "baseline_single_run.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()