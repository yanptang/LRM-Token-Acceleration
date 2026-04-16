记录论文的进度和原有计划清单

核心目标：如何在不明显降低推理质量的前提下，加速大型推理模型（LRM）的 token 生成速度
核心思路是两件事：
- 算法层面：用 FlashHead 替换原来的分类头（减少计算）
- 系统层面：用 Triton 优化 GPU kernel（提高执行效率）

总结来说回答的问题：FlashHead + Triton 能不能在长推理任务中显著降低推理成本，同时保持结果质量？

量化问题：
1. 能快多少？（Latency / TPOT）--->降低推理成本
2. 准确率会掉多少？          ---->保持结果质量
3. 最优参数配置是什么？      ----->参数结果

要做的事情：
1️⃣ 建立 baseline + 找瓶颈（你论文的起点）
- 跑原始模型（qwen2.5-1.5B）
- 测这些指标：
    - TPOT（每个token时间）
    - 总延迟
    - classification head占多少时间
    - 用 profiler 分析瓶颈


2️⃣ 集成 FlashHead + 做 trade-off 实验
把 FlashHead 接进模型（替换 classification head）
调参数：
cluster 数量（C）
probe 数量（P）

| 指标   | 你要看                |
| ---- | ------------------ |
| 速度   | latency / TPOT     |
| 准确性  | Top-1 / Top-k      |
| 推理能力 | benchmark（如 GSM8K） |


3️⃣ Triton kernel 优化（系统优化）
用 Triton 重写关键计算（FlashHead部分）
优化：
- memory access（coalescing）
- kernel fusion
- shared memory 使用