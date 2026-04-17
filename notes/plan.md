记录论文的进度和原有计划清单

核心目标：如何在不明显降低推理质量的前提下，加速大型推理模型（LRM）的 token 生成速度（把一个大模型“最后一步很慢的地方”改成更快的近似算法，并用系统优化让它跑得更快，然后证明这样做是值得的）
**但是作为毕业论文来说，在 LRM / reasoning-style generation 场景中复现 FlashHead，并验证其性能收益；这就是全部要做的事情**
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
- 跑原始模型（qwen2.5-1.5B），在单GPU上稳定运行，记录推理时间和生成的token数量（针对LRM长推理）
- 测这些指标：
    - TPOT（每个token时间）
    - 总延迟
    - 用 profiler 分析瓶颈,即什么阶段占用时间最多（如 attention, feedforward, head等）
    - classification head占多少时间
- 回答两个问题：
    - 1. 这个模型现在的速度如何？通过以下指标
        - latency（总推理时间）
        - TPOT（每个token的平均推理时间，Time Per Output Token），计算公式=总推理时间 / 生成的token数量
    - 2. 慢主要慢在哪个阶段？通过profiler分析来找出瓶颈，特别关注classification head的时间占比（因为FlashHead算法专注解决这个问题）
        - Decode-stage latency（解码阶段的延迟）
        - classification head 占总时间的比例

本阶段输出的结论类似为：
```
    “在 batch size=1、单 GPU、长推理输出下，classification head 占 decode 时间 xx%，因此值得优化”
    或者 “classification head 占比不高，真正瓶颈在 attention / KV cache / memory bandwidth”
```

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


-------------------------------------------------------

主线：
baseline：建立 LRM workload
profiler：建立解释框架
FlashHead：实现迁移
before/after：做应用验证
analysis：解释适用性和意义

高性能triton：
FlashHead集成下的LRM，再加入triton算子使得多GPU计算；看是否提升