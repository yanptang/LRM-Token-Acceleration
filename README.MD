### 论文说明
- 本论文研究 FlashHead 在长推理语言模型推理场景中的应用与验证，评估该方法在 LRM 负载下的适用性与效果
- 在此基础上，加入 Triton 优化，进一步提升性能，并分析优化效果
- 以上总结为，单GPU下的算法提升+多核心优化，能否在 LRM 场景下显著降低推理成本，同时保持结果质量



### 指标定义



### Projest Structure
scripts/：python scripts

prompts/：测试输入的promt

results/：实验输出

notes/：记录你开发信息和日志

documents/：存放相关文献，前置知识

models/：存放模型文件

result_analysis/：存放结果分析的代码和输出

README.md：写项目说明


### 开发环境信息和简要操作指南

- 实验环境集群：minerva
- Hostname:uranus,neptune
- GPU:NVIDIA L40s
- 远程实验环境：mt
- 基础模型：Qwen2.5-1.5b
- 推理精度：BF16

```bash
conda activate mt
```

服务器操作指南：
minerva 是login node → 写代码
→ 提交到集群（europa 等compute node）运行
Slurm进行调度和任务管理，有两种常用操作方式
→ srun 是交互式，可以直接在命令行上运行程序，适合调试和小规模任务
```bash
srun --gpus=1 --pty bash
```
→ sbatch 是提交任务排队，可以提交提交批处理任务进行正式实验


git 提交
```bash
git status #查看当前文件状态，了解哪些文件被修改了，哪些文件处于暂存区，哪些文件未跟踪等信息
git add .  #添加所有修改的文件到暂存区
git commit -m "本次提交说明"  #将暂存区的修改提交到本地仓库，并附上提交说明
git push #将本地提交推送到远程仓库
```

常用slurm指令
```bash
squeue -u $USER  #查看当前用户的任务队列
scontrol show job 你的JOBID #查看指定JOBID的详细信息
watch -n 2 squeue -u tongf #每2秒刷新一次，查看当前用户的任务队列
sinfo -N -o "%N %G %t %C" #查看集群节点状态，显示节点名称ac、GPU类型、状态和CPU/GPU资源使用情况
scancel JOBID #取消指定JOBID的任务 b  
```



### 指标清单
| 指标名                     | 公式 / 定义                            |             单位 | 反映什么                       | 你现在怎么用                                      |
| ----------------------- | ---------------------------------- | -------------: | -------------------------- | ------------------------------------------- |
| `input_tokens`          | prompt 经 tokenizer 后的输入长度          |         tokens | 输入规模，主要影响 prefill 计算量      | 用来描述每条测试样本的输入长度，辅助解释 prefill 开销             |
| `output_tokens`         | 模型实际生成的输出长度                        |         tokens | 输出规模，主要影响 decode 总开销       | 用来衡量回答有多长，也是 latency 和 TPOT 的核心解释变量         |
| `total_tokens`          | `input_tokens + output_tokens`     |         tokens | 一次完整推理处理的总 token 规模        | 用来做整体规模统计，但不是最核心瓶颈指标                        |
| `latency_s`             | 一次 `generate()` 从开始到结束的总耗时         |        seconds | 端到端总推理时延                   | 作为最直观的整体性能指标，比较不同 prompt 或优化前后的总速度          |
| `tpot_ms`               | `latency_s / output_tokens * 1000` |       ms/token | 端到端平均每个输出 token 的生成成本      | 作为 baseline 的核心效率指标，比较优化前后“单位 token 成本”是否下降 |
| `hit_max_new_tokens`    | 是否因为达到 `max_new_tokens` 才停止        |           bool | 输出是否被人工截断                  | 用来判断实验是否接近真实自然生成；若大量为 `true`，说明结果受人为上限影响    |
| `prefill_time_s`        | profiler 中首轮前向传播耗时                 |        seconds | 处理整段输入上下文的成本               | 用来衡量 prefill 阶段在总时延中的占比                     |
| `decode_time_s`         | profiler 中逐 token 生成阶段的累计耗时        |        seconds | autoregressive decode 的总成本 | 用来验证长推理模型的主要时间是否集中在 decode 阶段               |
| `total_latency_s`       | `prefill_time_s + decode_time_s`   |        seconds | profiler 视角下的总生成时延         | 用来和端到端 `latency_s` 对照，检查拆解是否合理              |
| `decode_steps_measured` | profiler 实际记录的 decode step 数       | steps / tokens | 实际测到多少轮逐 token 生成          | 用来确认 profiler 覆盖范围是否完整                      |
| `prefill_top_ops`       | prefill 阶段按设备时间排序的主要算子列表           |           list | prefill 期间主要消耗在哪些算子上       | 用来分析 prefill 的热点，例如 attention、linear、matmul |
| `decode_top_ops`        | decode 阶段按设备时间排序的主要算子列表            |           list | decode 期间主要消耗在哪些算子上        | 用来定位 decode 热点，并进一步分析 LM head 相关算子是否显著      |
| `prefill_trace_file`    | prefill 的 Chrome trace 文件路径        |      file path | 可视化时间线证据                   | 用来在浏览器里打开 trace，查看具体时间线和算子调用顺序              |
| `decode_trace_file`     | decode 的 Chrome trace 文件路径         |      file path | 可视化 decode 时间线证据           | 用来检查 decode 循环中的热点位置和重复模式                   |
| `generated_text`        | 模型生成的原始文本                          |           text | 输出内容本身                     | 用来检查是否自然结束、是否包含长推理链 `<think>`、是否存在异常输出      |

如果你要，我下一条可以直接把这个表格改成更适合放进论文正文的英文版。
