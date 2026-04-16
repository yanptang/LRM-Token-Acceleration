### Projest Structure
scripts/：python scripts

prompts/：测试输入的promt

results/：实验输出

notes/：记录你开发信息和日志

documents/：存放相关文献，前置知识

models/：存放模型文件

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
