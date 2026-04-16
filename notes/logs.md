记录开发信息


2026.04.16 周四
今日目标：
1. 知识相关：
- 理解transformer的推理过程
- LLM和LRM的区别，知道一个最简的LLM的推理流程
- 从代码层面理解LLM的每个阶段（tokenization, embedding, transformer, sampling, next token）以及它们的计算特点     

2. 实验相关：
开发环境：minerva集群，GPU型号为NVIDIA L40s，远程实验环境为mt，基础模型为Qwen2.5-1.5b，推理精度设置为BF16。
开发进度：
    1. 完成了baseline的搭建，使用Qwen2.5-1.5b模型进行推理测试，并记录了输入输出的token数量、总推理时间、每token的平均推理时间（TPOT）以及生成的文本内容
    2. 结果已保存为JSON格式，方便后续分析和比较