记录开发信息


# 2026.04.16 周四
今日目标：
1. 知识相关：
- 理解transformer的推理过程
- LLM和LRM的区别，知道一个最简的LLM的推理流程
- 从代码层面理解LLM的每个阶段（tokenization, embedding, transformer, sampling, next token）以及它们的计算特点     

2. 实验相关：
开发环境：minerva集群，GPU型号为NVIDIA L40s，远程实验环境为mt，基础模型为Qwen2.5-1.5b，推理精度设置为BF16,Torch version: 2.9.0+cu128
开发进度：
    1. 完成了baseline的搭建，使用Qwen2.5-1.5b模型进行推理测试，并记录了输入输出的token数量、总推理时间、每token的平均推理时间（TPOT）以及生成的文本内容
    2. 结果已保存为JSON格式，方便后续分析和比较
    
第一次batch测试
    1. 自定义prompts测试集，保存在/prompts/perf_prompts.json中，包含了三种类型的推理任务，
    - A组短输出、低推理负载
    - B组中等输出、适中推理负载，有 chain-of-thought 倾向，输出长度中等
    - C组长输出、高推理负载，明显的 chain-of-thought 倾向，输出长度较长，会诱导更长解释
    2.结果
     发现需要设置warm up步骤，第一次推理的结果不稳定，后续需要在正式测试前进行warm up，以获得更稳定的性能数据
     因为存在max_new_tokens的限制，部分prompts的输出被截断了，后续需要调整max_new_tokens的设置，以确保完整的输出结果
     机器自己开始输出，你是一个AI。。。不是单纯在回答你的问题，而像是在续写某种训练语料/指令模板

第二次测试：
    warmp up步骤已经设置，推理结果更稳定了；不会出现A1需要大量时间；
    但是max_new_tokens的限制仍然存在，部分prompts的输出被截断了，
    初步结果，看 v2 的 tpot_ms：
A1: 13.63
A2: 13.66
A3: 13.38
B1: 13.39
B2: 10.45
B3: 9.79
C1: 9.81
C2: 9.76
C3: 9.74
你会发现一个很清楚的结构：

第一段：短输出 / 较短生成

大约在 13.3–13.7 ms/token

第二段：长生成 / 接近 256 token

大约在 9.7–10.5 ms/token

这说明：

随着生成变长，平均 TPOT 下降并趋于稳定；

说明，可能前几个token的生成需要更多的计算资源，随着生成的进行，模型可能会进入一个更高效的状态，或者是因为前几个token的生成涉及到更多的上下文处理，而后续token的生成则相对简单。

第三次测试，把max_new_tokens调整为512，结果如下：
TOPT_MS:
短输出
A1: 15.83
A2: 15.83
A3: 15.47
中长 reasoning
B1: 15.54
B2: 15.47
B3: 14.49
长 reasoning
C1: 9.83
C2: 9.87
C3: 9.84

最重要的是最后三条：

C1: 9.834
C2: 9.875
C3: 9.839

```
层 1：短输出场景

A1/A2/A3/B1/B2 这类样本，TPOT 大约在：

14.5–15.8 ms/token

这部分更容易受到固定启动成本影响。

层 2：长输出 steady-state 场景

C1/C2/C3 这类长推理样本，TPOT 大约在：

9.8 ms/token

这更接近模型持续 decode 时的真实速度。
```

**小结**：
通过调整 max_new_tokens 的设置，观察到不同长度输出的 TPOT 变化，短输出的 TPOT 较高，长输出的 TPOT 较低，这可能是由于模型在生成初期需要处理更多的上下文信息，而在生成后期进入一个更高效的状态。长输出场景下，模型的 steady-state TPOT 仍稳定在约 9.8 ms/token，表明 decode 性能具有较高一致性




# 2026.04.17 周五
## 今日目标：
1. 知识相关：
- 继续了解LLM的具体推理过程，特别是Transformer阶段

2. 实验相关：
- 完成阶段1的测试，补完整为“baseline + profiling”，主要测LM head的时间占比，看看是否值得优化

## 今日进度：
### 1.一些思考与更新
1. 替换模型为Qwen3-1.7b小模型，因为这个模型更加符合LRM的定义
2. 测试的思路更新为：LRM的推理过程，就是会“长输出”+“autoregressive decode 重复很多轮”；而非昨天GPT给我的思路，先去验证短回答，长回答谁更慢，这个对本身论文没有什么意义；
- **LRM就是会token很多，我们做的事情就是加速这个token的生成；所以在LRM上去加速这个过程，这个是整个论文的意义，是非常solid的**
- 测试的第一步应该是我们找的方法，或者说是LM head对应的这个阶段是占比多少，得到一个基准线
- 占比高，意义更大；占比低也没有关系，仍然可以优化（蚊子再小也是肉，而且当基数很大时，优化的绝对收益仍然可观）

### 2. 思路实验整理
1. 不设置max_new_tokens的限制（仍然建议保留一个很大的 safety cap，防止无限续写和异常回答，比如9999），模拟真实生成场景下的端到端测量
2. 固定 prompt 集，测试端到端指标(latency，token numbers，TPOT)，并用 profiler 分析每个阶段的时间占比，特别关注 LM head 的占比
3. 阶段拆分：
   - Tokenization
   - Embedding
   - Transformer
   - LM Head
   - Sampling
   - Autoregressive Loop

### 3.实际做的事情
1. 先做端到端测试，因为profiling本身会拖慢推理的速度，先直接模拟一个实际生产，无干扰的环境下的性能水平
2. 抽样进行profiler测试，先大致看看水平