项目相关的基础知识


## LLM的生成过程（逻辑流程）：

1. tokenization：将输入文本转换为token序列,纯预处理
2. embedding：将token序列转换为向量表示，每个token对应一个向量
3. transformer：前向传播计算，这是最核心的部分，包含多层transformer block，每一层都有self-attention和feedforward网络
- 输入：当前 token + KV cache（包括之前生成的token）
- 输出：每个token对应的输出向量最后一层 hidden state 
4. LM head：将transformer的输出向量映射到词表大小的维度(把“语义表示”映射到“整个词表概率空间”)，得到每个token的概率分布，这部分通常是一个线性层;当词表很大的时候，这部分计算会非常耗时，因此可以通过一些优化方法（如分层softmax、采样等）来加速计算
5. Sampling：进行采样动作，根据概率分布进行采样，选择下一个token；常用的采样方法包括greedy decoding（选择概率最高的token）、beam search（保留多个候选序列）和top-k/top-p采样（限制候选token的数量或概率总和）等；采样方法的选择会影响生成文本的质量和多样性
6. Autoregressive Loop：将生成的token添加到输入序列中，形成新的输入，进入下一轮循环，直到生成结束；重复 Step 3 → Step 5

- 输入（step1，2）->思考（step3）->表达（step4）->选择（step5）->输出（step6）->循环，直到生成结束
- 停止条件，可以是生成特定的结束token，或者达到最大生成长度，或者满足某些特定的条件（如生成的文本符合某种格式）


```
[Input Text]
     ↓
(1) Tokenization
     ↓
[Token IDs]
     ↓
(2) Embedding
     ↓
[Token Embeddings]
     ↓
(3) Transformer (N layers)
     ↓
[Hidden State h_t]
     ↓
(4) LM Head / Classification Head   ← ⭐ FlashHead优化这里
     ↓
[Logits over Vocabulary]
     ↓
(5) Sampling / Decoding
     ↓
[Next Token]
     ↓
(6) Autoregressive Loop
     ↺ (回到 Transformer，继续生成)
```
| 阶段                         | 输入                          | 输出                 | 作用                | 是否计算瓶颈         |
| -------------------------- | --------------------------- | ------------------ | ----------------- | -------------- |
| **1. Tokenization**        | 原始文本                        | token IDs          | 把文本转成模型可处理的离散符号   | ❌              |
| **2. Embedding**           | token IDs                   | 向量（embedding）      | 把离散 token 映射到连续空间 | ❌              |
| **3. Transformer**         | embedding + 历史上下文（KV cache） | hidden state (h_t) | 理解上下文，建模语义关系      | ✅（主要计算）        |
| **4. LM Head（分类头）**        | hidden state (h_t)          | logits（词表大小 V）     | 计算每个 token 的概率    | ✅（FlashHead优化） |
| **5. Sampling**            | logits                      | 下一个 token          | 选择/采样输出 token     | ❌              |
| **6. Autoregressive Loop** | 新 token + 历史序列              | 新一轮输入              | 递归生成序列            | ✅（放大成本）        |

其中：
- hidden state 是对当前上下文的压缩语义表示，是对“当前序列 + 历史上下文”计算出来的一个向量，用于计算下一个 token 的概率分布；可以说Transformer负责“思考”，这个hidden state就是思考的结果
- LM head负责“表达”，把思考的结果映射到具体的词表概率空间，决定“说什么”；LM head 先输出 logits，再通过 softmax 变成概率分布
- sampling负责“选择”，根据概率分布选择下一个 token；已经有了概率分布，这个阶段负责采用某种方法，比如greedy（最高概率），top-k / top-p / temperature sampling，进行token的选择；这一步结束后，新的token就生成了，进入下一轮循环

## LLM的执行过程（prefill/decode）：
这种划分方式，主要是为了区分模型在预填充（prefill）和解码（decode）阶段的不同工作方式。
- prefill只发生一次，主要是处理输入文本，生成初始的hidden state，特点是没有新的token产生；
- decode阶段则是一个循环过程，每次生成一个token，并将其添加到输入序列中，继续生成下一个token。本质上还是逻辑过程，只是从计算的角度来划分不同阶段的工作内容和计算特点。


| 步骤           | Prefill | Decode    |
| -------------- | ------- | --------- |
| Tokenization   | ✅       | ❌         |
| Embedding      | ✅       | ❌（或极少）    |
| Transformer    | ✅（全序列）  | ✅（逐token） |
| LM head        | ❌       | ✅         |
| Sampling       | ❌       | ✅         |
| Autoregressive | ❌       | ✅         |

- 这个概念实际上不是特别重要，只是因为在论文里强调decode阶段的不停重复，生成N个tokens需要N次重复，因此在LRM里，这个过程会极其庞大，导致推理成本过高；因此需要优化。


## LRM VS LLM
- LRM（Large Reasoning Model）是专门针对推理任务优化的大型模型，通常在架构和训练数据上进行了调整，以更好地处理复杂的推理问题
- LLM（Large Language Model）是更通用的大型语言模型，虽然也可以用于推理任务，但可能在某些特定的推理场景下表现不如LRM
其中LLM为单步推理，没有显式的推理过程，而LRM则可能包含多步推理过程，能够更好地处理需要多轮推理的复杂任务，最显著的区别就是“推理链（Chain-of-Thought）”，因此会产生较多的token输出，导致推理成本过高，好处是推理链会帮助模型理解复杂问题，自我完成推理过程，提升推理质量。

比如，一个问题“如果小明有3个苹果，吃掉了2个，还剩几个？”对于LLM来说，它可能直接输出“1”（基于next token的预测），而LRM则可能先生成一个推理链：

1. 小明有3个苹果
2. 吃掉了2个苹果
3. 还剩下1个苹果

答案是1




## Transformer的本质：
Transformer是一种基于自注意力机制的神经网络架构，本质是通过self-attention机制来捕捉输入序列中不同位置之间的关系，从而实现上下文理解和生成。

