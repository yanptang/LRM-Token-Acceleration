项目相关的基础知识


LLM的生成过程：

1. tokenization：将输入文本转换为token序列
2. embedding：将token序列转换为向量表示，每个token对应一个向量
3. transformer：前向传播计算，这是最核心的部分，包含多层transformer block，每一层都有self-attention和feedforward网络
- 输入：当前的token序列（包括之前生成的token）
- 输出：每个token对应的输出向量，最后一层的输出向量会被送到分类头（classification head）进行下一步的计算

4. sampling：根据transformer的输出向量，计算下一个token的概率分布，并采样生成下一个token
5. next token：将生成的token添加到输入序列中，形成新的输入，进入下一轮循环，直到生成结束


            ┌──────────────┐
Input text →│ Tokenization │
            └──────┬───────┘
                   ↓
            ┌──────────────┐
            │ Embedding    │
            └──────┬───────┘
                   ↓
            ┌────────────────────┐
            │ Transformer (N层)  │ ← 优化的核心区域
            └──────┬─────────────┘
                   ↓
            ┌──────────────┐
            │ Sampling     │
            └──────┬───────┘
                   ↓
            ┌──────────────┐
            │ Next Token   │
            └──────┬───────┘
                   ↓
                 Loop 🔁  ← 重点优化（decode）



LRM VS LLM
- LRM（Large Reasoning Model）是专门针对推理任务优化的大型模型，通常在架构和训练数据上进行了调整，以更好地处理复杂的推理问题
- LLM（Large Language Model）是更通用的大型语言模型，虽然也可以用于推理任务，但可能在某些特定的推理场景下表现不如LRM
其中LLM为单步推理，没有显式的推理过程，而LRM则可能包含多步推理过程，能够更好地处理需要多轮推理的复杂任务，最显著的区别就是“推理链（Chain-of-Thought）”，因此会产生较多的token输出，导致推理成本过高，好处是推理链会帮助模型理解复杂问题，自我完成推理过程，提升推理质量。

比如，一个问题“如果小明有3个苹果，吃掉了2个，还剩几个？”对于LLM来说，它可能直接输出“1”（基于next token的预测），而LRM则可能先生成一个推理链：

1. 小明有3个苹果
2. 吃掉了2个苹果
3. 还剩下1个苹果

答案是1





Transformer的本质：

Transformer是一种基于自注意力机制的神经网络架构，本质是通过self-attention机制来捕捉输入序列中不同位置之间的关系，从而实现上下文理解和生成。

