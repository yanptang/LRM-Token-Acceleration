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
            │ Transformer (N层)  │ ← 你优化的核心区域
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
                 Loop 🔁  ← 你重点优化（decode）