"""RAG 人工评估数据集。

样本基于知识库文档 transformer.md、rag_overview.md 和 agent.md 编写。
"""

from phase2_04_models import EvaluationSample


EVAL_SAMPLES: list[EvaluationSample] = [
    # Transformer
    EvaluationSample(
        question="Transformer 的自注意力机制是怎么计算的？",
        ground_truth=(
            "自注意力机制将输入向量分别线性变换得到 Query、Key、Value 三个矩阵，"
            "然后计算注意力得分 Score = Q × K^T / √d_k，对得分做 Softmax 归一化得到注意力权重，"
            "最后用注意力权重对 V 加权求和得到输出。除以 √d_k 是为了防止点积值过大导致梯度消失。"
        ),
    ),
    EvaluationSample(
        question="多头注意力（Multi-Head Attention）相比单头注意力有什么优势？",
        ground_truth=(
            "多头注意力将输入分成多个“头”，每个头独立进行自注意力计算，再将所有头的输出拼接。"
            "好处是让模型能从不同角度关注信息，例如一个头关注语法关系，另一个头关注语义关系，"
            "从而捕获更丰富的特征。"
        ),
    ),
    EvaluationSample(
        question="Transformer 为什么需要位置编码？",
        ground_truth=(
            "Transformer 没有循环或卷积结构，无法自动感知输入序列的顺序，"
            "因此需要额外添加位置编码来注入位置信息。"
            "原始论文使用正弦/余弦函数生成固定位置编码，后续模型也有可学习位置编码或 RoPE 等方案。"
        ),
    ),
    # RAG
    EvaluationSample(
        question="RAG 是什么？它解决了哪些核心问题？",
        ground_truth=(
            "RAG（检索增强生成）是将信息检索与文本生成相结合的技术。"
            "它在 LLM 生成前先从外部知识库检索相关信息作为上下文，"
            "主要解决了：幻觉问题（减少 LLM 编造）、知识时效性（可检索最新信息）、"
            "私有知识访问（企业内部文档等）以及答案可溯源性。"
        ),
    ),
    EvaluationSample(
        question="RAG 的文本分块有哪些常见策略，各有什么优缺点？",
        ground_truth=(
            "常见分块策略有三种：1）固定大小分块：实现简单，块大小均匀，但可能在语义中间切断；"
            "2）基于分隔符的分块：按段落、章节等自然边界切分，语义完整性好，但块大小不均匀；"
            "3）语义分块：用 Embedding 模型计算相邻句子的语义相似度，在语义变化处切分，"
            "语义完整性最好，但计算成本高、实现复杂。"
        ),
    ),
    EvaluationSample(
        question="向量数据库为什么要用近似最近邻搜索（ANN）而不是精确搜索？",
        ground_truth=(
            "精确搜索（暴力搜索）需要遍历所有向量计算距离，在大规模数据集（如百万级向量）下速度极慢。"
            "ANN 算法（如 HNSW、IVF）以牺牲极少精度换取大幅提升的搜索速度，"
            "能在毫秒级完成百万级向量检索，适合生产环境。"
        ),
    ),
    # Agent
    EvaluationSample(
        question="ReAct 架构的工作流程是什么？",
        ground_truth=(
            "ReAct 让模型交替进行推理（Reasoning）和行动（Acting）。"
            "工作流程为：Thought（模型思考应该做什么）→ Action（选择并调用工具）"
            "→ Observation（获取工具返回结果），循环以上步骤直到得到最终答案。"
            "ReAct 的优势是推理过程可见（chain-of-thought），便于调试。"
        ),
    ),
    EvaluationSample(
        question="AI Agent 的记忆系统分哪几类？",
        ground_truth=(
            "Agent 记忆系统分为短期记忆和长期记忆。"
            "短期记忆是当前对话的上下文窗口，管理策略包括滑动窗口、摘要压缩、Token 截断。"
            "长期记忆是跨会话持久化的信息，存储在外部数据库中，包括语义记忆（事实知识）、"
            "情景记忆（历史经历）和程序性记忆（如何完成任务的技能）。"
        ),
    ),
    EvaluationSample(
        question="MCP（Model Context Protocol）是什么，有什么意义？",
        ground_truth=(
            "MCP 是 Anthropic 在 2024 年底推出的开放协议，用于标准化 AI 模型与外部工具/数据源的通信。"
            "核心概念包括：Server（提供工具的服务端）、Client（集成在 Agent 中）、"
            "Tools（可调用功能）、Resources（只读数据源）、Prompts（模板）。"
            "其意义在于：一次编写 Server，所有支持 MCP 的 Agent 都可以使用，实现工具的标准化和可复用。"
        ),
    ),
    EvaluationSample(
        question="多 Agent 系统有哪些协作模式？",
        ground_truth=(
            "多 Agent 协作模式包括：主从模式（主 Agent 分配任务，从 Agent 执行）、"
            "辩论模式（多个 Agent 提出不同观点，讨论达成共识）、"
            "流水线模式（Agent 按顺序处理，每个负责一个阶段）、"
            "层级模式（类似公司架构，层层分解和汇报）。"
        ),
    ),
]
