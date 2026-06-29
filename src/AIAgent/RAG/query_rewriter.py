"""
Query Rewriting —— 查询优化层

在用户的原始问题送进检索之前，先用 LLM 对它做优化。
支持三种策略：

┌─────────────────┬──────────────────────────────────────────────────────────┐
│ 策略             │ 核心思想                                                  │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ rewrite         │ LLM 把口语/模糊问题改写成更适合向量检索的规范表述          │
│ hyde            │ 先让 LLM 生成「假设答案」，用假设答案的 embedding 去检索   │
│ multi_query     │ 把复杂问题拆成 N 个子查询，分别检索后 RRF 合并结果         │
└─────────────────┴──────────────────────────────────────────────────────────┘

为什么需要查询优化？

用户的提问和知识库里文档的表达方式往往不同——

  用户问：「这东西怎么用？」
  文档写：「XXX 的使用方法」或「如何配置 XXX」

向量模型虽然能处理语义相似，但「问题空间」和「答案空间」之间仍然存在
天然的「词汇鸿沟（vocabulary gap）」和「语义漂移」。

三种策略各解决一个痛点：
  - rewrite   → 表达不规范（口语、缩写、歧义）
  - hyde      → 问题向量和答案向量天然不同（问题是短疑问句，文档是长陈述句）
  - multi_query → 一个问题含多个子维度，单条查询只能命中其中一个
"""

from openai import OpenAI


# ========== Prompt 模板 ==========

_REWRITE_SYSTEM = """你是一个检索查询优化助手。
你的任务是把用户的口语化/模糊问题，改写成更适合向量语义检索的规范化查询语句。

改写规则：
1. 保留核心意图，不要改变问题的本质含义
2. 去掉语气词和口语表达（「我想知道」「能不能告诉我」等）
3. 补全缩写和代词（「它」「这个」→ 具体名词）
4. 使用更专业、文档化的表述风格
5. 只输出改写后的查询语句，不加任何解释"""

_REWRITE_USER = "原始问题：{query}\n\n改写后的检索查询："

# -------

_HYDE_SYSTEM = """你是一个知识渊博的助手。
用户会给你一个问题，请你生成一个简洁、准确的「假设性回答」。

要求：
1. 回答要像真实文档中的段落，而不是对话风格
2. 使用陈述句，语言要规范专业
3. 长度控制在 2-4 句话（约 100-200 字）
4. 如果你不确定答案，也要生成一个合理的、相关领域的假设性回答
5. 只输出回答内容，不加「根据你的问题」等前缀"""

_HYDE_USER = "问题：{query}\n\n假设性回答："

# -------

_MULTI_QUERY_SYSTEM = """你是一个查询分解专家。
你的任务是把用户的复杂问题分解成多个独立的子查询，用于分别检索不同角度的相关信息。

分解规则：
1. 每个子查询要能独立检索到有意义的内容
2. 子查询之间要覆盖原始问题的不同方面
3. 每个子查询简洁明了，一句话
4. 只输出子查询列表，每行一条，不加编号、不加解释"""

_MULTI_QUERY_USER = "原始问题：{query}\n\n请分解成 {n} 个子查询（每行一条）："


class QueryRewriter:
    """
    查询改写器

    三种策略可单独调用，也可通过 rewrite_query() 统一入口调用。

    使用示例：
        rewriter = QueryRewriter(llm_client, model="deepseek-chat")

        # 策略1：直接改写
        better_q = rewriter.rewrite("这玩意儿咋用？")

        # 策略2：HyDE
        hyp_doc = rewriter.hyde("Transformer 的自注意力有什么特点？")

        # 策略3：多查询
        sub_qs = rewriter.multi_query("RAG 和 Fine-tuning 有什么区别，各适合什么场景？", n=3)
    """

    def __init__(self, llm_client: OpenAI, model: str):
        """
        Args:
            llm_client: 已初始化的 OpenAI 客户端（兼容任意 OpenAI 格式 API）
            model:      使用的 LLM 模型名称
        """
        self.client = llm_client
        self.model = model

    # ========== 私有辅助 ==========

    def _call_llm(self, system: str, user: str) -> str:
        """调用 LLM，返回纯文本回复"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    # ========== 三种策略 ==========

    def rewrite(self, query: str) -> str:
        """
        策略1：Query Rewriting（直接改写）

        把口语化、模糊的问题改写成更适合向量检索的规范表述。

        例：
            原始：「这个 transformer 注意力的那个公式怎么来的」
            改写：「Transformer 自注意力机制 Scaled Dot-Product Attention 公式推导」

        Args:
            query: 原始用户问题

        Returns:
            改写后的检索查询字符串
        """
        return self._call_llm(
            _REWRITE_SYSTEM,
            _REWRITE_USER.format(query=query),
        )

    def hyde(self, query: str) -> str:
        """
        策略2：HyDE（Hypothetical Document Embeddings）

        不用问题本身去检索，而是让 LLM 先生成一个「假设性答案文档」，
        再用这个假设文档的 embedding 去检索真实文档。

        为什么有效？
          - 问题："X 是什么？" → 短疑问句向量
          - 知识库里写的是："X 是……（长陈述句）"
          - 假设答案的向量和真实答案的向量更相近（都是陈述句，语义空间对齐）

        Args:
            query: 原始用户问题

        Returns:
            假设性回答文本（用于替代原始 query 进行 embedding 检索）
        """
        return self._call_llm(
            _HYDE_SYSTEM,
            _HYDE_USER.format(query=query),
        )

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """
        策略3：Multi-Query（多查询分解）

        把一个复杂问题拆成 N 个子查询，每个子查询独立检索，
        最后用 RRF 把多路结果合并，取并集。

        适用场景：
          - 问题含多个维度：「A 和 B 有什么区别，各适合什么场景？」
          - 问题较长且复杂：可能只有某个子角度能匹配到文档

        Args:
            query: 原始用户问题
            n:     分解成几个子查询（默认 3）

        Returns:
            子查询字符串列表，长度 ≤ n（LLM 可能少给几条）
        """
        raw = self._call_llm(
            _MULTI_QUERY_SYSTEM,
            _MULTI_QUERY_USER.format(query=query, n=n),
        )
        # 按行分割，过滤空行，去除可能的编号前缀（"1. " "- " 等）
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # 去掉常见的列表前缀
            for prefix in ["- ", "* ", "• "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            # 去掉 "1. " "2. " 等编号前缀
            if len(line) > 2 and line[0].isdigit() and line[1] in (".", "、", ")"):
                line = line[2:].strip()
            elif len(line) > 3 and line[:2].isdigit() and line[2] in (".", "、", ")"):
                line = line[3:].strip()
            if line:
                lines.append(line)
        # 最多取 n 条
        return lines[:n]

    def rewrite_query(
        self,
        query: str,
        strategy: str = "rewrite",
        multi_query_n: int = 3,
    ) -> str | list[str]:
        """
        统一入口：根据 strategy 调用对应策略

        Args:
            query:        原始用户问题
            strategy:     改写策略，可选 "rewrite" | "hyde" | "multi_query"
            multi_query_n: multi_query 策略时分解的子查询数

        Returns:
            - "rewrite" / "hyde"  → str（单条改写结果）
            - "multi_query"       → list[str]（子查询列表）

        Raises:
            ValueError: 未知的 strategy
        """
        if strategy == "rewrite":
            return self.rewrite(query)
        elif strategy == "hyde":
            return self.hyde(query)
        elif strategy == "multi_query":
            return self.multi_query(query, n=multi_query_n)
        else:
            raise ValueError(
                f"未知的 query_rewrite 策略: {strategy!r}，"
                f"可选值为 'rewrite' | 'hyde' | 'multi_query'"
            )


# ===== 测试 =====
if __name__ == "__main__":
    from config import config
    from openai import OpenAI

    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    rewriter = QueryRewriter(client, model=config.llm_model)

    test_queries = [
        "这个 transformer 注意力的那个公式怎么来的",
        "RAG 是啥，怎么用？",
        "ReAct 和普通 Agent 有啥区别，分别适合啥场景？",
    ]

    print("=" * 60)
    print("策略1：Query Rewriting（直接改写）")
    print("=" * 60)
    for q in test_queries[:2]:
        rewritten = rewriter.rewrite(q)
        print(f"\n原始：{q}")
        print(f"改写：{rewritten}")

    print("\n" + "=" * 60)
    print("策略2：HyDE（假设答案）")
    print("=" * 60)
    q = test_queries[1]
    hyp = rewriter.hyde(q)
    print(f"\n原始问题：{q}")
    print(f"假设答案：\n{hyp}")

    print("\n" + "=" * 60)
    print("策略3：Multi-Query（多子查询）")
    print("=" * 60)
    q = test_queries[2]
    subs = rewriter.multi_query(q, n=3)
    print(f"\n原始问题：{q}")
    print("分解为：")
    for i, sq in enumerate(subs, 1):
        print(f"  [{i}] {sq}")
