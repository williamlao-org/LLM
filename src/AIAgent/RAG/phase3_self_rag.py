"""
Self-RAG —— 检索结果质量评估

Agentic RAG 的核心能力之一：Agent 检索到东西后，不是盲目塞进 Prompt，
而是先「照一下镜子」—— 让 LLM 评估检索结果的质量，再决定下一步怎么走。

这就是 Self-RAG（Self-Reflective RAG）和 CRAG（Corrective RAG）的核心思想：

  ┌───────────────────────────────────────────────────────────────┐
  │                 检索结果质量评估流程                              │
  │                                                               │
  │   检索结果 + 用户问题                                           │
  │       │                                                       │
  │       ▼                                                       │
  │   LLM 评估两个维度:                                             │
  │   ┌─────────────┐    ┌──────────────┐                         │
  │   │ Relevance   │    │ Sufficiency  │                         │
  │   │ (相关性)     │    │ (充分性)      │                         │
  │   └──────┬──────┘    └──────┬───────┘                         │
  │          │                  │                                  │
  │          ▼                  ▼                                  │
  │   ┌─────────────────────────────────┐                         │
  │   │ 综合决策: 下一步怎么走？          │                         │
  │   │                                 │                         │
  │   │  answer    → 质量好，直接回答     │                         │
  │   │  refine    → 部分相关，改写重搜    │                         │
  │   │  fallback  → 完全不行，放弃检索    │                         │
  │   └─────────────────────────────────┘                         │
  └───────────────────────────────────────────────────────────────┘

Self-RAG vs CRAG 的区别：
  - Self-RAG (Asai et al., 2023): 在生成过程中插入「反思 token」，
    模型边生成边评估自己的检索需求和生成质量。需要微调模型。
  - CRAG (Yan et al., 2024): 不需要微调。用一个独立的评估步骤判断
    检索质量，质量差就触发纠正动作（改写查询、补充检索、甚至放弃检索）。

我们这里实现的是 CRAG 的思路（不微调，用独立评估步骤），
但在概念上统称为 Self-RAG（更直觉：Agent 自我评估检索质量）。
"""

import json
from dataclasses import dataclass
from openai import OpenAI


# ========== 评估结果数据结构 ==========


@dataclass
class RetrievalAssessment:
    """检索质量评估结果"""

    # --- 两个评估维度 ---
    relevance: str  # "relevant" | "partially_relevant" | "irrelevant"
    sufficiency: str  # "sufficient" | "insufficient" | "conflicting"

    # --- 综合决策 ---
    action: str  # "answer" | "refine" | "fallback"
    reason: str  # LLM 给出的评估理由
    suggested_query: str | None = None  # action == "refine" 时，建议的改写查询

    @property
    def should_answer(self) -> bool:
        return self.action == "answer"

    @property
    def should_refine(self) -> bool:
        return self.action == "refine"

    @property
    def should_fallback(self) -> bool:
        return self.action == "fallback"


# ========== 评估 Prompt ==========

_ASSESS_SYSTEM = """你是一个检索质量评估专家。你的任务是评估给定的检索结果是否足以回答用户的问题。

你需要从两个维度评估：

## 1. Relevance（相关性）
检索到的内容和用户问题是否相关？
- "relevant": 检索结果与问题高度相关，包含直接回答问题所需的信息
- "partially_relevant": 检索结果与问题部分相关，有些有用信息但不够直接
- "irrelevant": 检索结果与问题完全无关

## 2. Sufficiency（充分性）
检索到的内容是否足够回答问题？
- "sufficient": 信息充分完整，可以直接据此回答
- "insufficient": 信息不够完整，需要补充更多信息
- "conflicting": 检索结果之间存在矛盾，需要更多来源验证

## 综合决策

根据两个维度的评估，给出下一步动作：
- "answer": relevance 至少为 partially_relevant 且 sufficiency 为 sufficient → 可以直接回答
- "refine": relevance 为 partially_relevant 或 sufficiency 为 insufficient → 需要改写查询重新检索
- "fallback": relevance 为 irrelevant → 放弃检索，用模型自身知识回答或告知无法回答

如果 action 是 "refine"，还需要给出一个改写后的查询建议（suggested_query）。

请以 JSON 格式输出：
{
    "relevance": "relevant|partially_relevant|irrelevant",
    "sufficiency": "sufficient|insufficient|conflicting",
    "action": "answer|refine|fallback",
    "reason": "你的评估理由（一两句话）",
    "suggested_query": "改写后的查询（仅 action=refine 时提供，否则为 null）"
}

只输出 JSON，不加任何其他内容。"""

_ASSESS_USER = """用户问题：{question}

检索结果（共 {count} 条）：
{context}

请评估以上检索结果的质量。"""


# ========== 评估器 ==========


class SelfRAGAssessor:
    """
    Self-RAG 检索质量评估器

    接收用户问题和检索结果，让 LLM 评估质量并给出下一步决策。

    使用示例：
        assessor = SelfRAGAssessor(llm_client, model="deepseek-chat")
        assessment = assessor.assess(question, retrieved_chunks)
        if assessment.should_answer:
            # 质量好，直接用检索结果回答
        elif assessment.should_refine:
            # 质量差，用 assessment.suggested_query 重新检索
    """

    def __init__(self, llm_client: OpenAI, model: str):
        self.llm_client = llm_client
        self.model = model

    def assess(
        self,
        question: str,
        results: list,  # list[SearchResult]
        verbose: bool = True,
    ) -> RetrievalAssessment:
        """
        评估检索结果的质量。

        Args:
            question: 用户的原始问题
            results: 检索结果列表（SearchResult 对象）
            verbose: 是否打印评估过程

        Returns:
            RetrievalAssessment 评估结果
        """
        if not results:
            return RetrievalAssessment(
                relevance="irrelevant",
                sufficiency="insufficient",
                action="fallback",
                reason="检索结果为空，没有找到任何相关信息。",
            )

        # 把检索结果格式化为文本
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("source", "未知来源")
            score = r.score
            context_parts.append(
                f"[{i}] (相关度: {score:.4f}) [来源: {source}]\n{r.chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_prompt = _ASSESS_USER.format(
            question=question,
            count=len(results),
            context=context,
        )

        if verbose:
            print("     🔍 Self-RAG: 评估检索质量...")

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _ASSESS_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # 评估任务用低温度，保持稳定
        )

        raw = (response.choices[0].message.content or "").strip()

        # 解析 JSON（兼容 markdown 代码块包裹）
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("评估结果必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError):
            # 解析失败时保守处理：直接回答
            if verbose:
                print(f"     ⚠️ 评估结果解析失败，保守处理（直接使用检索结果）")
            return RetrievalAssessment(
                relevance="partially_relevant",
                sufficiency="sufficient",
                action="answer",
                reason="评估结果解析失败，保守使用检索结果。",
            )

        assessment = RetrievalAssessment(
            relevance=data.get("relevance", "partially_relevant"),
            sufficiency=data.get("sufficiency", "sufficient"),
            action=data.get("action", "answer"),
            reason=data.get("reason", ""),
            suggested_query=data.get("suggested_query"),
        )

        if verbose:
            # 用 emoji 直觉显示质量
            rel_icon = {
                "relevant": "✅",
                "partially_relevant": "⚠️",
                "irrelevant": "❌",
            }.get(assessment.relevance, "❓")
            suf_icon = {
                "sufficient": "✅",
                "insufficient": "⚠️",
                "conflicting": "🔀",
            }.get(assessment.sufficiency, "❓")
            act_icon = {
                "answer": "💡",
                "refine": "🔄",
                "fallback": "🚫",
            }.get(assessment.action, "❓")

            print(f"     相关性: {rel_icon} {assessment.relevance}")
            print(f"     充分性: {suf_icon} {assessment.sufficiency}")
            print(f"     决策:   {act_icon} {assessment.action}")
            print(f"     理由:   {assessment.reason}")
            if assessment.suggested_query:
                print(f"     建议查询: {assessment.suggested_query}")

        return assessment
