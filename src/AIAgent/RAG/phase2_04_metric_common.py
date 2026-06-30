"""检索指标与生成指标共享的基础设施。"""

from __future__ import annotations

from typing import Any, Protocol

from phase2_04_models import MetricScore, RAGOutput


# 字典顺序决定 CLI 报告顺序：RAG 先检索，后生成。
METRIC_LABELS = {
    # Retrieval metrics
    "context_precision": "Context Precision（上下文精确率）",
    "context_recall": "Context Recall（上下文召回率）",
    # Generation metrics
    "faithfulness": "Faithfulness（忠实度）",
    "answer_relevancy": "Answer Relevancy（答案相关性）",
}

METRIC_TIPS = {
    "context_precision": "靠前召回的 chunk 是否真正有用，越高表示排序信噪比越好",
    "context_recall": "参考答案所需信息有多少存在于召回上下文中，越高表示漏捞越少",
    "faithfulness": '回答中的陈述是否都能由召回上下文支持，越高越不易“幻觉”',
    "answer_relevancy": "回答反推的问题与原问题是否一致，越高越切题",
}


class EvaluatorMetric(Protocol):
    """所有自定义指标统一实现这个最小接口。"""

    name: str
    prompt_profile: Any

    async def score(self, sample: RAGOutput) -> MetricScore | None: ...
