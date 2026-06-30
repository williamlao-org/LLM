"""Phase 2.04 指标总入口。

指标按 RAG 数据流排序：先评估 Retrieval，再评估 Generation。

当前实现的是经典四指标最小集合，而不是“所有可用指标”：

Retrieval（检索质量）
    1. Context Precision：相关 chunk 是否排在前面；
    2. Context Recall：参考答案所需事实是否被召回。

Generation（生成质量）
    3. Faithfulness：答案陈述是否受到 contexts 支持；
    4. Answer Relevancy：答案是否真正回答了 question。

暂未实现的常见扩展指标：
    - Context Relevancy：衡量 context 与 question 的相关程度，与 Context Precision
      有一定重叠；
    - Answer Correctness：比较 answer 与 ground_truth 的事实正确性，需要参考答案。

不存在“检索和生成必须各三个指标”的统一标准，应根据评估目标、标注数据和
调用成本选择。此学习项目先用对称的 2 + 2 覆盖 RAG 两个核心阶段。

Prompt 通过 ``prompt_profile`` 独立切换：
    - official：RAGAS 0.4.3 官方 Prompt 的版本化快照（默认）；
    - custom_zh：项目自有中文 Prompt 与注入防护实验版本。

两套 Profile 只改变 Prompt、few-shot 和输出 Schema 描述，不改变指标输入、
数学公式、执行顺序或 Trace 结构。
"""

from __future__ import annotations

from typing import Any

from phase2_04_metric_common import (
    METRIC_LABELS,
    METRIC_TIPS,
    EvaluatorMetric,
)
from phase2_04_retrieval_metrics import (
    ContextPrecisionMetric,
    ContextRecallMetric,
)
from phase2_04_generation_metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from phase2_04_prompt_profiles import (
    DEFAULT_PROMPT_PROFILE,
    PromptProfile,
    get_prompt_profile,
)


def build_metrics(
    llm: Any,
    embeddings: Any,
    prompt_profile: str | PromptProfile = DEFAULT_PROMPT_PROFILE,
) -> dict[str, EvaluatorMetric]:
    """按 Retrieval → Generation 顺序构建同一 Prompt Profile 的指标。"""
    profile = (
        get_prompt_profile(prompt_profile)
        if isinstance(prompt_profile, str)
        else prompt_profile
    )
    metrics: list[EvaluatorMetric] = [
        # Retrieval metrics：RAG 先检索
        ContextPrecisionMetric(llm, profile),
        ContextRecallMetric(llm, profile),
        # Generation metrics：再基于检索上下文生成答案
        FaithfulnessMetric(llm, profile),
        AnswerRelevancyMetric(llm, embeddings, prompt_profile=profile),
    ]
    return {metric.name: metric for metric in metrics}


__all__ = [
    "METRIC_LABELS",
    "METRIC_TIPS",
    "EvaluatorMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "DEFAULT_PROMPT_PROFILE",
    "PromptProfile",
    "get_prompt_profile",
    "build_metrics",
]
