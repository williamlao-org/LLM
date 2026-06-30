"""Phase 2.04 评估流程使用的显式数据模型。"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """人工准备的一条评估样本。"""

    question: str
    ground_truth: str


@dataclass(frozen=True, slots=True)
class RAGOutput:
    """RAG 系统对一条评估样本的完整输出。"""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str

    @classmethod
    def failed(cls, sample: EvaluationSample) -> "RAGOutput":
        """查询失败时仍保留样本，后续指标会跳过无效输入。"""
        return cls(
            question=sample.question,
            answer="",
            contexts=[],
            ground_truth=sample.ground_truth,
        )


@dataclass(frozen=True, slots=True)
class JudgeStep:
    """一次真实的 LLM-as-Judge 调用及其结构化返回。"""

    name: str
    prompt: str
    output: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MetricScore:
    """某个指标对单条样本的评分，包含完整可审计过程。"""

    metric_name: str
    prompt_profile: str
    value: float
    calculation: str
    judge_steps: list[JudgeStep] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    """一条 RAG 输出对应的全部指标结果。"""

    output: RAGOutput
    scores: dict[str, MetricScore]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """一次 RAG 配置评估的所有逐条结果与聚合方法。"""

    prompt_profile: str
    samples: list[SampleEvaluation]

    def average_scores(self) -> dict[str, float | None]:
        names = {
            name
            for sample in self.samples
            for name in sample.scores
        }
        averages: dict[str, float | None] = {}
        for name in names:
            values = [
                sample.scores[name].value
                for sample in self.samples
                if name in sample.scores and math.isfinite(sample.scores[name].value)
            ]
            averages[name] = statistics.fmean(values) if values else None
        return averages
