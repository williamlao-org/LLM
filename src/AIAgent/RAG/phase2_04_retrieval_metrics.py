"""检索质量指标：先判断找得准不准，再判断找得全不全。"""

from __future__ import annotations

from typing import Any

from phase2_04_models import JudgeStep, MetricScore, RAGOutput
from phase2_04_prompt_profiles import PromptProfile, get_prompt_profile


# ──────────────────────────────────────────────
# 1. Context Precision：召回结果是否有用且排序靠前
# ──────────────────────────────────────────────


class ContextPrecisionMetric:
    """逐 chunk 判断相关性，再根据原始召回顺序计算 Average Precision。"""

    name = "context_precision"

    def __init__(
        self,
        llm: Any,
        prompt_profile: PromptProfile | None = None,
    ):
        self.llm = llm
        self.prompt_profile = prompt_profile or get_prompt_profile()

    async def score(self, sample: RAGOutput) -> MetricScore | None:
        if not sample.ground_truth or not sample.contexts:
            return None

        judge_steps: list[JudgeStep] = []
        verdicts: list[int] = []
        for index, context in enumerate(sample.contexts):
            prepared = self.prompt_profile.context_precision.prepare(
                question=sample.question,
                reference_answer=sample.ground_truth,
                retrieved_context=context,
            )
            result = await self.llm.agenerate(
                prepared.text,
                prepared.response_model,
            )
            verdicts.append(result.verdict)
            judge_steps.append(
                JudgeStep(
                    name=f"判断召回 chunk #{index + 1} 是否有用",
                    prompt=prepared.text,
                    output=result.model_dump(),
                )
            )

        relevant_count = sum(verdicts)
        if relevant_count == 0:
            value = 0.0
        else:
            precision_at_k = [
                (sum(verdicts[: index + 1]) / (index + 1)) * verdict
                for index, verdict in enumerate(verdicts)
            ]
            value = sum(precision_at_k) / relevant_count

        return MetricScore(
            metric_name=self.name,
            prompt_profile=self.prompt_profile.name,
            value=float(value),
            calculation=f"按召回顺序计算 Average Precision，verdicts={verdicts}",
            judge_steps=judge_steps,
        )


# ──────────────────────────────────────────────
# 2. Context Recall：参考答案所需信息是否被召回
# ──────────────────────────────────────────────


class ContextRecallMetric:
    """检查 ground truth 中的每个事实能否由全部召回上下文支持。"""

    name = "context_recall"

    def __init__(
        self,
        llm: Any,
        prompt_profile: PromptProfile | None = None,
    ):
        self.llm = llm
        self.prompt_profile = prompt_profile or get_prompt_profile()

    async def score(self, sample: RAGOutput) -> MetricScore | None:
        if not sample.ground_truth or not sample.contexts:
            return None

        prepared = self.prompt_profile.context_recall.prepare(
            question=sample.question,
            reference_answer=sample.ground_truth,
            retrieved_contexts=sample.contexts,
            retrieved_context="\n".join(sample.contexts),
        )
        result = await self.llm.agenerate(
            prepared.text,
            prepared.response_model,
        )
        values = [item.attributed for item in result.classifications]
        value = sum(values) / len(values) if values else float("nan")
        return MetricScore(
            metric_name=self.name,
            prompt_profile=self.prompt_profile.name,
            value=float(value),
            calculation=f"已召回事实数 {sum(values)} / 参考答案事实总数 {len(values)}",
            judge_steps=[
                JudgeStep(
                    name="拆分参考答案并判断事实是否被召回",
                    prompt=prepared.text,
                    output=result.model_dump(),
                )
            ],
        )
