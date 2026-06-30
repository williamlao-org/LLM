"""生成质量指标：先检查是否忠于上下文，再检查是否回答了问题。"""

from __future__ import annotations

from typing import Any

import numpy as np

from phase2_04_models import JudgeStep, MetricScore, RAGOutput
from phase2_04_prompt_profiles import PromptProfile, get_prompt_profile


# ──────────────────────────────────────────────
# 3. Faithfulness：生成答案是否忠于召回上下文
# ──────────────────────────────────────────────


class FaithfulnessMetric:
    """先从答案提取事实，再逐条与召回上下文核验。"""

    name = "faithfulness"

    def __init__(
        self,
        llm: Any,
        prompt_profile: PromptProfile | None = None,
    ):
        self.llm = llm
        self.prompt_profile = prompt_profile or get_prompt_profile()

    async def score(self, sample: RAGOutput) -> MetricScore | None:
        if not sample.answer or not sample.contexts:
            return None

        extraction = self.prompt_profile.statement_extraction.prepare(
            question=sample.question,
            answer=sample.answer,
        )
        statements = await self.llm.agenerate(
            extraction.text,
            extraction.response_model,
        )

        judgement = self.prompt_profile.faithfulness_judgement.prepare(
            context="\n".join(sample.contexts),
            statements=statements.statements,
        )
        verdicts = await self.llm.agenerate(
            judgement.text,
            judgement.response_model,
        )

        values = [item.verdict for item in verdicts.statements]
        value = sum(values) / len(values) if values else float("nan")
        return MetricScore(
            metric_name=self.name,
            prompt_profile=self.prompt_profile.name,
            value=float(value),
            calculation=f"支持陈述数 {sum(values)} / 总陈述数 {len(values)}",
            judge_steps=[
                JudgeStep(
                    name="拆分回答中的原子陈述",
                    prompt=extraction.text,
                    output=statements.model_dump(),
                ),
                JudgeStep(
                    name="根据上下文逐条验证陈述",
                    prompt=judgement.text,
                    output=verdicts.model_dump(),
                ),
            ],
        )


# ──────────────────────────────────────────────
# 4. Answer Relevancy：生成答案是否切题
# ──────────────────────────────────────────────


class AnswerRelevancyMetric:
    """反推问题并用 Embedding 比较它与原问题的语义相似度。"""

    name = "answer_relevancy"

    def __init__(
        self,
        llm: Any,
        embeddings: Any,
        strictness: int = 3,
        prompt_profile: PromptProfile | None = None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.strictness = strictness
        self.prompt_profile = prompt_profile or get_prompt_profile()

    async def score(self, sample: RAGOutput) -> MetricScore | None:
        if not sample.answer:
            return None

        judge_steps: list[JudgeStep] = []
        generated_questions: list[str] = []
        noncommittal_flags: list[int] = []

        for index in range(self.strictness):
            prepared = self.prompt_profile.answer_relevancy.prepare(
                answer=sample.answer,
            )
            result = await self.llm.agenerate(
                prepared.text,
                prepared.response_model,
            )
            generated_questions.append(result.question)
            noncommittal_flags.append(result.noncommittal)
            judge_steps.append(
                JudgeStep(
                    name=f"根据回答反推问题（第 {index + 1} 次）",
                    prompt=prepared.text,
                    output=result.model_dump(),
                )
            )

        original_vector = np.asarray(
            await self.embeddings.aembed_text(sample.question),
            dtype=float,
        )
        generated_vectors = np.asarray(
            await self.embeddings.aembed_texts(generated_questions),
            dtype=float,
        )
        denominator = np.linalg.norm(generated_vectors, axis=1) * np.linalg.norm(
            original_vector
        )
        similarities = np.divide(
            generated_vectors @ original_vector,
            denominator,
            out=np.zeros(len(generated_vectors), dtype=float),
            where=denominator != 0,
        )
        is_noncommittal = all(noncommittal_flags)
        value = float(similarities.mean()) * int(not is_noncommittal)

        similarities_text = ", ".join(f"{item:.4f}" for item in similarities)
        return MetricScore(
            metric_name=self.name,
            prompt_profile=self.prompt_profile.name,
            value=value,
            calculation=(
                f"反推问题与原问题的余弦相似度均值 mean([{similarities_text}])"
                f" × 非回避系数 {int(not is_noncommittal)}"
            ),
            judge_steps=judge_steps,
        )
