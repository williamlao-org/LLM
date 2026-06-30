"""Prompt Profile 与自定义 RAG 指标的纯本地测试。"""

import asyncio
import sys

import pytest

from phase2_04_evaluator import parse_args
from phase2_04_metrics import (
    AnswerRelevancyMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
    build_metrics,
)
from phase2_04_models import RAGOutput
from phase2_04_official_ragas_prompts import (
    AnswerRelevancePrompt as LocalAnswerRelevancePrompt,
    ContextPrecisionPrompt as LocalContextPrecisionPrompt,
    ContextRecallPrompt as LocalContextRecallPrompt,
    NLIStatementPrompt as LocalNLIStatementPrompt,
    StatementGeneratorPrompt as LocalStatementGeneratorPrompt,
)
from phase2_04_prompt_profiles import get_prompt_profile
from ragas.metrics.collections.answer_relevancy.util import (
    AnswerRelevancePrompt as UpstreamAnswerRelevancePrompt,
)
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionPrompt as UpstreamContextPrecisionPrompt,
)
from ragas.metrics.collections.context_recall.util import (
    ContextRecallPrompt as UpstreamContextRecallPrompt,
)
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementPrompt as UpstreamNLIStatementPrompt,
)
from ragas.metrics.collections.faithfulness.util import (
    StatementGeneratorPrompt as UpstreamStatementGeneratorPrompt,
)


PROFILE_NAMES = ("official", "custom_zh")


class FakeLLM:
    """按顺序返回符合 Prompt response model 的模拟 Judge 输出。"""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []
        self.response_models = []

    async def agenerate(self, prompt, response_model):
        self.prompts.append(prompt)
        self.response_models.append(response_model)
        return response_model.model_validate(self.responses.pop(0))


class FakeEmbeddings:
    async def aembed_text(self, text):
        return [1.0, 0.0]

    async def aembed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


def assert_profile_prompt(prompt: str, profile_name: str) -> None:
    if profile_name == "official":
        assert "Please return the output in a JSON format" in prompt
        assert "Now perform the same with the following input" in prompt
    else:
        assert "【任务】" in prompt
        assert "【安全规则】" in prompt
        assert "【当前输入】" in prompt


def test_metric_order_and_default_profile_follow_rag_pipeline():
    metrics = build_metrics(FakeLLM(), FakeEmbeddings())

    assert list(metrics) == [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    assert {metric.prompt_profile.name for metric in metrics.values()} == {"official"}

    custom_metrics = build_metrics(
        FakeLLM(),
        FakeEmbeddings(),
        prompt_profile="custom_zh",
    )
    assert {metric.prompt_profile.name for metric in custom_metrics.values()} == {
        "custom_zh"
    }


def test_cli_defaults_to_official_and_accepts_custom_zh(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["phase2_04_evaluator.py"])
    assert parse_args().prompt_profile == "official"

    monkeypatch.setattr(
        sys,
        "argv",
        ["phase2_04_evaluator.py", "--prompt-profile", "custom_zh"],
    )
    assert parse_args().prompt_profile == "custom_zh"


@pytest.fixture
def rag_output():
    return RAGOutput(
        question="什么是 RAG？",
        answer="RAG 结合了检索和生成，也能减少幻觉。",
        contexts=["RAG 将检索到的外部知识提供给生成模型。", "天气晴朗。"],
        ground_truth="RAG 结合检索和生成，可以减少幻觉。",
    )


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_faithfulness_keeps_prompts_verdicts_and_formula(rag_output, profile_name):
    llm = FakeLLM(
        {
            "statements": [
                "RAG 结合了检索和生成。",
                "RAG 能减少幻觉。",
            ]
        },
        {
            "statements": [
                {
                    "statement": "RAG 结合了检索和生成。",
                    "reason": "上下文支持。",
                    "verdict": 1,
                },
                {
                    "statement": "RAG 能减少幻觉。",
                    "reason": "上下文没有直接说明。",
                    "verdict": 0,
                },
            ]
        },
    )
    metric = FaithfulnessMetric(llm, get_prompt_profile(profile_name))

    result = asyncio.run(metric.score(rag_output))

    assert result is not None
    assert result.prompt_profile == profile_name
    assert result.value == pytest.approx(0.5)
    assert result.calculation == "支持陈述数 1 / 总陈述数 2"
    assert len(result.judge_steps) == 2
    assert result.judge_steps[1].output["statements"][1]["verdict"] == 0
    assert_profile_prompt(result.judge_steps[0].prompt, profile_name)


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_answer_relevancy_uses_generated_questions_and_embeddings(
    rag_output,
    profile_name,
):
    response = {"question": "什么是 RAG？", "noncommittal": 0}
    metric = AnswerRelevancyMetric(
        FakeLLM(response, response, response),
        FakeEmbeddings(),
        strictness=3,
        prompt_profile=get_prompt_profile(profile_name),
    )

    result = asyncio.run(metric.score(rag_output))

    assert result is not None
    assert result.prompt_profile == profile_name
    assert result.value == pytest.approx(1.0)
    assert len(result.judge_steps) == 3
    assert "余弦相似度" in result.calculation
    assert_profile_prompt(result.judge_steps[0].prompt, profile_name)


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_context_precision_uses_ranked_average_precision(rag_output, profile_name):
    metric = ContextPrecisionMetric(
        FakeLLM(
            {"reason": "第一个 chunk 有用。", "verdict": 1},
            {"reason": "第二个 chunk 无关。", "verdict": 0},
        ),
        get_prompt_profile(profile_name),
    )

    result = asyncio.run(metric.score(rag_output))

    assert result is not None
    assert result.prompt_profile == profile_name
    assert result.value == pytest.approx(1.0)
    assert result.calculation.endswith("verdicts=[1, 0]")
    assert len(result.judge_steps) == 2
    assert_profile_prompt(result.judge_steps[0].prompt, profile_name)


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_context_recall_keeps_each_reference_fact_reason(rag_output, profile_name):
    metric = ContextRecallMetric(
        FakeLLM(
            {
                "classifications": [
                    {
                        "statement": "RAG 结合检索和生成。",
                        "reason": "上下文支持。",
                        "attributed": 1,
                    },
                    {
                        "statement": "RAG 可以减少幻觉。",
                        "reason": "上下文没有直接说明。",
                        "attributed": 0,
                    },
                ]
            }
        ),
        get_prompt_profile(profile_name),
    )

    result = asyncio.run(metric.score(rag_output))

    assert result is not None
    assert result.prompt_profile == profile_name
    assert result.value == pytest.approx(0.5)
    assert result.calculation == "已召回事实数 1 / 参考答案事实总数 2"
    assert result.judge_steps[0].output["classifications"][1]["attributed"] == 0
    assert_profile_prompt(result.judge_steps[0].prompt, profile_name)


@pytest.mark.parametrize(
    ("local_prompt", "upstream_prompt", "input_data"),
    [
        (
            LocalContextPrecisionPrompt(),
            UpstreamContextPrecisionPrompt(),
            {"question": "q", "context": "c", "answer": "a"},
        ),
        (
            LocalContextRecallPrompt(),
            UpstreamContextRecallPrompt(),
            {"question": "q", "context": "c", "answer": "a"},
        ),
        (
            LocalStatementGeneratorPrompt(),
            UpstreamStatementGeneratorPrompt(),
            {"question": "q", "answer": "a"},
        ),
        (
            LocalNLIStatementPrompt(),
            UpstreamNLIStatementPrompt(),
            {"context": "c", "statements": ["s"]},
        ),
        (
            LocalAnswerRelevancePrompt(),
            UpstreamAnswerRelevancePrompt(),
            {"response": "a"},
        ),
    ],
)
def test_official_prompt_snapshot_matches_ragas_0_4_3(
    local_prompt,
    upstream_prompt,
    input_data,
):
    local_input = local_prompt.input_model.model_validate(input_data)
    upstream_input = upstream_prompt.input_model.model_validate(input_data)

    assert local_prompt.instruction == upstream_prompt.instruction
    assert len(local_prompt.examples) == len(upstream_prompt.examples)
    assert local_prompt.output_model.model_json_schema() == (
        upstream_prompt.output_model.model_json_schema()
    )
    assert local_prompt.to_string(local_input) == upstream_prompt.to_string(
        upstream_input
    )
