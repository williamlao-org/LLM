"""RAGAS 0.4.3 官方 Prompt 的本地快照。

来源：ragas==0.4.3 的 ``ragas.metrics.collections`` 与
``ragas.prompt.metrics.base_prompt``。这里复制并固定 Prompt、few-shot、
Pydantic Schema 和渲染格式，运行时不导入 RAGAS 私有 Prompt 模块。
"""

from __future__ import annotations

import json
import typing as t
from abc import ABC

from pydantic import BaseModel, Field


RAGAS_PROMPT_SNAPSHOT_VERSION = "0.4.3"

InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class OfficialRagasPrompt(ABC, t.Generic[InputModel, OutputModel]):
    """RAGAS 0.4.3 ``BasePrompt.to_string`` 的固定副本。"""

    input_model: type[InputModel]
    output_model: type[OutputModel]
    instruction: str
    examples: list[tuple[InputModel, OutputModel]]
    language: str = "english"

    def to_string(self, data: InputModel) -> str:
        output_schema = json.dumps(self.output_model.model_json_schema())
        examples_str = self._generate_examples()
        input_json = data.model_dump_json(indent=4, exclude_none=True)
        return f"""{self.instruction}
Please return the output in a JSON format that complies with the following schema as specified in JSON Schema:
{output_schema}Do not use single quotes in your response but double quotes,properly escaped with a backslash.

{examples_str}
-----------------------------

Now perform the same with the following input
input: {input_json}
Output: """

    def _generate_examples(self) -> str:
        example_strings = [
            f"Example {index}\n"
            f"Input: {input_data.model_dump_json(indent=4)}\n"
            f"Output: {output_data.model_dump_json(indent=4)}"
            for index, (input_data, output_data) in enumerate(self.examples, 1)
        ]
        return "--------EXAMPLES-----------\n" + "\n\n".join(example_strings)


# ──────────────────────────────────────────────
# Retrieval: Context Precision
# ──────────────────────────────────────────────


class ContextPrecisionInput(BaseModel):
    question: str = Field(..., description="The question being asked")
    context: str = Field(..., description="The context to evaluate for usefulness")
    answer: str = Field(
        ...,
        description="The answer/reference/response to compare against",
    )


class ContextPrecisionOutput(BaseModel):
    """Structured output for context precision evaluation."""

    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


class ContextPrecisionPrompt(
    OfficialRagasPrompt[ContextPrecisionInput, ContextPrecisionOutput]
):
    input_model = ContextPrecisionInput
    output_model = ContextPrecisionOutput
    instruction = (
        'Given question, answer and context verify if the context was useful in '
        'arriving at the given answer. Give verdict as "1" if useful and "0" if '
        "not with json output."
    )
    examples = [
        (
            ContextPrecisionInput(
                question="What can you tell me about Albert Einstein?",
                context=(
                    "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
                    "theoretical physicist, widely held to be one of the greatest and "
                    "most influential scientists of all time. Best known for developing "
                    "the theory of relativity, he also made important contributions to "
                    "quantum mechanics, and was thus a central figure in the revolutionary "
                    "reshaping of the scientific understanding of nature that modern physics "
                    "accomplished in the first decades of the twentieth century. His mass–energy "
                    "equivalence formula E = mc2, which arises from relativity theory, has been "
                    "called 'the world's most famous equation'. He received the 1921 Nobel Prize "
                    "in Physics 'for his services to theoretical physics, and especially for his "
                    "discovery of the law of the photoelectric effect', a pivotal step in the "
                    "development of quantum theory. His work is also known for its influence on "
                    "the philosophy of science. In a 1999 poll of 130 leading physicists worldwide "
                    "by the British journal Physics World, Einstein was ranked the greatest "
                    "physicist of all time. His intellectual achievements and originality have "
                    "made Einstein synonymous with genius."
                ),
                answer=(
                    "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                    "physicist, widely held to be one of the greatest and most influential "
                    "scientists of all time. He received the 1921 Nobel Prize in Physics for "
                    "his services to theoretical physics."
                ),
            ),
            ContextPrecisionOutput(
                reason=(
                    "The provided context was indeed useful in arriving at the given answer. "
                    "The context includes key information about Albert Einstein's life and "
                    "contributions, which are reflected in the answer."
                ),
                verdict=1,
            ),
        ),
        (
            ContextPrecisionInput(
                question="who won 2020 icc world cup?",
                context=(
                    "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, "
                    "2022, in Australia, was the eighth edition of the tournament. Originally "
                    "scheduled for 2020, it was postponed due to the COVID-19 pandemic. England "
                    "emerged victorious, defeating Pakistan by five wickets in the final to "
                    "clinch their second ICC Men's T20 World Cup title."
                ),
                answer="England",
            ),
            ContextPrecisionOutput(
                reason=(
                    "the context was useful in clarifying the situation regarding the 2020 ICC "
                    "World Cup and indicating that England was the winner of the tournament that "
                    "was intended to be held in 2020 but actually took place in 2022."
                ),
                verdict=1,
            ),
        ),
        (
            ContextPrecisionInput(
                question="What is the tallest mountain in the world?",
                context=(
                    "The Andes is the longest continental mountain range in the world, located "
                    "in South America. It stretches across seven countries and features many of "
                    "the highest peaks in the Western Hemisphere. The range is known for its "
                    "diverse ecosystems, including the high-altitude Andean Plateau and the "
                    "Amazon rainforest."
                ),
                answer="Mount Everest.",
            ),
            ContextPrecisionOutput(
                reason=(
                    "the provided context discusses the Andes mountain range, which, while "
                    "impressive, does not include Mount Everest or directly relate to the "
                    "question about the world's tallest mountain."
                ),
                verdict=0,
            ),
        ),
    ]


# ──────────────────────────────────────────────
# Retrieval: Context Recall
# ──────────────────────────────────────────────


class ContextRecallInput(BaseModel):
    question: str = Field(..., description="The original question asked by the user")
    context: str = Field(..., description="The retrieved context passage to evaluate")
    answer: str = Field(
        ...,
        description="The reference answer containing statements to classify",
    )


class ContextRecallClassification(BaseModel):
    """Classification of a single statement."""

    statement: str = Field(
        ...,
        description="Individual statement extracted from the answer",
    )
    reason: str = Field(
        ...,
        description="Reasoning for why the statement is or isn't attributable to context",
    )
    attributed: int = Field(
        ...,
        description=(
            "Binary classification: 1 if the statement can be attributed to context, "
            "0 otherwise"
        ),
    )


class ContextRecallOutput(BaseModel):
    """Structured output for context recall classifications."""

    classifications: list[ContextRecallClassification] = Field(
        ...,
        description="List of statement classifications",
    )


class ContextRecallPrompt(OfficialRagasPrompt[ContextRecallInput, ContextRecallOutput]):
    input_model = ContextRecallInput
    output_model = ContextRecallOutput
    instruction = """Given a context and an answer, analyze each statement in the answer and classify if the statement can be attributed to the given context or not.
Use only binary classification: 1 if the statement can be attributed to the context, 0 if it cannot.
Provide detailed reasoning for each classification."""
    examples = [
        (
            ContextRecallInput(
                question="What can you tell me about Albert Einstein?",
                context=(
                    "Albert Einstein (14 March 1879 - 18 April 1955) was a German-born "
                    "theoretical physicist, widely held to be one of the greatest and most "
                    "influential scientists of all time. Best known for developing the theory "
                    "of relativity, he also made important contributions to quantum mechanics, "
                    "and was thus a central figure in the revolutionary reshaping of the "
                    "scientific understanding of nature that modern physics accomplished in the "
                    "first decades of the twentieth century. His mass-energy equivalence formula "
                    "E = mc2, which arises from relativity theory, has been called 'the world's "
                    "most famous equation'. He received the 1921 Nobel Prize in Physics 'for his "
                    "services to theoretical physics, and especially for his discovery of the law "
                    "of the photoelectric effect', a pivotal step in the development of quantum "
                    "theory. His work is also known for its influence on the philosophy of science. "
                    "In a 1999 poll of 130 leading physicists worldwide by the British journal "
                    "Physics World, Einstein was ranked the greatest physicist of all time. His "
                    "intellectual achievements and originality have made Einstein synonymous "
                    "with genius."
                ),
                answer=(
                    "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                    "physicist, widely held to be one of the greatest and most influential "
                    "scientists of all time. He received the 1921 Nobel Prize in Physics for his "
                    "services to theoretical physics. He published 4 papers in 1905. Einstein "
                    "moved to Switzerland in 1895."
                ),
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement=(
                            "Albert Einstein, born on 14 March 1879, was a German-born theoretical "
                            "physicist, widely held to be one of the greatest and most influential "
                            "scientists of all time."
                        ),
                        reason="The date of birth of Einstein is mentioned clearly in the context.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement=(
                            "He received the 1921 Nobel Prize in Physics for his services to "
                            "theoretical physics."
                        ),
                        reason="The exact sentence is present in the given context.",
                        attributed=1,
                    ),
                    ContextRecallClassification(
                        statement="He published 4 papers in 1905.",
                        reason="There is no mention about papers he wrote in the given context.",
                        attributed=0,
                    ),
                    ContextRecallClassification(
                        statement="Einstein moved to Switzerland in 1895.",
                        reason="There is no supporting evidence for this in the given context.",
                        attributed=0,
                    ),
                ]
            ),
        ),
        (
            ContextRecallInput(
                question="who won 2020 icc world cup?",
                context=(
                    "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, "
                    "2022, in Australia, was the eighth edition of the tournament. Originally "
                    "scheduled for 2020, it was postponed due to the COVID-19 pandemic. England "
                    "emerged victorious, defeating Pakistan by five wickets in the final to "
                    "clinch their second ICC Men's T20 World Cup title."
                ),
                answer="England",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="England",
                        reason=(
                            "The context clarifies that England won the 2022 edition (which was "
                            "originally scheduled for 2020)."
                        ),
                        attributed=1,
                    )
                ]
            ),
        ),
        (
            ContextRecallInput(
                question="What is the tallest mountain in the world?",
                context=(
                    "The Andes is the longest continental mountain range in the world, located "
                    "in South America. It stretches across seven countries and features many of "
                    "the highest peaks in the Western Hemisphere. The range is known for its "
                    "diverse ecosystems, including the high-altitude Andean Plateau and the "
                    "Amazon rainforest."
                ),
                answer="Mount Everest.",
            ),
            ContextRecallOutput(
                classifications=[
                    ContextRecallClassification(
                        statement="Mount Everest.",
                        reason=(
                            "The provided context discusses the Andes mountain range, which does "
                            "not include Mount Everest or directly relate to the world's tallest "
                            "mountain."
                        ),
                        attributed=0,
                    )
                ]
            ),
        ),
    ]


# ──────────────────────────────────────────────
# Generation: Faithfulness
# ──────────────────────────────────────────────


class StatementGeneratorInput(BaseModel):
    question: str = Field(..., description="The question being answered")
    answer: str = Field(
        ...,
        description="The answer text to break down into statements",
    )


class StatementGeneratorOutput(BaseModel):
    """Structured output for statement generation."""

    statements: list[str] = Field(
        ...,
        description="The generated statements from the answer",
    )


class StatementGeneratorPrompt(
    OfficialRagasPrompt[StatementGeneratorInput, StatementGeneratorOutput]
):
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    instruction = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement."""
    examples = [
        (
            StatementGeneratorInput(
                question="Who was Albert Einstein and what is he best known for?",
                answer=(
                    "He was a German-born theoretical physicist, widely acknowledged to be one "
                    "of the greatest and most influential physicists of all time. He was best "
                    "known for developing the theory of relativity, he also made important "
                    "contributions to the development of the theory of quantum mechanics."
                ),
            ),
            StatementGeneratorOutput(
                statements=[
                    "Albert Einstein was a German-born theoretical physicist.",
                    "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                    "Albert Einstein was best known for developing the theory of relativity.",
                    "Albert Einstein made important contributions to the development of the theory of quantum mechanics.",
                ]
            ),
        )
    ]


class StatementFaithfulnessAnswer(BaseModel):
    """Individual statement with reason and verdict for NLI evaluation."""

    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness")


class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context to evaluate statements against")
    statements: list[str] = Field(
        ...,
        description="The statements to judge for faithfulness",
    )


class NLIStatementOutput(BaseModel):
    """Structured output for NLI statement evaluation."""

    statements: list[StatementFaithfulnessAnswer] = Field(
        ...,
        description="Evaluated statements with verdicts",
    )


class NLIStatementPrompt(
    OfficialRagasPrompt[NLIStatementInput, NLIStatementOutput]
):
    input_model = NLIStatementInput
    output_model = NLIStatementOutput
    instruction = """Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context."""
    examples = [
        (
            NLIStatementInput(
                context=(
                    "John is a student at XYZ University. He is pursuing a degree in Computer "
                    "Science. He is enrolled in several courses this semester, including Data "
                    "Structures, Algorithms, and Database Management. John is a diligent student "
                    "and spends a significant amount of time studying and completing assignments. "
                    "He often stays late in the library to work on his projects."
                ),
                statements=[
                    "John is majoring in Biology.",
                    "John is taking a course on Artificial Intelligence.",
                    "John is a dedicated student.",
                    "John has a part-time job.",
                ],
            ),
            NLIStatementOutput(
                statements=[
                    StatementFaithfulnessAnswer(
                        statement="John is majoring in Biology.",
                        reason="John's major is explicitly stated as Computer Science, not Biology.",
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is taking a course on Artificial Intelligence.",
                        reason=(
                            "The context mentions courses in Data Structures, Algorithms, and "
                            "Database Management, but does not mention Artificial Intelligence."
                        ),
                        verdict=0,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John is a dedicated student.",
                        reason=(
                            "The context states that John is a diligent student who spends a "
                            "significant amount of time studying and completing assignments."
                        ),
                        verdict=1,
                    ),
                    StatementFaithfulnessAnswer(
                        statement="John has a part-time job.",
                        reason="There is no information in the context about John having a part-time job.",
                        verdict=0,
                    ),
                ]
            ),
        )
    ]


# ──────────────────────────────────────────────
# Generation: Answer Relevancy
# ──────────────────────────────────────────────


class AnswerRelevanceInput(BaseModel):
    response: str = Field(
        ...,
        description="The response/answer to generate questions from",
    )


class AnswerRelevanceOutput(BaseModel):
    """Structured output for answer relevance question generation."""

    question: str = Field(
        ...,
        description="Question that can be answered from the response",
    )
    noncommittal: int = Field(
        ...,
        description="1 if the response is evasive/vague, 0 if it is substantive",
    )


class AnswerRelevancePrompt(
    OfficialRagasPrompt[AnswerRelevanceInput, AnswerRelevanceOutput]
):
    input_model = AnswerRelevanceInput
    output_model = AnswerRelevanceOutput
    instruction = """Generate a question for the given answer and identify if the answer is noncommittal.
Give noncommittal as 1 if the answer is noncommittal (evasive, vague, or ambiguous) and 0 if the answer is substantive.
Examples of noncommittal answers: "I don't know", "I'm not sure", "It depends"."""
    examples = [
        (
            AnswerRelevanceInput(response="Albert Einstein was born in Germany."),
            AnswerRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(
                response=(
                    "The capital of France is Paris, a city known for its architecture and "
                    "culture."
                )
            ),
            AnswerRelevanceOutput(
                question="What is the capital of France?",
                noncommittal=0,
            ),
        ),
        (
            AnswerRelevanceInput(
                response=(
                    "I don't know about the groundbreaking feature of the smartphone invented "
                    "in 2023 as I am unaware of information beyond 2022."
                )
            ),
            AnswerRelevanceOutput(
                question=(
                    "What was the groundbreaking feature of the smartphone invented in 2023?"
                ),
                noncommittal=1,
            ),
        ),
    ]
