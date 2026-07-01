"""自适应多跳 RAG 的单跳证据评估器。"""

import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from phase1_dense_retriever import SearchResult
from phase3_query_decomposer import QueryStep


class HopAssessment(BaseModel):
    """单跳检索的结构化事实与控制信号，不包含模型思维链。"""

    relevance: Literal["relevant", "partially_relevant", "irrelevant"]
    sufficiency: Literal["sufficient", "insufficient", "conflicting"]
    can_answer_question: bool = Field(
        description="当前累计证据是否已经足以完整回答原始问题"
    )
    extracted_facts: list[str] = Field(
        default_factory=list,
        description="从本跳证据中提取的、可被后续查询使用的简短事实",
    )
    resolved_entities: dict[str, str] = Field(
        default_factory=dict,
        description="占位概念到具体实体的映射，例如 company -> Adept AI",
    )
    suggested_query: str | None = Field(
        default=None,
        description="证据不足时建议的改写查询",
    )
    reason: str = Field(description="简短评估理由，不输出详细推理过程")


_HOP_ASSESS_SYSTEM = """你是多跳检索执行器中的单跳证据评估器。

你的任务是基于原始问题、当前步骤、依赖事实和本次检索结果，输出结构化评估：
1. 判断本次证据与当前步骤的相关性和充分性。
2. 提取后续步骤真正需要的简短事实和实体映射。
3. 判断累计证据是否已足以回答原始问题。
4. 如果证据不足，给出更具体、可直接用于检索的 suggested_query。

不要输出思维链、分析过程或长篇解释；reason 只写一句可审计的结论。"""


class HopAssessor:
    """通过强制 function calling 生成 `HopAssessment`。"""

    def __init__(self, llm_client: OpenAI, model: str):
        self.llm_client = llm_client
        self.model = model

    @staticmethod
    def _format_results(results: list[SearchResult]) -> str:
        if not results:
            return "（本次没有检索到结果）"

        parts = []
        for index, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "未知来源")
            parts.append(f"[{index}] [来源: {source}]\n{result.chunk.content}")
        return "\n\n---\n\n".join(parts)

    def assess(
        self,
        original_question: str,
        step: QueryStep,
        executed_query: str,
        dependency_facts: list[str],
        accumulated_facts: list[str],
        results: list[SearchResult],
        verbose: bool = False,
    ) -> HopAssessment:
        """评估一个检索跳，并返回事实和控制信号。"""
        if verbose:
            print(f"     🧪 评估 Step {step.step_id} 的单跳证据...")

        payload = {
            "original_question": original_question,
            "current_step": step.model_dump(),
            "executed_query": executed_query,
            "dependency_facts": dependency_facts,
            "accumulated_facts": accumulated_facts,
            "retrieval_results": self._format_results(results),
        }
        tools = [{
            "type": "function",
            "function": {
                "name": "submit_hop_assessment",
                "description": "提交单跳检索的结构化评估",
                "parameters": HopAssessment.model_json_schema(),
            },
        }]
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _HOP_ASSESS_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            "tools": tools,
            "temperature": 0.1,
        }

        try:
            response = self.llm_client.chat.completions.create(
                **kwargs,
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_hop_assessment"},
                },
            )
        except Exception as error:
            message = str(error).lower()
            if "tool_choice" not in message and "thinking" not in message:
                raise
            response = self.llm_client.chat.completions.create(
                **kwargs,
                tool_choice="auto",
            )

        message = response.choices[0].message
        if message.tool_calls:
            raw = message.tool_calls[0].function.arguments
        else:
            raw = message.content or ""

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("单跳评估必须是 JSON 对象")
            return HopAssessment.model_validate(data)
        except Exception as error:
            raise ValueError(f"无法解析单跳评估: {error}") from error
