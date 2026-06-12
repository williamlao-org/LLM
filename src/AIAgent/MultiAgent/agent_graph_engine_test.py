from Agent.MultiAgent.agent_graph_engine import (
    AgentModule,
    AgentGraph,
    ModuleContext,
    RunState,
    END,
)

import os
import json
import time
from typing import Any, cast
from Agent.MultiAgent.modular_agent_graph import JsonUtils

from openai.types.chat import ChatCompletionMessageParam

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
MODEL = os.getenv("OPENAI_MODEL", "zai-org/GLM-4.6")


class LLMGateway:
    def __init__(
        self,
        model: str = MODEL,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def ask_json(
        self, system_prompt: str, payload: dict[str, Any], temperature: float = 0.1
    ) -> dict[str, Any]:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "只返回有效 JSON。\n输入：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    timeout=self.timeout_seconds,
                    response_format={"type": "json_object"},
                )
                parsed = JsonUtils.parse_object(
                    response.choices[0].message.content or ""
                )
                if parsed:
                    return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

            if attempt < self.max_retries:
                time.sleep(attempt * 1.1)

        raise RuntimeError(f"LLM JSON response failed after retries: {last_error}")


llm = LLMGateway()

WRITER_PROMPT = """
你是一个专业的文案写手。请根据给定的主题（topic）撰写一段高质量、引人入胜的初稿。

注意：
1. 风格要生动有趣，逻辑清晰。
2. 必须且只能返回有效的 JSON 格式。

输出 JSON 格式要求：
{
  "draft": "这里是撰写的文案内容..."
}
""".strip()

REVIEWER_PROMPT = """
你是一个资深编辑。请审核以下草稿（draft）。

你的任务：
1. 评估内容质量。
2. 给出最终判定（verdict）：
   - 如果草稿已经很完美，返回 "pass"。
   - 如果需要修改（如太短、逻辑不通、风格不对），返回 "revise"。
3. 给出具体的审稿意见（feedback）。

必须且只能返回 JSON：
{
  "verdict": "pass", // 或 "revise"
  "feedback": "你的具体修改建议或通过理由"
}
""".strip()

PUBLISHER_PROMPT = """
你是一个内容发布专员。
你收到了已经审核通过的终稿（draft）。请最后检查一遍格式，并确认发布。

必须且只能返回 JSON：
{
  "published_url": "https://blog.example.com/posts/unique-id-123"
}
""".strip()


def write(ctx: ModuleContext) -> dict[str, Any]:
    topic = ctx.inputs["topic"]
    draft = llm.ask_json(
        system_prompt=WRITER_PROMPT,
        payload={"topic": topic},
    )

    return draft


def review(ctx: ModuleContext) -> dict[str, Any]:
    draft = ctx.inputs["draft"]
    result = llm.ask_json(system_prompt=REVIEWER_PROMPT, payload={"draft": draft})
    return result


def publish(ctx: ModuleContext) -> dict[str, Any]:
    draft = ctx.inputs["draft"]
    result = llm.ask_json(system_prompt=PUBLISHER_PROMPT, payload={"draft": draft})
    return result


def review_router(state: RunState) -> str:
    verdict = state.data["verdict"]
    if verdict == "pass":
        return "pass"
    else:
        return "revise"


writer = AgentModule("writer", inputs=["topic"], outputs=["draft"], run=write)

reviewer = AgentModule(
    "reviewer", inputs=["draft"], outputs=["verdict", "feedback"], run=review
)


publisher = AgentModule(
    "publisher", inputs=["draft"], outputs=["published_url"], run=publish
)

graph = AgentGraph()
graph.add_module(writer).add_module(reviewer).add_module(publisher)

graph.add_edge("writer", "reviewer")

graph.add_conditional_edges(
    "reviewer", review_router, {"revise": "writer", "pass": "publisher"}
)

graph.add_edge("publisher", END)

graph.set_start("writer")

compiled_graph = graph.compile(initial_keys={"topic"})

state = RunState(data={"topic": "雨后小故事"})

state = compiled_graph.run(state=state)

print(state)
