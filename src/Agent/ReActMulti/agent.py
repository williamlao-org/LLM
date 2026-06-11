from .prompt import SYSTEM_PROMPT
from .renderer import Renderer
from .events import ReasoningDelta, ContentDelta, ContentDone
from .llm import LLMClient
from .tools.base import Tool
from openai.types.chat import ChatCompletionMessageParam

import json


class Agent:
    def __init__(self, client: LLMClient, tools: list[Tool], renderer: Renderer):
        self.client = client
        self.tools = tools
        self.renderer = renderer

        self.messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools=json.dumps(
                        [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
                    )
                ),
            }
        ]

    def run_turn(self) -> str:
        """跑一轮 LLM 调用：实时渲染事件流，返回拼接好的完整 content。"""

        for event in self.client(self.messages):
            if isinstance(event, ReasoningDelta):
                self.renderer.on_reasoning_delta(event.piece)
            elif isinstance(event, ContentDelta):
                self.renderer.on_content_delta(event.piece)
            elif isinstance(event, ContentDone):
                content = event.content
        return content

    def run(prompt: str):
        pass
