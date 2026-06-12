"""
LLM 传输层：屏蔽"流式 / 非流式"的差异，对外统一吐出事件流。

核心思想：
    "内容是逐 token 到达，还是一次性到达"是底层传输细节，
    不应该泄漏给主循环和展示层。call_llm 把这个差异在这里抹平：

    - 流式：边收 chunk 边 yield ReasoningDelta/ContentDelta（保证实时），
            chunk 循环结束后再 yield 一个 ContentDone（携带完整内容）。
    - 非流式：内部一次性拿到完整响应，直接 yield 一个 ContentDone。

    实时打印靠 Delta，完整内容靠 Done —— 上层永远只消费事件，
    不需要知道也不关心底层走的是哪条路径。
"""

from typing import Iterator

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .events import LLMEvent, ReasoningDelta, ContentDelta, ContentDone, UsageEvent


def call_llm(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
    stream: bool = True,
) -> Iterator[LLMEvent]:
    """调用一次 LLM，以事件流的形式产出结果。

    Args:
        client: OpenAI 客户端（由调用方传入，便于测试与替换）
        messages: 完整对话上下文
        model: 模型名
        stream: 是否走流式。无论真假，都必定以一个 ContentDone 收尾。

    Yields:
        LLMEvent: ReasoningDelta / ContentDelta / ContentDone / UsageEvent
    """
    if stream:
        yield from _call_stream(client, messages, model)
    else:
        yield from _call_once(client, messages, model)


def _call_stream(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
) -> Iterator[LLMEvent]:
    """流式路径：逐 chunk 实时 yield Delta，最后汇总成 ContentDone。"""
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
        stream_options={"include_usage": True},
    )

    content: list[str] = []
    reasoning: list[str] = []

    for chunk in resp:
        # 最后一个携带 usage 的 chunk 通常没有 choices
        if not chunk.choices:
            if getattr(chunk, "usage", None):
                yield UsageEvent(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 部分模型（如 DeepSeek）会在 reasoning_content 里给出思维链
        reasoning_piece = getattr(delta, "reasoning_content", None)
        content_piece = delta.content or ""

        # 关键：yield 写在 chunk 循环内部，token 一到就立刻交给上层渲染
        if reasoning_piece:
            reasoning.append(reasoning_piece)
            yield ReasoningDelta(reasoning_piece)

        if content_piece:
            content.append(content_piece)
            yield ContentDelta(content_piece)

    # chunk 流结束：吐出完整内容，供主循环解析 JSON
    yield ContentDone(content="".join(content), reasoning="".join(reasoning))


def _call_once(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
) -> Iterator[LLMEvent]:
    """非流式路径：一次性拿到完整响应，直接产出一个 ContentDone。"""
    resp = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=False,
    )

    message = resp.choices[0].message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None) or ""

    if getattr(resp, "usage", None):
        yield UsageEvent(resp.usage)

    yield ContentDone(content=content, reasoning=reasoning)
