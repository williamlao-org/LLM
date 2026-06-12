import uuid
import json
from typing import Literal, Any
from openai.types.chat import ChatCompletionMessageParam

from .tools.base import ToolCall, ToolResult


class TurnAbort(Exception):
    """本轮 LLM 输出无法解析或路由,这一轮没法继续。

    与"单个工具失败"区分开:工具失败是数据(ToolResult.fail),整轮照常;
    TurnAbort 是整轮中止,只能把错误喂回 LLM 让它重答。
    主循环专门捕获它,其它异常一律放行(那是真 bug,不该被静默吞掉)。
    """


def llm_json_parser(llm_content: str) -> dict:
    """从 LLM 回复文本里抠出最外层 { } 之间的 JSON 并解析。

    找不到 { } 或 JSON 非法 → 本轮没法路由 → 抛 TurnAbort。
    """
    try:
        start = llm_content.index("{")
        end = llm_content.rindex("}")
        return json.loads(llm_content[start : end + 1])
    except (ValueError, json.JSONDecodeError) as e:
        # .index 找不到子串抛 ValueError;json.loads 非法抛 JSONDecodeError
        raise TurnAbort(f"LLM 输出不是合法 JSON: {e}") from e


def parse_tool_calls(raw_calls: list[dict]) -> list[ToolCall]:
    """把 LLM 给的一批 raw dict 解析成 ToolCall 列表。
    非法的不抛异常,而是造一个带 error 的占位,交给执行阶段统一 fail。"""
    tool_calls: list[ToolCall] = []
    for raw in raw_calls:
        call_id = f"call_{uuid.uuid4().hex[:6]}"  # 先盖章,跟合不合法无关
        try:
            tool_calls.append(ToolCall.from_dict(raw, call_id))
        except (ValueError, KeyError) as e:
            tool_calls.append(
                ToolCall(
                    name="", arguments={}, id=call_id, error=f"非法 tool_call: {e}"
                )
            )
    return tool_calls


def route(
    content_json: dict,
) -> tuple[Literal["final", "tool_calls"], Any]:
    tool_calls = content_json.get("tool_calls")
    final_answer = content_json.get("final_answer")
    if tool_calls and final_answer is None:
        return "tool_calls", tool_calls
    elif not tool_calls and final_answer:
        return "final", final_answer
    else:
        raise TurnAbort("必须仅存在 tool_calls 或 final_answer")


def build_tool_results_message(
    tool_tuple: list[tuple[ToolCall, ToolResult]],
) -> ChatCompletionMessageParam:
    msg: ChatCompletionMessageParam = {
        "role": "user",
        "content": json.dumps(
            {
                "tool_results": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "result": tool_result.to_dict(),
                    }
                    for tool_call, tool_result in tool_tuple
                ]
            },
            ensure_ascii=False,
        ),
    }
    return msg
