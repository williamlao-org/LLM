"""
ReActMulti Agent 主入口模块（多工具版）

和隔壁 ReAct 的唯一区别：一个回合可以发起【多个】工具调用。
单工具版是严格串行 think→act(1个)→observe；这一版是 think→act(N个)→observe(N个)。

本文件里【没改动】的部分（直接复用单工具版的设计）：
    - llm_json_parser   : 从 LLM 文本里抠出 JSON
    - execute_tool_call : 查表 + 执行【单个】工具，返回标准化 ToolResult
    - run_turn          : 跑一轮 LLM、实时渲染、拿到完整 content

需要你写的【3 处】已用 “TODO（第 N 处）” 标出，集中在：
    1. prompt.py 的 SYSTEM_PROMPT（多工具 schema）
    2. route()              —— 从 JSON 里取出"多个" tool_call
    3. main() 主循环 + 结果回传 —— 遍历执行、把 N 个结果一起喂回去
"""

import uuid

from typing import Literal, Any, Callable

import json
import os
from .logger import get_logger

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

from .tools import tools
from .tools.base import ToolCall, ToolResult
from .prompt import SYSTEM_PROMPT
from .llm import call_llm
from .events import ReasoningDelta, ContentDelta, ContentDone
from .renderer import Renderer, ConsoleRenderer

load_dotenv()

logger = get_logger(__name__)


class TurnAbort(Exception):
    """本轮 LLM 输出无法解析或路由,这一轮没法继续。

    与"单个工具失败"区分开:工具失败是数据(ToolResult.fail),整轮照常;
    TurnAbort 是整轮中止,只能把错误喂回 LLM 让它重答。
    主循环专门捕获它,其它异常一律放行(那是真 bug,不该被静默吞掉)。
    """


tool_registry = {tool.name: tool.func for tool in tools}

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT.format(
            tools=json.dumps(
                [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
            )
        ),
    },
    {
        "role": "user",
        # 这条任务故意包含多个互相独立的子任务，正好用来检验"一回合发多个工具"
        "content": "执行 python 代码 print(1/0)，并且用 web_search 搜索 2024 年奥运会在哪举办，"
        "再对 ifconfig.me/ip 发起 http 请求拿到公网 IP。",
    },
]

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


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


def execute_tool_call(tool_call: ToolCall) -> ToolResult:
    """查找并执行【单个】工具，返回标准化 tool_result。"""
    if tool_call.error:
        return ToolResult.fail(tool_call.error)

    tool_name = tool_call.name
    tool_arguments = tool_call.arguments

    tool_fn = tool_registry.get(tool_name)
    if tool_fn is None:
        return ToolResult.fail(err=f"Unknown tool:{tool_name}")

    try:
        tool_result = tool_fn(**tool_arguments)
    except Exception as e:
        tool_result = ToolResult.fail(f"{type(e).__name__}: {e}")

    return tool_result


def execute_tool_calls(
    tool_calls: list[ToolCall],
    on_call: Callable[[ToolCall], None] | None = None,
    on_result: Callable[[ToolResult], None] | None = None,
) -> list[tuple[ToolCall, ToolResult]]:
    """逐个执行工具,返回 (call, result) 列表。

    on_call / on_result 是可选回调,在每个工具执行前/后被喊一声。
    不传则纯执行(无副作用,可单测);传了就能实时渲染进度。
    循环的所有权始终在本函数,渲染只是从插槽注入。

    ### Explain
    execute_tool_calls 想保持纯粹（可测、无副作用），但它跑的那个 for 循环又是渲染唯一能"实时插话"的地方。矛盾点在于：循环的所有权在执行函数手里，但渲染想在循环的每一步插一脚。 回调就是执行函数对外开的两个"插槽"——"我每调一个工具前/后，会喊一声，你想接就接，不接我照跑"。这样循环归执行函数独有（不重复），渲染从外部注入（不污染纯粹性）。
    """
    results: list[tuple[ToolCall, ToolResult]] = []
    for tool_call in tool_calls:
        if on_call:
            on_call(tool_call)
        result = execute_tool_call(tool_call)
        if on_result:
            on_result(result)
        results.append((tool_call, result))
    return results


def run_turn(model: str, stream: bool, renderer: Renderer) -> str:
    """跑一轮 LLM 调用：实时渲染事件流，返回拼接好的完整 content。"""
    content = ""
    for ev in call_llm(client, messages, model, stream):
        if isinstance(ev, ReasoningDelta):
            renderer.on_reasoning_delta(ev.piece)
        elif isinstance(ev, ContentDelta):
            renderer.on_content_delta(ev.piece)
        elif isinstance(ev, ContentDone):
            content = ev.content
    return content


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


def main(
    stream: bool = True,
    renderer: Renderer | None = None,
    max_steps: int = 25,
):
    """ReActMulti 主循环（纯编排）。"""
    renderer = renderer or ConsoleRenderer()
    model = os.getenv("OPENAI_MODEL") or "gpt"

    for _ in range(max_steps):
        # ----- 步骤 1：调用 LLM 推理 -----
        content = run_turn(model, stream, renderer)
        messages.append({"role": "assistant", "content": content})

        # ----- 步骤 2：解析 + 路由 -----
        try:
            content_json = llm_json_parser(content)  # 若解析失败，会丢给 except
            kind, payload = route(content_json)

            if kind == "final":
                renderer.on_final(payload)
                print(json.dumps(messages, ensure_ascii=False, indent=2))

                return

            # payload 是一组 raw tool_call dict。三步走:解析 → 执行(回调实时渲染) → 回传。
            assert isinstance(payload, list)
            tool_calls = parse_tool_calls(payload)
            results = execute_tool_calls(
                tool_calls,
                on_call=renderer.on_tool_call,
                on_result=renderer.on_tool_result,
            )
            messages.append(build_tool_results_message(results))

        except TurnAbort as e:
            # 本轮 LLM 输出无法解析/路由,把错误喂回去让它重答(其它异常不在此捕,留给真 bug 暴露)
            msg: ChatCompletionMessageParam = {
                "role": "user",
                "content": json.dumps(
                    {"error": f"LLM 输出无法解析或路由：{e}"},
                    ensure_ascii=False,
                ),
            }
            messages.append(msg)
            continue

    else:
        renderer.on_final(f"已达到最大步数上限（{max_steps} 步），任务未完成。")


main()
