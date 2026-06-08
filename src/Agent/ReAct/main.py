"""
ReAct Agent 主入口模块

实现了 ReAct (Reasoning + Acting) 模式的 Agent 循环：
1. Reasoning (推理)：LLM 接收当前上下文，思考下一步该做什么
2. Action (行动)：根据推理结果调用对应的工具
3. Observation (观察)：将工具执行结果反馈给 LLM，作为下一轮推理的输入

整个循环持续进行，直到 LLM 判断任务已完成并返回 final_answer。
"""

import json
import os
from .logger import get_logger

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

from .tools import tools  # 导入所有已注册的工具定义
from .tools.base import ToolCall, ToolResult
from .prompt import SYSTEM_PROMPT  # 导入系统提示词模板
from .llm import call_llm  # 传输层：屏蔽流式/非流式差异，吐出事件流
from .events import ReasoningDelta, ContentDelta, ContentDone  # 事件类型
from .renderer import Renderer, ConsoleRenderer  # 展示层

# 从 .env 文件加载环境变量（OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL 等）
load_dotenv()

logger = get_logger(__name__)

# ===== 工具注册表 =====
# 将工具列表转换为 {工具名 -> 工具函数} 的字典，方便后续按名称快速查找并调用
tool_registry = {tool.name: tool.func for tool in tools}

# ===== 初始化对话消息列表 =====
# messages 是整个对话的上下文，包含 system prompt、用户输入、助手回复和工具结果
messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        # 将可用工具的 JSON 描述注入到系统提示词中，让 LLM 知道它可以使用哪些工具
        "content": SYSTEM_PROMPT.format(
            tools=json.dumps(
                [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
            )
        ),
    },
    # 用户的初始任务请求（包含多个子任务，用于测试 Agent 的多工具调用能力）
    {
        "role": "user",
        # "content": "执行 python 代码  print(1/0)，并且使用 web_search 工具搜索一下 2024 年的奥运会在哪里举办？还有对ifconfig.me这个网站发起一个http请求，获取一下你的公网IP地址。",
        # "content": "看看当前工作区有哪些文件,看看文件中有什么内容，把其中的“我”改成“李建”，并新建一个贪吃蛇项目",
        "content": "看看我电脑的详细情况，使用command工具",
    },
]

# ===== 初始化 OpenAI 客户端 =====
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


def llm_json_parser(llm_content: str) -> dict:
    """
    从 LLM 的回复文本中提取 JSON 对象。

    LLM 的回复可能包含额外的文本（如 markdown 代码块标记），
    此函数通过定位最外层的 { } 来提取有效的 JSON 部分。

    Args:
        llm_content: LLM 返回的原始文本内容

    Returns:
        解析后的 Python 字典对象

    Raises:
        ValueError: 当文本中找不到有效的 JSON 结构时
    """
    start = llm_content.index("{")  # 找到第一个 '{' 的位置
    end = llm_content.rindex("}")  # 找到最后一个 '}' 的位置
    if start == -1 or end == -1:
        raise ValueError("Invalid JSON")
    return json.loads(llm_content[start : end + 1])


def route(content_json: dict):
    if content_json.get("final_answer"):
        return ("final", content_json["final_answer"])
    if content_json.get("tool_call"):
        return ("call", content_json["tool_call"])
    raise ValueError("既不是 final_answer 也不是 tool_call")


def execute_tool_call(tool_call: ToolCall) -> ToolResult:
    """查找并执行工具，返回标准化 tool_result"""

    tool_name = tool_call.name  # 工具名称
    tool_arguments = tool_call.arguments  # 工具参数

    # 从注册表中查找对应的工具函数
    tool_fn = tool_registry.get(tool_name)
    if tool_fn is None:
        return ToolResult.fail(err=f"Unknown tool:{tool_name}")

    # 调用工具并获取结果（Observation）
    try:
        tool_result = tool_fn(**tool_arguments)
    except Exception as e:
        tool_result = ToolResult.fail(f"{type(e).__name__}: {e}")

    return tool_result


def build_tool_result_message(
    tool_result: ToolResult | dict,
) -> ChatCompletionMessageParam:
    """把工具结果包装成下一轮给 LLM 的消息"""

    if isinstance(tool_result, ToolResult):
        tool_result = tool_result.to_dict()

    msg: ChatCompletionMessageParam = {
        "role": "user",
        "content": json.dumps({"tool_result": tool_result}, ensure_ascii=False),
    }
    return msg


def run_turn(model: str, stream: bool, renderer: Renderer) -> str:
    """跑一轮 LLM 调用：实时渲染事件流，并返回拼接好的完整 content。

    传输层 call_llm 吐出事件，这里只负责把事件分发给 renderer，
    同时从 ContentDone 里取出完整内容交还给主循环去解析。
    """
    content = ""
    for ev in call_llm(client, messages, model, stream):
        if isinstance(ev, ReasoningDelta):
            renderer.on_reasoning_delta(ev.piece)
        elif isinstance(ev, ContentDelta):
            renderer.on_content_delta(ev.piece)
        elif isinstance(ev, ContentDone):
            content = ev.content  # 完整内容，留给主循环解析 JSON
        # UsageEvent 暂时忽略，以后接监控 / 计费再处理
    return content


def main(
    stream: bool = True,
    renderer: Renderer | None = None,
    max_steps: int = 25,
):
    """
    ReAct Agent 的主循环（纯编排）。

    核心流程：
        1. run_turn：调用 LLM、实时渲染、拿到完整回复
        2. 解析回复 JSON，判断是 final_answer（结束）还是 tool_call（继续）
        3. 若是 tool_call，执行工具并把结果作为"观察"追加进 messages
        4. 回到步骤 1，开始下一轮推理

    循环最多跑 max_steps 步，避免 LLM 始终不给 final_answer
    （或反复吐出无法解析的内容）导致无限循环、持续烧 API 调用。

    Args:
        stream: 是否使用流式输出。为 True 时逐 token 打印，体验更好。
        renderer: 展示层。默认 ConsoleRenderer（终端实时输出）。
        max_steps: 最大推理步数上限，防止死循环。
    """
    renderer = renderer or ConsoleRenderer()
    model = os.getenv("OPENAI_MODEL") or "gpt"

    for _ in range(max_steps):
        # ----- 步骤 1：调用 LLM 推理（实时渲染由 renderer 完成）-----
        content = run_turn(model, stream, renderer)
        messages.append({"role": "assistant", "content": content})

        # ----- 步骤 2：解析回复，判断下一步动作 -----
        try:
            content_json = llm_json_parser(content)
            kind, payload = route(content_json)

            # 若返回 final_answer，任务完成，结束循环
            if kind == "final":
                renderer.on_final(payload)
                return

            tool_call = ToolCall.from_dict(payload)
        except (ValueError, KeyError) as e:
            # 解析失败也是一种观察,喂回去让 LLM 重发
            tool_result=ToolResult.fail(f'解析 tool_call 时失败：{e}')
            messages.append(
                build_tool_result_message(tool_result=tool_result)
            )
            continue

        renderer.on_tool_call(tool_call)

        # ----- 步骤 3：执行工具调用（Action）-----
        tool_result = execute_tool_call(tool_call)
        renderer.on_tool_result(tool_result)

        # ----- 步骤 4：将工具结果作为"观察"反馈给 LLM -----
        messages.append(build_tool_result_message(tool_result))
    else:
        # 跑满 max_steps 仍未给出 final_answer：主动收尾，不伪装成功
        renderer.on_final(f"已达到最大步数上限（{max_steps} 步），任务未完成。")


main()
