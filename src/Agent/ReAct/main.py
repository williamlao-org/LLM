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
from .prompt import SYSTEM_PROMPT  # 导入系统提示词模板

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


def llm_json_parser(llm_content: str):
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


def main(stream: bool = True):
    """
    ReAct Agent 的主循环。

    核心流程：
        1. 将当前 messages 发送给 LLM，获取回复
        2. 解析回复中的 JSON，判断是 final_answer（结束）还是 tool_call（继续）
        3. 如果是 tool_call，执行对应工具，将结果追加到 messages 中
        4. 回到步骤 1，开始下一轮推理

    Args:
        stream: 是否使用流式输出。为 True 时逐 token 打印，体验更好。
    """
    # 收集每一轮的推理过程（reasoning_content），用于调试和分析
    reasoning_contents = []
    model = os.getenv("OPENAI_MODEL") or "gpt"

    # ===== ReAct 主循环 =====
    while True:
        if stream:
            # ----- 步骤 1：调用 LLM 进行推理 -----
            resp = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                stream_options={"include_usage": True},
            )

            # ----- 步骤 2：处理 LLM 的响应 -----

            # 流式模式：逐 chunk 接收并拼接内容
            content = []
            reasoning_content = []

            for chunk in resp:
                if not chunk.choices:
                    continue  # 跳过空 chunk（如最后的 usage 信息）

                delta = chunk.choices[0].delta

                # 提取推理内容（部分模型如 DeepSeek 支持 reasoning_content 字段）
                reasoning_piece = getattr(delta, "reasoning_content", None)
                # 提取正式回复内容
                content_piece = delta.content or ""

                if reasoning_piece:
                    reasoning_content.append(reasoning_piece)

                if content_piece:
                    # 实时打印输出，flush=True 确保立即刷新缓冲区
                    print(content_piece, end="", flush=True)
                    content.append(content_piece)

            # 将所有 chunk 拼接为完整字符串
            content = "".join(content)
            reasoning_content = "".join(reasoning_content)

        else:
            # 非流式模式：直接获取完整响应
            resp = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=False,
            )

            message = resp.choices[0].message
            content = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None)

        # 保存本轮推理内容，便于后续分析 LLM 的思考过程
        reasoning_contents.append(reasoning_content)

        # 将 LLM 的回复追加到对话历史中，维护完整的上下文
        messages.append({"role": "assistant", "content": content})

        # ----- 步骤 3：解析 LLM 回复，判断下一步动作 -----
        content_json = llm_json_parser(content)

        # 如果 LLM 返回了 final_answer，说明任务已完成，跳出循环
        if content_json.get("final_answer"):
            break

        # ----- 步骤 4：执行工具调用（Action） -----
        tool_call = content_json.get("tool_call")

        if tool_call:
            tool_name = tool_call["name"]  # 工具名称
            tool_arguments = tool_call["arguments"]  # 工具参数

            # 从注册表中查找对应的工具函数
            tool_fn = tool_registry.get(tool_name)
            if tool_fn is None:
                raise ValueError(f"Unknown tool:{tool_name}")

            # 调用工具并获取结果（Observation）
            try:
                tool_result = tool_fn(**tool_arguments)
            except Exception as e:
                tool_result = {"ok": False, "err": f"{type(e).__name__}: {e}"}

            # ----- 步骤 5：将工具结果作为 "观察" 反馈给 LLM -----
            # 以 user 角色追加，让 LLM 在下一轮推理时能看到工具的执行结果
            messages.append(
                {"role": "user", "content": json.dumps({"tool_result": tool_result})}
            )

    # ===== 循环结束，输出完整的对话记录和推理过程 =====
    logger.info(
        "完整的对话内容:\n %s", json.dumps(messages, ensure_ascii=False, indent=2)
    )
    logger.info(
        "Reasoning contents:\n %s",
        json.dumps(reasoning_contents, ensure_ascii=False, indent=2),
    )


main()
