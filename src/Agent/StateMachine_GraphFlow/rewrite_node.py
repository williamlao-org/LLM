"""
GraphFlow 节点定义

现代 Agent 只需要 3 个核心节点：
1. node_entry    — 把用户输入包装成 messages
2. node_llm      — 调用大模型（带 tools），模型自己决定调不调工具
3. node_tool_exec — 执行模型要求的工具，把结果喂回 messages
"""

import json
from .rewrite_graph import State

from openai import OpenAI

client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key="sk-ougrusqbfdllgscdzvlaknmfcothjdevxgdsxjzmvopvinhtlff")
MODEL = "zai-org/GLM-4.6"

# ============================================================
# 工具注册表：想加新工具只需在这里加一项
# 这就是"意图列表"的现代替代品
# 模型会自动根据 description 判断该不该调用
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 北京"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如 3*5+2",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# 工具的具体实现（模型不知道这些代码，它只看上面的 description）
# ============================================================
def _exec_get_weather(city: str) -> str:
    """假的天气 API，真实项目换成 requests 调用"""
    fake_data = {"北京": "晴，23°C", "上海": "多云，19°C"}
    return fake_data.get(city, f"{city}：暂无数据")


def _exec_calculator(expression: str) -> str:
    try:
        return str(eval(expression))  # 教学用，生产环境不要用 eval
    except Exception as e:
        return f"计算错误: {e}"


# 工具名 -> 实现函数 的映射
TOOL_MAP = {
    "get_weather": _exec_get_weather,
    "calculator": _exec_calculator,
}


# ============================================================
# 节点 1：入口 — 初始化 messages
# ============================================================
def node_entry(state: State):
    """把 user_input 包装成标准 messages 格式"""
    # 首轮：初始化 system prompt
    if not state["messages"]:
        state["messages"].append(
            {"role": "system", "content": "你是一个智能助手，可以回答用户问题。如果需要，你可以调用工具。"}
        )

    # 每轮开头插入分隔符，方便按轮次切分 trace
    state["trace"].append("---")

    # 每轮：追加用户消息
    state["messages"].append({"role": "user", "content": state["user_input"]})
    state["tool_calls"] = []
    state["done"] = False
    state["trace"].append(f"entry: 用户说 '{state['user_input']}'")


# ============================================================
# 节点 2：LLM — 调用大模型，模型自己决定：直接回复 or 调工具
# ============================================================
def node_llm(state: State):
    """
    调用 LLM（带 tools），两种可能的结果：
    - 模型直接回复 → state["done"] = True
    - 模型要求调用工具 → state["tool_calls"] 里会有内容
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=state["messages"],
        tools=TOOLS,
        tool_choice="auto",  # 让模型自己判断要不要调工具
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        # 模型决定调用工具
        state["tool_calls"] = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,  # JSON 字符串
            }
            for tc in msg.tool_calls
        ]
        # 把模型的这次回复（含 tool_calls）也存入历史
        state["messages"].append(msg.model_dump())
        state["done"] = False
        state["trace"].append(f"llm: 决定调用工具 {[tc['name'] for tc in state['tool_calls']]}")
    else:
        # 模型直接回复，不需要工具
        state["answer"] = msg.content
        state["messages"].append({"role": "assistant", "content": msg.content})
        state["done"] = True
        state["trace"].append("llm: 直接回复，不需要工具")


# ============================================================
# 节点 3：工具执行 — 跑完工具后把结果塞回 messages
# ============================================================
def node_tool_exec(state: State):
    """执行 tool_calls 里的每个工具，把结果以 tool message 格式存回 messages"""
    for tc in state["tool_calls"]:
        func = TOOL_MAP.get(tc["name"])
        if func is None:
            result = f"未知工具: {tc['name']}"
        else:
            args = json.loads(tc["arguments"])
            result = func(**args)

        # OpenAI 格式：工具结果要用 role="tool" 存回去
        state["messages"].append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": str(result),
        })
        state["trace"].append(f"tool: {tc['name']}({tc['arguments']}) → {result}")

    # 清空，等下一轮 LLM 再决定
    state["tool_calls"] = []
