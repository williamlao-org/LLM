"""
节点定义

生产级 ReAct Agent 只需要 3 个节点 + 2 个纯函数路由：
1. node_entry      — 初始化 messages + 安全检查
2. node_llm        — 调用大模型（带 tools）
3. node_tool_exec  — 执行工具，结果喂回 messages

路由函数（纯函数，不调模型）：
- route_after_entry  — 入口后的安全检查
- route_after_llm    — LLM 后判断走工具还是结束
"""

import json
from .rewrite_graph import State, END
from .rewrite_tools import TOOLS, execute_tool

from openai import OpenAI

client = OpenAI(base_url="https://api.siliconflow.cn/v1", api_key="sk-ougrusqbfdllgscdzvlaknmfcothjdevxgdsxjzmvopvinhtlff")
MODEL = "zai-org/GLM-4.6"


# ============================================================
# 安全词表（guardrail 用，纯规则，不调模型）
# ============================================================
BANNED_WORDS = ["炸弹", "攻击", "违法", "毒品"]


# ============================================================
# 节点 1：入口
# ============================================================
def node_entry(state: State):
    """初始化 messages，追加用户输入"""
    if not state["messages"]:
        state["messages"].append(
            {"role": "system", "content": "你是一个智能助手，可以回答用户问题。如果需要，你可以调用工具。"}
        )
    state["messages"].append({"role": "user", "content": state["user_input"]})
    state["tool_calls"] = []


# ============================================================
# 路由函数 1：入口后的安全检查（纯函数，不调模型）
# ============================================================
def route_after_entry(state: State) -> str:
    """检查用户输入是否安全"""
    text = state["user_input"]
    if any(word in text for word in BANNED_WORDS):
        state["trace"].append(f"guardrail → 拦截（命中: {[w for w in BANNED_WORDS if w in text]}）")
        return "reject"
    return "safe"


# ============================================================
# 拒绝节点：纯函数，不调模型，零成本
# ============================================================
def node_reject(state: State):
    """直接拒绝，不浪费 token"""
    state["answer"] = "抱歉，这个请求我无法处理。如果你有其他问题，我很乐意帮助。"
    state["messages"].append({"role": "assistant", "content": state["answer"]})


# ============================================================
# 节点 2：LLM（ReAct 核心）
# ============================================================
def node_llm(state: State):
    """调用 LLM（带 tools），模型自己决定直接回复还是调工具"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=state["messages"],
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        # 模型要调工具
        state["tool_calls"] = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in msg.tool_calls
        ]
        state["messages"].append(msg.model_dump())
        state["trace"].append(f"llm → 调用工具 {[tc['name'] for tc in state['tool_calls']]}")
    else:
        # 模型直接回复
        state["answer"] = msg.content
        state["messages"].append({"role": "assistant", "content": msg.content})
        state["trace"].append("llm → 直接回复")


# ============================================================
# 路由函数 2：LLM 后判断走工具还是结束（纯函数）
# ============================================================
def route_after_llm(state: State) -> str:
    """检查 LLM 是否要求调用工具"""
    if state["tool_calls"]:
        return "tools"
    return END


# ============================================================
# 节点 3：工具执行
# ============================================================
def node_tool_exec(state: State):
    """执行工具，结果存回 messages"""
    for tc in state["tool_calls"]:
        result = execute_tool(tc["name"], tc["arguments"])
        state["messages"].append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })
        state["trace"].append(f"tool: {tc['name']}({tc['arguments']}) → {result}")

    state["tool_calls"] = []
