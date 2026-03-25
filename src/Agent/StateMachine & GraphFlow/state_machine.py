from typing import Literal, TypedDict, Optional


# ==========================================
# 1. 定义核心状态 (State)
# ==========================================


class State(TypedDict):
    user_input: str  # 用户的初始输入
    intention: Optional[Literal["weather", "joke", "chat"]]  # 用户的意图
    response: Optional[str]
    next_state: str
    done: bool
    error: Optional[str]


# ==========================================
# 2. 定义节点函数骨架 (每个节点只做一件事)
# ==========================================


def node_understand(state: State) -> State:
    """理解用户输入，识别意图"""
    text = state["user_input"].strip().lower()

    # 每次重新理解前，先清理旧结果
    state["error"] = None
    state["response"] = None

    weather_keywords = ["天气", "下雨", "温度", "weather", "rain", "temperature"]
    joke_keywords = ["笑话", "段子", "joke", "funny"]

    if any(k in text for k in weather_keywords):
        state["intention"] = "weather"
    elif any(k in text for k in joke_keywords):
        state["intention"] = "joke"
    else:
        state["intention"] = "chat"

    # 理解完成后，下一步固定进入路由节点
    state["next_state"] = "route"
    return state


def node_route(state: State) -> State:
    """根据意图设置 next_state"""
    intention = state["intention"]

    if intention == "weather":
        state["next_state"] = "weather"
    elif intention == "joke":
        state["next_state"] = "joke"
    elif intention == "chat":
        state["next_state"] = "chat"
    else:
        # 理论上不会进来，做一层保护
        state["error"] = f"未知意图: {intention}"
        state["next_state"] = "end"

    return state


def node_weather(state: State) -> State:
    """处理天气查询"""
    state["response"] = "今天天气晴，气温 24°C，适合出门散步。"
    state["next_state"] = "end"
    return state


def node_joke(state: State) -> State:
    """处理讲笑话请求"""
    state["response"] = "给你一个冷笑话：程序员最怕两件事，1. 这也能跑？2. 昨天还能跑。"
    state["next_state"] = "end"
    return state


def node_chat(state: State) -> State:
    """处理兜底闲聊"""
    state["response"] = "我可以帮你查天气、讲笑话，或者陪你聊聊。"
    state["next_state"] = "end"
    return state


def node_end(state: State) -> State:
    """结束节点"""
    state["done"] = True
    return state


# ==========================================
# 3. 状态机引擎
# ==========================================

def run_agent(user_input: str) -> State:
    state: State = {
        "user_input": user_input,
        "intention": None,
        "response": None,
        "next_state": "understand",
        "done": False,
        "error": None,
    }

    nodes = {
        "understand": node_understand,
        "route": node_route,
        "weather": node_weather,
        "joke": node_joke,
        "chat": node_chat,
        "end": node_end,
    }

    max_steps = 20
    step_count = 0

    while not state["done"]:
        step_count += 1
        if step_count > max_steps:
            state["error"] = "状态机超过最大步数，疑似死循环"
            state["next_state"] = "end"

        current = state["next_state"]
        node_fn = nodes.get(current)

        if node_fn is None:
            state["error"] = f"未知节点: {current}"
            state["next_state"] = "end"
            continue

        state = node_fn(state)

    return state


if __name__ == "__main__":
    text = input("请输入一句话: ")
    final_state = run_agent(text)

    print("\n=== Agent Result ===")
    print("intention:", final_state["intention"])
    print("response:", final_state["response"])
    print("error:", final_state["error"])