import json
import os
import re
from typing import Literal, Optional, TypedDict, List, Dict

from openai import OpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.64.0.4:8080/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "1234567890")
MODEL = os.getenv("OPENAI_MODEL", "Qwen3.5-27B-Opus4.6-Q4_K_M.gguf")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

Intention = Literal["weather", "joke", "chat"]
NodeName = Literal["wait_user", "understand", "route", "weather", "joke", "chat", "end"]


class State(TypedDict):
    messages: List[Dict[str, str]]  # 共享对话上下文（关键）
    user_input: str
    intention: Optional[Intention]
    response: Optional[str]
    next_state: NodeName
    done: bool
    error: Optional[str]
    turn_count: int
    max_turns: int


def llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 256,
) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def classify_intention_from_history(state: State) -> Intention:
    history = state["messages"][-10:]
    classify_messages = [
        {
            "role": "system",
            "content": (
                "你是意图分类器。只能输出 JSON，格式为 "
                '{"intention":"weather"}，且 intention 只能是 weather/joke/chat。'
            ),
        },
        *history,
        {
            "role": "user",
            "content": "请根据以上对话，判断我最后一句话的意图。",
        },
    ]
    raw = llm_chat(classify_messages, temperature=0.0, max_tokens=64)
    parsed = parse_json_object(raw)
    if parsed and parsed.get("intention") in {"weather", "joke", "chat"}:
        return parsed["intention"]  # type: ignore[return-value]

    # 兜底
    lower = state["user_input"].lower()
    if any(
        k in lower for k in ["天气", "下雨", "温度", "weather", "rain", "temperature"]
    ):
        return "weather"
    if any(k in lower for k in ["笑话", "段子", "joke", "funny"]):
        return "joke"
    return "chat"


def node_wait_user(state: State) -> State:
    text = input("你: ").strip()
    state["user_input"] = text
    state["error"] = None
    state["response"] = None

    if text.lower() in {"exit", "quit", "bye", "退出", "再见"}:
        state["response"] = "好的，我们下次聊。"
        state["next_state"] = "end"
        return state

    if not text:
        state["response"] = "请输入内容（输入 退出 可结束会话）。"
        state["next_state"] = "wait_user"
        return state

    state["messages"].append({"role": "user", "content": text})
    state["next_state"] = "understand"
    return state


def node_understand(state: State) -> State:
    try:
        state["intention"] = classify_intention_from_history(state)
        state["next_state"] = "route"
    except Exception as e:
        state["error"] = f"意图识别失败: {e}"
        state["next_state"] = "end"
    return state


def node_route(state: State) -> State:
    if state["intention"] == "weather":
        state["next_state"] = "weather"
    elif state["intention"] == "joke":
        state["next_state"] = "joke"
    elif state["intention"] == "chat":
        state["next_state"] = "chat"
    else:
        state["error"] = f"未知意图: {state['intention']}"
        state["next_state"] = "end"
    return state


def node_weather(state: State) -> State:
    try:
        msgs = [
            {
                "role": "system",
                "content": (
                    "你在 weather 节点。可基于上下文回答天气相关问题；"
                    "若无实时数据，明确说明并给建议，不编造精确实时天气。"
                ),
            },
            *state["messages"][-12:],
        ]
        reply = llm_chat(msgs, temperature=0.3, max_tokens=220)
        state["response"] = reply
        state["messages"].append({"role": "assistant", "content": reply})
    except Exception as e:
        state["error"] = f"weather 节点失败: {e}"
    finally:
        state["turn_count"] += 1
        state["next_state"] = (
            "end" if state["turn_count"] >= state["max_turns"] else "wait_user"
        )
    return state


def node_joke(state: State) -> State:
    try:
        msgs = [
            {
                "role": "system",
                "content": "你在 joke 节点。讲一个中文轻松笑话，1-3 句，不冒犯。",
            },
            *state["messages"][-12:],
        ]
        reply = llm_chat(msgs, temperature=0.8, max_tokens=120)
        state["response"] = reply
        state["messages"].append({"role": "assistant", "content": reply})
    except Exception as e:
        state["error"] = f"joke 节点失败: {e}"
    finally:
        state["turn_count"] += 1
        state["next_state"] = (
            "end" if state["turn_count"] >= state["max_turns"] else "wait_user"
        )
    return state


def node_chat(state: State) -> State:
    try:
        msgs = [
            {
                "role": "system",
                "content": "你在 chat 节点。进行自然、简洁、有帮助的中文对话。",
            },
            *state["messages"][-12:],
        ]
        reply = llm_chat(msgs, temperature=0.6, max_tokens=220)
        state["response"] = reply
        state["messages"].append({"role": "assistant", "content": reply})
    except Exception as e:
        state["error"] = f"chat 节点失败: {e}"
    finally:
        state["turn_count"] += 1
        state["next_state"] = (
            "end" if state["turn_count"] >= state["max_turns"] else "wait_user"
        )
    return state


def node_end(state: State) -> State:
    state["done"] = True
    return state


def run_agent(max_turns: int = 10) -> State:
    state: State = {
        "messages": [{"role": "system", "content": "你是一个有帮助的中文助手。"}],
        "user_input": "",
        "intention": None,
        "response": None,
        "next_state": "wait_user",
        "done": False,
        "error": None,
        "turn_count": 0,
        "max_turns": max_turns,
    }

    nodes = {
        "wait_user": node_wait_user,
        "understand": node_understand,
        "route": node_route,
        "weather": node_weather,
        "joke": node_joke,
        "chat": node_chat,
        "end": node_end,
    }

    max_steps = 200
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

        if state["response"]:
            print(f"助手: {state['response']}")

        if state["error"]:
            print(f"[错误] {state['error']}")

    return state


if __name__ == "__main__":
    final_state = run_agent(max_turns=10)
    print("\n=== Session End ===")
    print("turn_count:", final_state["turn_count"])
    print("error:", final_state["error"])
