from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict


class FlowState(TypedDict):
    user_input: str
    intent: str
    confidence: float
    tool_result: Optional[Dict[str, Any]]
    draft: str
    answer: str
    rejected: bool
    trace: List[str]
    loop_count: int


NodeFunc = Callable[[FlowState], None]
Condition = Callable[[FlowState], bool]


@dataclass
class Edge:
    to_node: str
    condition: Condition
    reason: str


class GraphFlow:
    """A tiny graph execution engine with conditional edges."""

    def __init__(self) -> None:
        self.nodes: Dict[str, NodeFunc] = {}
        self.edges: Dict[str, List[Edge]] = {}
        self.start_node: Optional[str] = None
        self.end_nodes: set[str] = set()

    def add_node(self, name: str, fn: NodeFunc, is_end: bool = False) -> None:
        self.nodes[name] = fn
        self.edges.setdefault(name, [])
        if is_end:
            self.end_nodes.add(name)

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Condition] = None,
        reason: str = "",
    ) -> None:
        if condition is None:
            condition = lambda _: True
        self.edges.setdefault(from_node, []).append(
            Edge(to_node=to_node, condition=condition, reason=reason or "match")
        )

    def set_start(self, name: str) -> None:
        self.start_node = name

    def _next_node(self, current: str, state: FlowState) -> Optional[str]:
        for edge in self.edges.get(current, []):
            if edge.condition(state):
                state["trace"].append(f"{current} -> {edge.to_node} ({edge.reason})")
                return edge.to_node
        return None

    def run(self, state: FlowState, max_steps: int = 20) -> FlowState:
        if self.start_node is None:
            raise ValueError("Start node is not set")

        current = self.start_node
        steps = 0

        while steps < max_steps:
            steps += 1
            state["trace"].append(f"enter:{current}")

            node = self.nodes.get(current)
            if node is None:
                raise ValueError(f"Unknown node: {current}")

            node(state)

            if current in self.end_nodes:
                state["trace"].append(f"end:{current}")
                return state

            nxt = self._next_node(current, state)
            if nxt is None:
                state["trace"].append(f"stop:no_edge_from:{current}")
                return state

            current = nxt

        state["answer"] = "流程超过最大步数，已安全停止。"
        state["trace"].append("stop:max_steps")
        return state


def detect_intent(text: str) -> tuple[str, float]:
    t = text.strip().lower()
    if any(k in t for k in ["天气", "weather", "下雨", "温度"]):
        return "ask_weather", 0.92
    if any(k in t for k in ["你好", "hi", "hello"]):
        return "greet", 0.95
    if any(k in t for k in ["炸弹", "攻击", "违法"]):
        return "unsafe", 0.99
    return "general", 0.55


def tool_weather_fake() -> Dict[str, Any]:
    return {
        "city": "北京",
        "temperature": 23,
        "desc": "多云",
    }


def n_understand(state: FlowState) -> None:
    intent, confidence = detect_intent(state["user_input"])
    state["intent"] = intent
    state["confidence"] = confidence


def n_tool_weather(state: FlowState) -> None:
    state["tool_result"] = tool_weather_fake()


def n_generate(state: FlowState) -> None:
    intent = state["intent"]
    text = state["user_input"]

    if intent == "greet":
        state["draft"] = "你好呀，很高兴见到你。你想聊点什么？"
        return

    if intent == "ask_weather" and state["tool_result"]:
        w = state["tool_result"]
        state["draft"] = (
            f"{w['city']} 当前 {w['desc']}，气温约 {w['temperature']}°C。"
            "如果你要出门，我也可以给你穿衣建议。"
        )
        return

    state["draft"] = f"你说的是：{text}。我可以继续帮你细化这个问题。"


def n_guardrail(state: FlowState) -> None:
    banned = ["炸弹", "攻击", "违法"]
    if any(k in state["draft"] or k in state["user_input"] for k in banned):
        state["rejected"] = True
        state["answer"] = "这个请求我不能协助，但我可以提供安全、合法的替代建议。"
        return
    state["rejected"] = False
    state["answer"] = state["draft"]


def n_retry(state: FlowState) -> None:
    state["loop_count"] += 1
    if state["loop_count"] > 2:
        state["answer"] = "我暂时无法稳定完成这次请求，请换个问法再试。"


def n_end(_: FlowState) -> None:
    return


def build_demo_flow() -> GraphFlow:
    g = GraphFlow()

    g.add_node("understand", n_understand)
    g.add_node("tool_weather", n_tool_weather)
    g.add_node("generate", n_generate)
    g.add_node("guardrail", n_guardrail)
    g.add_node("retry", n_retry)
    g.add_node("end", n_end, is_end=True)

    g.set_start("understand")

    g.add_edge(
        "understand",
        "retry",
        condition=lambda s: s["confidence"] < 0.4,
        reason="low_confidence",
    )
    g.add_edge(
        "understand",
        "tool_weather",
        condition=lambda s: s["intent"] == "ask_weather",
        reason="need_weather_tool",
    )
    g.add_edge(
        "understand",
        "generate",
        condition=lambda _: True,
        reason="default_generate",
    )

    g.add_edge("tool_weather", "generate", reason="tool_ready")
    g.add_edge("generate", "guardrail", reason="draft_ready")

    g.add_edge(
        "guardrail",
        "end",
        condition=lambda s: not s["rejected"],
        reason="safe_answer",
    )
    g.add_edge(
        "guardrail",
        "end",
        condition=lambda s: s["rejected"],
        reason="blocked_answer",
    )

    g.add_edge(
        "retry",
        "understand",
        condition=lambda s: s["loop_count"] <= 2,
        reason="retry_again",
    )
    g.add_edge(
        "retry",
        "end",
        condition=lambda s: s["loop_count"] > 2,
        reason="retry_exhausted",
    )

    return g


def run_once(text: str) -> FlowState:
    flow = build_demo_flow()
    state: FlowState = {
        "user_input": text,
        "intent": "general",
        "confidence": 0.0,
        "tool_result": None,
        "draft": "",
        "answer": "",
        "rejected": False,
        "trace": [],
        "loop_count": 0,
    }
    return flow.run(state)


if __name__ == "__main__":
    print("Graph Flow Demo，输入 exit 退出")
    while True:
        q = input("你: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = run_once(q)
        print("助手:", result["answer"])
        print("trace:")
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
