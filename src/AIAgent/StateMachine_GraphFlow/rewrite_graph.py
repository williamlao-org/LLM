"""
图引擎 + 状态定义

纯框架代码，不包含任何业务逻辑。
对标 LangGraph 的核心概念：State / Node / Edge / ConditionalEdge / END
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TypedDict

# 特殊标记：路由函数返回 END 表示流程结束
END = "__END__"


class State(TypedDict):
    user_input: str                       # 用户输入的原始文本
    messages: List[dict[str, Any]]        # 对话历史 (OpenAI 格式)
    tool_calls: List[dict[str, Any]]      # 模型返回的 tool_call 列表
    answer: str                           # 最终返回给用户的答案
    trace: List[str]                      # 运行轨迹（用于调试）


NodeFunc = Callable[[State], None]
RouteFunc = Callable[[State], str]        # 路由函数：返回节点名或 END


@dataclass
class Edge:
    """普通边：无条件跳转"""
    target: str


@dataclass
class ConditionalEdge:
    """条件边：根据路由函数的返回值决定下一个节点"""
    route_func: RouteFunc
    route_map: dict[str, str]   # { 路由函数返回值 → 目标节点名 }


@dataclass
class Node:
    name: str
    func: NodeFunc
    edges: List[Edge | ConditionalEdge] = field(default_factory=list)


class GraphFlow:
    def __init__(self):
        self.start_node: Optional[str] = None
        self.nodes: dict[str, Node] = {}

    def add_node(self, name: str, func: NodeFunc):
        """注册节点"""
        self.nodes[name] = Node(name=name, func=func)

    def add_edge(self, from_node: str, to_node: str):
        """无条件边：from → to"""
        self.nodes[from_node].edges.append(Edge(target=to_node))

    def add_conditional_edges(self, from_node: str, route_func: RouteFunc, route_map: dict[str, str]):
        """
        条件边（对标 LangGraph）：

        用法：
            graph.add_conditional_edges(
                "llm",                                      # 从哪个节点出发
                route_after_llm,                            # 纯函数，返回 "tools" 或 END
                {"tools": "tool_exec", END: END}            # 映射表
            )
        """
        self.nodes[from_node].edges.append(
            ConditionalEdge(route_func=route_func, route_map=route_map)
        )

    def set_start(self, name: str):
        self.start_node = name

    def run(self, state: State, max_steps: int = 10) -> State:
        if self.start_node is None:
            raise ValueError("未设置起始节点")

        current = self.start_node
        for _ in range(max_steps):
            node = self.nodes.get(current)
            if node is None:
                raise ValueError(f"节点 '{current}' 不存在")

            state["trace"].append(f"-> {current}")
            node.func(state)

            # 解析下一个节点
            next_node = self._resolve_next(node, state)

            if next_node is None or next_node == END:
                break

            current = next_node

        return state

    def _resolve_next(self, node: Node, state: State) -> Optional[str]:
        """遍历节点的边，找到第一个可走的"""
        for edge in node.edges:
            if isinstance(edge, Edge):
                return edge.target
            elif isinstance(edge, ConditionalEdge):
                route_key = edge.route_func(state)
                target = edge.route_map.get(route_key)
                if target is not None:
                    return target
        return None