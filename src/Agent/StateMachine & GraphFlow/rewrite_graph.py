from ast import Dict
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypedDict
from openai import OpenAI


# 1. 定义状态 (State)
class State(TypedDict):
    user_input: str  # 用户输入的原始文本
    intent: str  # 意图识别结果
    confidence: float  # 意图置信度
    tool_result: Optional[dict[str, Any]]  # 工具调用的结果
    messages: List[dict[str, Any]]  # 消息列表
    draft: str  # 生成的草稿
    answer: str  # 最终返回给用户的答案
    rejected: bool  # 是否触发了安全审核
    trace: List[str]  # 运行轨迹（用于调试）


NodeFunc = Callable[[State], None]
ConditionFunc = Callable[[State], bool]


@dataclass
class Edge:
    condition: ConditionFunc  # 什么时候走这条路（默认可以是 lambda _: True）
    target_node_name: str  # 下一个节点的名称


@dataclass
class Node:
    name: str  # 节点名称
    func: NodeFunc  # 该节点要执行的具体函数
    edges: List[Edge]  # 从这个节点出发能够通往哪里


class GraphFlow:
    def __init__(self):
        self.start_node_name: Optional[str] = None
        self.end_node_name: Optional[str] = None
        self.nodes_dict: dict[str, Node] = {}

    def run(self, state: State, max_steps: int):
        if self.start_node_name is None:
            raise ValueError("start_node_name is not set")

        current_node_name = self.start_node_name
        step = 0

        while current_node_name and step < max_steps:
            node = self.nodes_dict.get(current_node_name)
            if node is None:
                raise ValueError(f"Node '{current_node_name}' not found")

            step += 1
            node.func(state)
            for edge in node.edges:
                if edge.condition(state):
                    current_node_name = edge.target_node_name
                    break

            if current_node_name == self.end_node_name:
                break

        return state

    def add_node(self, node: Node):
        self.nodes_dict[node.name] = node

    def add_edge(self, from_node_name, to_node_name, condition: ConditionFunc | None):
        if condition is None:
            condition = lambda _: True

        edge = Edge(condition, to_node_name)
        self.nodes_dict[from_node_name].edges.append(edge)

    def set_start_node(self, name):
        self.start_node_name = name

    def set_end_node(self, name):
        self.end_node_name = name


    