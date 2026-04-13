"""
主入口：组装 GraphFlow 并运行

流程图：
    node_entry → node_llm → 模型要调工具吗？
                    ↑           ├── 是 → node_tool_exec → 回到 node_llm
                    └───────────┘
                                └── 否 → 结束（done=True）
"""

import json
from .rewrite_graph import GraphFlow, Node, State
from .rewrite_node import node_entry, node_llm, node_tool_exec


def build_agent() -> GraphFlow:
    g = GraphFlow()

    # 注册节点（注意：Node 的 edges 初始为空，之后用 add_edge 连）
    g.add_node(Node(name="entry", func=node_entry, edges=[]))
    g.add_node(Node(name="llm", func=node_llm, edges=[]))
    g.add_node(Node(name="tool_exec", func=node_tool_exec, edges=[]))

    # 连边
    # entry → llm（无条件）
    g.add_edge("entry", "llm", condition=None)

    # llm → tool_exec（如果模型要调工具）
    g.add_edge("llm", "tool_exec", condition=lambda state: not state["done"] and len(state["tool_calls"]) > 0)

    # tool_exec → llm（工具执行完，回到模型继续思考）
    g.add_edge("tool_exec", "llm", condition=None)

    # 设置起点
    g.set_start_node("entry")

    return g


def run_once(text: str) -> State:
    """单轮独立运行（不保留历史，用于测试）"""
    agent = build_agent()
    state: State = {
        "user_input": text,
        "messages": [],
        "tool_calls": [],
        "answer": "",
        "done": False,
        "trace": [],
    }
    return agent.run(state, max_steps=10)


if __name__ == "__main__":
    print("=== GraphFlow Agent (Tool Calling 版) ===")
    print("输入 exit 退出\n")

    agent = build_agent()

    # state 只创建一次，跨轮次保持 messages 和 trace
    state: State = {
        "user_input": "",
        "messages": [],
        "tool_calls": [],
        "answer": "",
        "done": False,
        "trace": [],
    }

    turn = 0
    while True:
        q = input("你: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # 每轮只更新用户输入，其他状态保持
        turn += 1
        state["user_input"] = q
        state = agent.run(state, max_steps=10)

        # 只打印当前轮的 trace（从最后一个 '---' 之后）
        last_sep = len(state["trace"]) - 1 - state["trace"][::-1].index("---")
        current_trace = state["trace"][last_sep + 1:]

        print(f"\n助手: {state['answer']}")
        print(f"\n--- 第 {turn} 轮 trace ---")
        for t in current_trace:
            print(f"  {t}")
        print()

    # 退出时打印完整历史
    if state["trace"]:
        print("\n=== 完整 trace 历史 ===")
        round_num = 0
        for t in state["trace"]:
            if t == "---":
                round_num += 1
                print(f"\n  [第 {round_num} 轮]")
            else:
                print(f"    {t}")