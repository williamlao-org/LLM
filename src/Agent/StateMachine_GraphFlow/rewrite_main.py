"""
主入口：组装图 + 运行

生产级流程图：

    entry → guardrail(纯函数) → 安全？
                                ├── reject → 停
                                └── llm ←→ tool_exec → 停 (END)
"""

from .rewrite_graph import GraphFlow, State, END
from .rewrite_node import (
    node_entry, node_reject, node_llm, node_tool_exec,
    route_after_entry, route_after_llm,
)


def build_agent() -> GraphFlow:
    g = GraphFlow()

    # 注册节点
    g.add_node("entry", node_entry)
    g.add_node("reject", node_reject)
    g.add_node("llm", node_llm)
    g.add_node("tool_exec", node_tool_exec)

    # ---- 连边（一看就懂整个流程）----

    # entry 之后：纯函数检查安全性
    g.add_conditional_edges("entry", route_after_entry, {
        "safe":   "llm",       # 安全 → 进入 ReAct 循环
        "reject": "reject",    # 不安全 → 直接拒绝
    })

    # llm 之后：纯函数检查是否需要调工具
    g.add_conditional_edges("llm", route_after_llm, {
        "tools": "tool_exec",  # 有工具要调
        END:     END,          # 模型直接回复了 → 结束
    })

    # tool_exec 之后：无条件回到 llm（把工具结果喂回去）
    g.add_edge("tool_exec", "llm")

    # reject 没有出边 → 图自动停

    g.set_start("entry")
    return g


def make_state() -> State:
    return {
        "user_input": "",
        "messages": [],
        "tool_calls": [],
        "answer": "",
        "trace": [],
    }


if __name__ == "__main__":
    print("=== GraphFlow Agent (生产级 ReAct) ===")
    print("输入 exit 退出\n")

    agent = build_agent()
    state = make_state()
    turn = 0

    while True:
        q = input("你: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        turn += 1
        state["trace"].append(f"=== 第 {turn} 轮 ===")
        state["user_input"] = q
        state = agent.run(state, max_steps=10)

        print(f"\n助手: {state['answer']}\n")

    # 退出时打印完整 trace
    if state["trace"]:
        print("\n=== 完整 trace ===")
        for t in state["trace"]:
            if t.startswith("==="):
                print(f"\n  {t}")
            else:
                print(f"    {t}")