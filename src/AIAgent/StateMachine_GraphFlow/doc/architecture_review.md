# GraphFlow 架构审查：对比 2026 生产标准

## 三大框架的共识

搜索了 LangGraph、OpenAI Agents SDK、Google ADK 三家的最新架构，核心共识如下：

### 标准 ReAct 循环（所有框架都用这个）

```
Agent Node (LLM) → 有 tool_calls？ → 是 → Tool Node → 回到 Agent Node
                                   → 否 → END
```

> [!IMPORTANT]
> **这就是我们现在 `llm_tools ←→ tool_exec` 的循环，完全一致。✅**

### Router 的真实用法

LangGraph 官方推荐的 Router 并不是一个单独调 LLM 来分类的节点，
而是一个**纯函数**，根据当前 state 的内容来决定下一步走哪。

```python
# LangGraph 官方写法
def route_after_agent(state) -> Literal["tools", END]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", route_after_agent)
```

> [!WARNING]
> **我们当前的 `node_router` 每次都调一次 LLM 来分类，这是多余的开销。**
> 生产系统中，简单路由用 **纯函数**，不用 LLM。

---

## ✅ 已对齐生产标准的部分

| 项目 | 我们的实现 | 生产标准 | 状态 |
|---|---|---|---|
| State 驱动 | `TypedDict` + 全局状态 | `TypedDict` 或 Pydantic | ✅ |
| ReAct 循环 | `llm_tools ←→ tool_exec` | `Agent ←→ ToolNode` | ✅ |
| Tool Calling | OpenAI 格式 + tool_choice=auto | 相同 | ✅ |
| 工具注册表 | `TOOLS` 列表 + `TOOL_MAP` | 相同模式 | ✅ |
| 无边则停 | `next_node is None → break` | `return END` | ✅ 等价 |
| Trace/可观察性 | `state["trace"]` | LangSmith / OpenTelemetry | ✅ 简化版 |
| max_steps 防护 | `for _ in range(max_steps)` | 相同 | ✅ |

## ❌ 需要修正的部分

### 1. Router 不应该调 LLM

**问题：** 我们的 `node_router` 每次调一次 LLM 做分类，浪费 token 和时间。

**生产做法：** Router 是纯函数，检查 state 或 messages 的内容来路由。
如果用户真的需要意图分类，让 `llm_tools` 节点自己处理——它本身就能通过 tool calling 来"路由"。

**建议：** 去掉 `node_router` 和 `node_chat`，改回简单的 ReAct 循环。
需要 guardrail 时把 reject 做成纯函数检查，不调模型。

### 2. 缺少 `add_conditional_edges` 方法

**问题：** 我们把路由函数和边分开写，不够清晰。

**LangGraph 做法：** 用 `add_conditional_edges` 把"从哪出发"和"路由函数"绑在一起。

```python
# LangGraph 风格
graph.add_conditional_edges(
    "llm",                    # 从 llm 出发
    route_function,           # 路由函数
    {"tools": "tool_exec", END: END}  # 映射表
)
```

## ⭐ 可选进阶（当前不必加，了解即可）

| 功能 | 说明 | 何时需要 |
|---|---|---|
| 持久化 Checkpointer | 崩溃后从上次断点恢复 | 长任务、生产部署 |
| Human-in-the-Loop | 暂停等人类审批 | 高风险操作 |
| Supervisor-Worker | 多 Agent 协作 | 复杂多步骤任务 |
| Scatter-Gather | 并行执行多个工具 | 需要同时查多个 API |
| Reflection/Self-Correction | 生成后自检质量 | 对输出质量要求高 |

---

## 建议的最终架构

```
entry → guardrail(纯函数) ──→ 安全？
                            ├── 否 → reject(纯函数) → 停
                            └── 是 → llm ←→ tool_exec → 停
```

**关键变化：**
- **去掉 `node_router`**（LLM 路由），改成纯函数 guardrail
- **去掉 `node_chat`**，统一走 `llm` 节点（不传 tools 时模型自然就是纯聊天）
- 保持 ReAct 循环不变
