# 📖 如何阅读 `realistic_multi_agent_graph_v2.py`

> [!TIP]
> **核心思路：不要从第1行读到第625行。** 这个文件有 625 行，但它的"骨架"只有 20 行。先抓骨架，再逐层填肉。

---

## 一、为什么读完会觉得"水流过去"？

因为这个文件把 **四个完全不同层次的东西** 混在了一个平面上：

| 层次 | 内容 | 行数 | 你需要的关注度 |
|------|------|------|---------------|
| 🔧 基础设施 | JSON解析、LLM调用、命令执行 | ~170行 | ⭐ 扫一眼就行 |
| 📋 状态 + 提示词 | State 定义 + 5个 Prompt | ~90行 | ⭐⭐ 理解字段含义 |
| 🧠 节点逻辑 | 5个 node_xxx 函数 | ~180行 | ⭐⭐⭐ 重点读 |
| 🔗 连线 + 运行 | build + make_state + main | ~100行 | ⭐⭐⭐⭐ 最先读 |

**问题出在**：你按照文件顺序从上往下读，先碰到了那 170 行基础设施代码，脑子已经占满了，等到真正重要的东西出来时已经没有精力了。

---

## 二、推荐阅读顺序：从骨架到细节

### 第① 步：先看 build 函数（整个程序的骨架）

**直接跳到** [build_multi_agent_v2](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L525-L547)

```python
def build_multi_agent_v2() -> GraphFlow:
    g = GraphFlow()

    g.add_node("planner", node_planner)      # 制定计划
    g.add_node("coder", node_coder)          # 写代码
    g.add_node("reviewer", node_reviewer)    # 审查代码
    g.add_node("tester", node_tester)        # 测试代码
    g.add_node("arbiter", node_arbiter)      # 最终裁决

    g.add_edge("planner", "coder")           # 计划 → 写码
    g.add_edge("coder", "reviewer")          # 写码 → 审查
    g.add_edge("reviewer", "tester")         # 审查 → 测试
    g.add_edge("tester", "arbiter")          # 测试 → 裁决

    g.add_conditional_edges("arbiter", route_after_arbiter, {
        "coder": "coder",   # 不通过 → 回到写码
        END: END,           # 通过 → 结束
    })

    g.set_start("planner")
    return g
```

**读完这段你应该能回答：**
- ✅ 整个系统有几个角色？（5个）
- ✅ 数据按什么顺序流动？（planner → coder → reviewer → tester → arbiter）
- ✅ 在哪里有分支？（arbiter 之后，通过→结束，不通过→回到 coder）

> [!IMPORTANT]
> 这 20 行就是整个 625 行文件的"灵魂"。和 v1 完全一样的结构！

### 第② 步：看路由函数（唯一的决策点）

**跳到** [route_after_arbiter](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L514-L522)

```python
def route_after_arbiter(state) -> str:
    if 验收通过:       return END      # 结束
    if 轮次用完:       return END      # 也结束（防止无限循环）
    return "coder"                     # 否则回到 coder 重写
```

**和 v1 的区别：** v2 多了一个 `max_rounds` 保护，不会无限循环。

### 第③ 步：看 State 定义（共享黑板）

**跳到** [MultiAgentV2State](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L107-L129)

不要试图记住每个字段。用这张图理解谁产生了什么：

```mermaid
graph LR
    subgraph "State（共享黑板）"
        task["task 📝"]
        plan["plan 📋"]
        draft["draft_code 💻"]
        reviews["individual_reviews 📝📝"]
        agg["aggregated_review 📊"]
        tester_d["tester_decision 🧪"]
        test_r["test_report 📄"]
        verdict["verdict ⚖️"]
        rounds["rounds 📚"]
    end

    Planner -->|写入| plan
    Coder -->|写入| draft
    Reviewer -->|写入| reviews
    Reviewer -->|写入| agg
    Tester -->|写入| tester_d
    Tester -->|写入| test_r
    Arbiter -->|写入| verdict
    Arbiter -->|写入| rounds

    task -->|读取| Planner
    plan -->|读取| Coder
    draft -->|读取| Reviewer
    agg -->|读取| Tester
    agg -->|读取| Arbiter
    test_r -->|读取| Arbiter
```

**关键理解：** 每个 node 就是"从黑板上读一些东西 → 调 LLM → 把结果写回黑板"。

### 第④ 步：逐个看 node 函数

现在带着"这个角色从黑板读了什么、写了什么"的问题，依次读每个 node：

#### 🏗️ node_planner（L330-338）
[查看源码](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L330-L338)

```
读取: task
写入: plan
一句话: 把任务拆成计划
```

#### 💻 node_coder（L341-357）
[查看源码](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L341-L357)

```
读取: task, plan, rounds（历史反馈）
写入: coding_result, draft_code, round_index++
一句话: 根据计划和历史反馈写代码
```

#### 🔍 node_reviewer（L381-416）
[查看源码](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L381-L416)

```
读取: task, plan, draft_code
写入: individual_reviews, aggregated_review
一句话: 2个审查员分别审查，然后聚合结果
```

> [!NOTE]
> 这是 v2 和 v1 差别最大的节点。v1 只有1个审查员直接返回 approved/issues。v2 模拟了"两个人分别审查 + 投票聚合"的真实场景。

#### 🧪 node_tester（L419-447）
[查看源码](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L419-L447)

```
读取: task, plan, draft_code, aggregated_review
写入: tester_decision, test_report
一句话: 决定要不要跑测试 → 如果要跑就执行命令 → 记录结果
```

#### ⚖️ node_arbiter（L464-511）
[查看源码](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L464-L511)

```
读取: task, plan, coding_result, aggregated_review, test_report
写入: verdict, rounds（存档本轮记录）, final_code（如果通过）
一句话: 综合所有信息做最终裁决，并存档本轮历史
```

> [!NOTE]
> arbiter 在调 LLM 之前会先过一遍 `_hard_gate`（硬规则检查）。这是 v2 新增的——即使 LLM 说"通过"，如果有 high 级别问题或测试失败，硬规则会否决。

### 第⑤ 步：最后再扫基础设施

现在你已经理解了整个流程，再回头扫这些工具类：

| 类/函数 | 行号 | 一句话作用 |
|---------|------|----------|
| [TaskSpec](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L99-L104) | 99-104 | 任务的结构化描述（比 v1 的纯字符串更规范） |
| [JsonUtils](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L131-L161) | 131-161 | 容错地解析 LLM 返回的 JSON |
| [LLMGateway](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L164-L212) | 164-212 | 封装 OpenAI 调用 + 重试逻辑 |
| [SafeCommandRunner](file:///d:/Projects/LLM/src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py#L215-L291) | 215-291 | 安全地执行测试命令（白名单 + 超时） |

这些都是"管道工程"，不影响你理解业务流程。

---

## 三、v1 → v2 到底改了什么？

v2 不是重写，是在 v1 同样的流程上 **加了工程健壮性**。

| 维度 | v1 (271行) | v2 (625行) | 多出来的行数干了什么 |
|------|-----------|-----------|-------------------|
| **任务描述** | 纯字符串 `task: str` | 结构化 `TaskSpec` dataclass | 更规范，但流程不变 |
| **LLM调用** | 裸调 `ask_json()`，无重试 | `LLMGateway` 类，有重试+超时 | 防止网络抖动 |
| **JSON解析** | 简单 `json.loads` | `JsonUtils` 容错解析 | 处理 LLM 返回格式不标准 |
| **审查** | 1个审查员 | 2个审查员 + 聚合 + severity 统计 | 模拟真实 code review |
| **命令执行** | 简单黑名单 | 白名单 + 超时 + 详细报告 | 更安全 |
| **裁决** | 纯 LLM 判断 | 硬规则 `_hard_gate` + LLM 判断 | 防止 LLM 误判 |
| **循环控制** | 无限制（靠 max_steps） | `max_rounds` 限制 | 更精确的控制 |
| **历史** | `history: list` 简单列表 | `rounds: list` 完整存档 | 信息更丰富 |
| **图的骨架** | ✅ 完全相同 | ✅ 完全相同 | — |

> [!IMPORTANT]
> **核心认知：v2 的图骨架和 v1 完全一样。** 多出来的 350 行全是"让每个环节更健壮"的工程代码，不是新的流程。

---

## 四、读代码的通用方法论

以后碰到任何大文件，可以用这个套路：

### 1. 🔭 找入口，先不读细节

```
找 main() → 看它调了什么 → 找到 build 函数 → 看图的骨架
```

### 2. 🗺️ 画数据流图

```
每个节点：读了什么 → 做了什么 → 写了什么
先用一句话概括，不要看实现
```

### 3. 📐 分层，给每层打标签

```
这段是基础设施？跳过。
这段是业务逻辑？重点读。
这段是连线？先读。
```

### 4. 🔄 和上一个版本做 diff

```
结构一样的部分快速跳过
真正新增的逻辑才是需要理解的
```

### 5. ❓ 带着问题读，不要"通读"

每进一个函数前先问自己：
- 这个角色的职责是什么？
- 它需要什么输入？
- 它产出什么输出？
- 它和 v1 的区别是什么？

读完能回答这4个问题就算"吸收了"，不需要记住每一行。

---

## 五、快速自测

不看代码，试着回答：

1. 整个系统有几种角色？分别是？
2. 数据按什么顺序在角色之间流动？
3. 什么时候循环结束？（两个条件）
4. reviewer 和 v1 相比最大的变化是？
5. `_hard_gate` 的作用是什么？
6. 如果 LLM 说"通过"但有 high severity 问题，会怎样？

如果能回答 4/6 以上，说明你已经吸收了这个文件的核心。剩下的工程细节用到时再查。

