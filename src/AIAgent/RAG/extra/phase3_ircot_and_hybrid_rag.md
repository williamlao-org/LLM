# IRCoT、Plan-and-Execute 与工程化混合 RAG

## 1. 一句话结论

多跳 RAG 没有唯一正确的控制方式，常见路线可以分为：

```text
单步 RAG             一次检索后回答
Iterative / IRCoT    走一步、检索一步，再决定下一步
Plan-and-Execute     先生成完整计划，再按计划执行
Adaptive Plan        先生成粗计划，执行时允许重试、提前结束和重规划
```

工程上更普适的组合是：

> Workflow 掌握控制权，Agent 提供判断力，Tool 负责确定性执行。

当前项目采用的是 **Adaptive Plan-and-Execute + 强制 CRAG**，同时允许外层 Agent 连续调用单步检索，形成 IRCoT-like 的探索路径。

---

## 2. 标准 IRCoT 是什么

IRCoT 全称 **Interleaving Retrieval with Chain-of-Thought Reasoning**。它的核心不是先生成完整计划，而是将推理和检索交错执行：

```text
原问题检索
  ↓
根据当前证据生成一条中间推理
  ↓
用这条推理继续检索
  ↓
根据新增证据生成下一条推理
  ↓
再次检索
  ↓
证据足够后生成答案
```

关键关系是：

```text
下一次检索什么
    取决于
当前已经推导出了什么
    而当前推导又取决于
前一次检索获得了什么
```

简化伪代码：

```python
def ircot(question: str, max_steps: int = 5):
    evidence = retrieve(question)
    reasoning_steps = []

    for _ in range(max_steps):
        next_step = llm_generate_next_reasoning(
            question=question,
            evidence=evidence,
            previous_steps=reasoning_steps,
        )

        if next_step.is_final_answer:
            return next_step.answer

        reasoning_steps.append(next_step.text)
        new_evidence = retrieve(next_step.text)
        evidence = deduplicate(evidence + new_evidence)

    return generate_final_answer(question, evidence)
```

这里的检索链是**执行过程中逐渐长出来的**，不是提前生成好的。

最终可以有两种收尾：

- 模型在循环中判断证据足够，直接输出最终答案。
- 循环只负责收集证据，最后交给独立 Reader/Generator 统一回答。

参考：

- [IRCoT 论文](https://arxiv.org/abs/2212.10509)
- [IRCoT 官方代码](https://github.com/StonyBrookNLP/ircot)

---

## 3. IRCoT 和 ReAct 的关系

两者都属于“走一步看一步”的循环：

```text
ReAct:
Thought → Action → Observation → Thought

IRCoT:
Reasoning Step → Retrieval → Evidence → Reasoning Step
```

区别主要是动作空间：

- ReAct 可以使用搜索、计算器、数据库、文件等多种工具。
- IRCoT 聚焦知识密集型问答，核心动作基本是检索。
- IRCoT 的中间推理句本身通常也是下一轮检索线索。

因此可以把 IRCoT 看成一种面向多跳检索的专用 ReAct 循环，但不能把所有 ReAct Agent 都称为 IRCoT。

---

## 4. Plan-and-Execute 是什么

Plan-and-Execute 会先生成完整或粗粒度计划：

```text
问题
  ↓
生成 QueryPlan
  ├─ Step 1: 查作者
  ├─ Step 2: 查作者所在公司（依赖 Step 1）
  └─ Step 3: 查公司业务（依赖 Step 2）
  ↓
按依赖顺序执行
  ↓
汇总证据并回答
```

它和 IRCoT 的核心差异：

| 维度 | IRCoT | Plan-and-Execute |
| --- | --- | --- |
| 路线 | 执行时逐步产生 | 执行前生成 |
| 下一步 | 由最新证据决定 | 主要由计划决定 |
| 灵活性 | 高 | 中等 |
| 可预测性 | 较低 | 较高 |
| 并行能力 | 弱，通常严格串行 | 无依赖步骤可并行 |
| 调用成本 | 通常较高 | 更容易控制 |
| 失败模式 | 容易逐步漂移 | 初始计划错误会连锁传播 |

---

## 5. Adaptive Plan-and-Execute

生产系统经常不在 IRCoT 和固定 Plan 之间二选一，而是使用自适应计划：

```text
生成粗计划
  ↓
执行当前步骤
  ↓
结构化评估证据
  ├─ 充分       → 执行下一计划步骤
  ├─ 不足       → 改写当前查询并重试
  ├─ 仍然不足   → 替换剩余计划
  └─ 已可回答   → 提前结束
```

它保留了两类方案的优点：

- Planner 提供全局方向和依赖结构。
- Adaptive Executor 根据现场证据调整执行。
- Evaluator 提供可审计的控制信号。
- Replanner 只替换失败步骤和剩余计划，不推翻已确认事实。

当前项目的多跳实现属于这一类：

```text
Planner
  → Adaptive Executor
  → Hop Evaluator
  → Replanner
  → Final CRAG Evaluator
```

---

## 6. 隐式评估和显式评估

原版 IRCoT 通常没有独立的质量评估节点。模型生成的下一条推理同时隐含了三件事：

```text
当前证据说明了什么
还缺少什么信息
下一步应该检索什么
```

即：

```text
Evidence → Thought / Query → Retrieval
```

这种方式简洁，但如果中间推理错误，下一次检索会沿着错误方向继续。

工程化实现可以把隐式推理控制拆成结构化评估：

```text
Evidence
  ↓
HopAssessment
  ├─ relevance
  ├─ sufficiency
  ├─ extracted_facts
  ├─ resolved_entities
  ├─ suggested_query
  └─ can_answer_question
  ↓
代码控制器
  ├─ accept
  ├─ retry
  ├─ replan
  └─ finish
```

这相当于把 IRCoT 中隐含在下一条推理里的能力，升级为独立、可测试、可限流的 Workflow 节点。

需要注意：如果检索、生成和评估使用同一个 LLM，这仍然属于“显式自评”，不等于完全独立验证。高可靠系统可以使用：

- 独立的小型 Grader 模型。
- 不同模型交叉评估。
- Reranker 分数和实体覆盖规则。
- 最终答案 groundedness 校验。

CRAG 的核心也是显式评估检索质量，并根据结果触发纠正动作：

- [CRAG 论文](https://arxiv.org/abs/2401.15884)

---

## 7. 工程里是否真的有 LLM 评估节点

有，而且在 Agentic RAG 中很常见。典型流程：

```text
检索
  ↓
证据评估
  ├─ 相关且充分 → 回答
  ├─ 相关但不足 → 改写/补检
  └─ 无关       → fallback
```

常见评估维度：

1. **Relevance**：文档是否与问题相关。
2. **Sufficiency**：证据是否足够回答。
3. **Conflict**：不同来源是否矛盾。
4. **Groundedness**：最终答案是否能被证据支持。

不过生产系统通常不会让昂贵 LLM 评估所有结果，而是分层处理：

```text
硬规则
  ↓
相似度 / Reranker / 实体覆盖阈值
  ├─ 明显好 → 直接通过
  ├─ 明显差 → 直接拒绝或重试
  └─ 模糊区间 → LLM Grader
```

官方 Workflow 示例也常把文档评分设计为检索和生成之间的强制节点：

- [LangGraph Agentic RAG 示例](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)

---

## 8. Workflow 和 Agent 如何分工

一个实用的工程原则：

> 能用确定性 Workflow 解决，就不要把控制权交给 Agent；只在无法预先枚举的局部问题中使用 Agent。

Workflow 负责：

- 允许哪些执行路径。
- 什么时候必须评估。
- 重试、超时和调用预算。
- 什么时候停止。
- 失败如何降级。
- 权限和安全边界。

Agent / LLM 节点负责：

- 查询如何改写。
- 问题如何拆解。
- 当前证据意味着什么。
- 剩余计划如何调整。
- 开放问题下一步应该探索什么。

Tool 负责：

- 确定性执行检索、计算、数据库查询等动作。

最终形成：

```text
Workflow 掌握控制权
Agent 提供判断力
Tool 负责确定性执行
```

---

## 9. Strategy Router 放在哪里

需要区分两种 Router：

- **Knowledge Router**：决定检索哪个知识库。
- **Strategy Router**：决定使用什么执行策略。

Strategy Router 的典型输出：

```text
DIRECT
SINGLE_RAG
ITERATIVE_RAG
PLANNED_RAG
```

它可以放在 Workflow 外层：

```text
用户问题
  ↓
Strategy Router
  ├─ DIRECT
  ├─ SINGLE_RAG
  ├─ ITERATIVE_RAG Agent
  └─ PLANNED_RAG Workflow
```

也可以隐含在 Agent 的工具选择里：

```text
direct_answer          → DIRECT
search_knowledge_base  → SINGLE_RAG
连续单步 search        → ITERATIVE_RAG / IRCoT-like
multi_hop_search       → PLANNED_RAG
```

更稳定的混合方式是：

```text
外层 Router 给出路径和预算
  ↓
只向 Agent 暴露该路径允许的工具
  ↓
Agent 在边界内自主执行
  ↓
Workflow 强制评估和停止条件
```

Router 本身只是决策组件；当它决定后续节点时，它就是 Workflow 的路由节点。

---

## 10. IRCoT 的缓存与成本

IRCoT 每一步都会增加上下文：

```text
Step 1: 问题 + 初始文档
Step 2: 问题 + 初始文档 + 推理 1 + 新文档 1
Step 3: 问题 + 初始文档 + 推理 1 + 新文档 1 + 推理 2 + 新文档 2
```

因此完整响应缓存和检索结果缓存通常较难命中，但不是完全无法缓存：

| 缓存 | 可用性 |
| --- | --- |
| 文档 Embedding / 向量索引 | 完全可用 |
| 相同查询的检索结果 | 可用，但动态查询导致命中率较低 |
| Prompt Prefix Cache | 上下文只追加且前缀完全一致时可用 |
| 本地模型 KV Cache | 同一 Session、追加式消息下可用 |
| 完整 LLM 响应缓存 | 输入每轮变化，通常难命中 |
| 最终问答缓存 | 相同问题可以命中 |

为了提高前缀/KV Cache 利用率，应尽量：

- 保持 System Prompt 和消息顺序稳定。
- 对旧证据只追加，不重新排序。
- 不在每轮重写整个历史。
- 在真正需要时才做上下文压缩。

即使命中 KV Cache，新 Token 仍需要关注不断增长的历史，而且 IRCoT 的检索和推理通常严格串行，端到端延迟仍会随跳数增加。

Plan-and-Execute 在工程上更容易并行、裁剪上下文和控制预算；IRCoT 更适合路径确实无法预知的问题。

---

## 11. 当前项目的准确定位

当前架构包含两层控制：

```text
外层 Agent Loop
  ├─ direct_answer
  ├─ search_knowledge_base
  └─ multi_hop_search

multi_hop_search 内部 Workflow
  Planner
    → Adaptive Executor
    → HopAssessor
    → Retry / Replanner / Early Finish
    → Final CRAG
```

行为映射：

- 单步检索和 CRAG 评估强绑定。
- Agent 连续调用单步检索时，可以形成 IRCoT-like 探索。
- `multi_hop_search` 是一个 Plan 多跳宏工具/技能。
- 选择多跳后，生成计划、逐跳评估和最终 CRAG 都是强制流程。
- 当前没有实现论文原版的纯 IRCoT 执行器。

因此最准确的名称是：

> Agent 内隐式策略路由 + 原子检索能力 + Adaptive Plan 多跳技能 + 强制评估 Workflow。

---

## 12. 选择建议

```text
常识或闲聊
  → DIRECT

单一事实查询
  → SINGLE_RAG

路径未知、每一步依赖新发现实体
  → ITERATIVE_RAG / IRCoT

依赖关系可以粗略规划，但执行中可能变化
  → Adaptive Plan-and-Execute

固定、高频、强合规流程
  → Deterministic Workflow
```

工程默认选择应是：

1. 优先固定 Workflow。
2. 节点内容无法确定时使用 LLM Node。
3. 下一步路径也无法枚举时才使用 Agent / IRCoT。
4. 所有路径都由代码限制预算、评估节点和停止条件。
