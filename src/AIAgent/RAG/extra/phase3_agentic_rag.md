
## 0. 我现在在哪（进度锚点）

```
Phase 1 ✅ 经典 RAG（document_loader / chunker / embedder / vector_store / rag_chain / main）
Phase 2 ✅ 进阶 RAG（Hybrid Search / Reranking / Query Rewriting / RAGAS 评估）
Phase 3 ✅ Agentic RAG   ← 本文，已落地、已跑通
   ├─ phase3_agentic_rag.py  核心 Agent 循环 + function calling 工具
   ├─ phase3_self_rag.py     Self-RAG / CRAG 检索质量评估
   ├─ phase3_router.py       多知识库 LLM 路由
   ├─ phase3_query_decomposer.py  多跳查询计划拆解
   ├─ phase3_hop_assessor.py 单跳事实提取与质量评估
   └─ phase3_main.py         交互式入口（/compare /steps）
Phase 4+ ⬜ 记忆系统 / GraphRAG（见 learning_roadmap.md）
```

延伸笔记：[IRCoT、Plan-and-Execute 与工程化混合 RAG](phase3_ircot_and_hybrid_rag.md)

---

## 1. 一句话与动机

**Agentic RAG = RAG + Agent。**

Phase 1-2 的 RAG 是一条**固定管线**：`问题 → 检索 → Rerank → 生成`，不管问什么都跑一遍，不管检索质量好坏都直接用。

它的死穴：

- 常识问题（“1+1=?”）也去检索 —— 浪费、还可能被无关内容带偏
- 一次检索不够时没有补救
- 多知识库时只能全量搜，噪声大、延迟高

Agentic RAG 的核心改动是：**把检索变成 Agent 手里的工具，由 LLM 决定是否检索以及使用普通还是多跳检索；检索发生后，代码强制执行质量评估和有限重试。**

<aside>
📌

Agentic RAG 没有玄学。它区别于传统 RAG 的**唯一定义性特征**就是：检索决策权从固定管线交给了会自主决策的 LLM。本质就是“检索封装成工具 + LLM 编排”，不多也不少。

</aside>

---

## 2. 从固定管线到 Agent 决策

```
传统 RAG（固定管线）
  问题 ──→ 检索 ──→ Rerank ──→ 生成
  （无脑，一条路跑到底）

Agentic RAG（Agent 决策）
  问题
    │
    ▼
  Agent 思考：需不需要检索？
    ├── 不需要 → direct_answer（自适应检索）
    └── 需要
         ├── 普通问题 → search_knowledge_base → CRAG 评估/改写
         └── 复杂问题 → multi_hop_search
                            ↓
                       Planner 生成粗计划
                            ↓
                       每跳检索 + 事实/实体评估
                            ├── 充分 → 下一跳
                            ├── 不足 → 当前跳改写重试
                            ├── 仍不足 → 重规划剩余步骤
                            └── 已可回答 → 提前结束
```

---

## 3. 文件结构（均已实现）

```
RAG/
├── phase3_agentic_rag.py    # 核心 Agent 循环 + 3 个 function calling 工具
├── phase3_self_rag.py       # Self-RAG / CRAG 检索质量评估器
├── phase3_router.py         # 多知识库 LLM 路由器
├── phase3_query_decomposer.py # 多跳查询计划拆解器
├── phase3_hop_assessor.py   # 单跳证据评估 + 事实/实体提取
└── phase3_main.py           # 交互式入口（/compare 对比传统 RAG，/steps 看决策步骤）
```

底层检索**复用 Phase 2** 的 Dense + BM25 混合检索与 Reranker，Phase 3 只在上层加了“决策大脑”。

---

## 4. 核心 Agent 循环（phase3_agentic_rag.py）

用 **OpenAI function calling + 代码强制 CRAG 闭环**：LLM 从三个入口工具中选择，质量评估不再作为可选工具暴露。

| 工具                      | 作用                                              |
| ------------------------- | ------------------------------------------------- |
| `search_knowledge_base` | 检索知识库（内部走路由 + 混合检索 + 可选 Rerank） |
| `multi_hop_search`      | 拆解带依赖的子查询，串行检索并聚合证据            |
| `direct_answer`         | 判断不需检索时直接回答（常识题、闲聊）            |

`assess_retrieval_quality` 是内部自动步骤：普通检索和多跳检索完成后必然执行，不能被 LLM 跳过。

主循环（`AgenticRAG.query()`）：

1. 把用户问题 + 工具定义发给 LLM
2. LLM 决定调用哪个工具（或直接给文本回答）
3. 若发生检索，代码强制执行 `检索 → 评估 → refine 时改写重试`
4. 把最终证据和评估作为一个完整 tool message 喂回 LLM
5. 循环直到得到最终回答；`max_iterations` 和 `max_tool_calls` 双重兜底

约束由代码执行：默认最多 3 次 Agent 工具调用、2 次普通/定向补检、每跳 1 次重试、1 次重规划、最多执行 6 个多跳步骤；达到上限后携带现有证据进入最终回答。

多知识库构建与持久化：`build_default_indexes()` 把 `docs/` 分成 `tech_docs`（技术库）+ `general_kb`（通用库），索引落盘为 `phase3_index_*.json`。

---

## 5. Self-RAG / CRAG 质量评估（phase3_self_rag.py）

Agent 检索完不盲目塞进 Prompt，而是先“照镜子”——`SelfRAGAssessor` 让 LLM 从**两个维度**打分，再综合出下一步动作：

- **Relevance（相关性）**：`relevant` / `partially_relevant` / `irrelevant`
- **Sufficiency（充分性）**：`sufficient` / `insufficient` / `conflicting`
- **决策 action**：
  - `answer` —— 质量够，直接回答
  - `refine` —— 部分相关/信息不足，附 `suggested_query` 改写重搜
  - `fallback` —— 完全无关，放弃检索

<aside>
⚠️

**CRAG 还是 Self-RAG？这里要分清（很多人搞混）：**
• **Self-RAG（Asai et al., 2023）**：在生成过程中插入“反思 token”，模型边生成边评估，**需要微调模型**。
• **CRAG（Yan et al., 2024）**：**不微调**，用独立评估步骤判断检索质量，差就触发纠正（改写/补检/放弃）。
本实现走的是 **CRAG 的思路**（独立评估、不微调），只是概念上统称 Self-RAG。

</aside>

### 5.1 Adaptive Plan-and-Execute 多跳闭环

多跳不是一次性照着初始计划机械执行，而是采用 **Planner + Adaptive Executor + CRAG Evaluator**：

1. `QueryDecomposer.decompose()` 生成带 `depends_on` 的粗粒度计划。
2. 每一步检索后，`HopAssessor` 输出结构化 `HopAssessment`：相关性、充分性、是否已经能回答原问题、简短事实、实体映射和建议查询。
3. 执行器只把声明依赖步骤的事实/实体交给下一跳，不传递模型思维链。
4. 单跳不足先局部改写；重试仍失败时，`replan()` 只替换失败步骤和剩余计划，保留已确认事实。
5. 全链证据最终再经过 CRAG；若仍需补充，只执行建议查询的定向普通检索，不重新运行整套多跳计划。

`/steps` 会记录 `assess_multi_hop_step`、`retry_multi_hop_step`、`replan_multi_hop` 和 `finish_multi_hop`，便于审计控制流。

---

## 6. 多知识库路由（phase3_router.py）

`KnowledgeRouter` 用 **LLM 做路由**：把每个库的名称 + 描述 + 文件清单喂给 LLM，让它选该查哪个库，支持多选，也支持“纯闲聊选空列表不检索”。`route_and_search()` 一步到位：路由 → 各库检索 → 按 score 合并取 Top-K。

<aside>
💡

**为什么不用 embedding 相似度做路由？** 库的“主题”是高层语义概念，而 embedding 擅长的是段落级相似度。让 LLM 判断“这个问题该查哪个库”比 embedding 准得多，代价只是一次轻量 LLM 调用。新增知识库也只要写个描述，不用重训分类器。

</aside>

---

## 7. 关键设计决策

- **不复用 `RAGChain.query()`**：Agent 需要拆开控制每一步（检索/评估/回答分离），而 `RAGChain` 是一条封闭管线，所以只复用它的底层组件（embedder / store / retriever / reranker）。
- **查询内状态隔离**：检索方法同时返回文本与 `SearchResult`，评估显式消费本次结果，不使用实例级 `_last_results`。
- **自适应多跳**：初始计划提供方向，每跳结构化提取事实和实体；证据不足时先局部重试，再有限重规划剩余步骤。
- **双层评估**：`HopAssessor` 控制单跳执行，最终 CRAG 判断整条证据链是否足以回答；最终 refine 只触发定向补检。
- **评估/路由都用低温度（0.1）**：判断类任务要稳定、可复现，不要发散。
- **解析失败保守兜底**：评估 JSON 解析失败 → 默认 `answer`（别卡住流程）；路由解析失败 → 回退搜全部库。

---

## 8. 两个流派：Agent（工具驱动）vs Workflow（图驱动）

Agentic RAG 工业界常见两派，本实现采用两者之间的**混合控制**：

|          | ① Agent / 工具驱动              | ② Graph / 状态机驱动                  |
| -------- | -------------------------------- | -------------------------------------- |
| 控制流   | LLM 即兴决定调哪个工具、什么顺序 | 写死在图里的固定边                     |
| 评估节点 | 可由 LLM 自主选择                | 检索后**必然**走评估节点，跳不过 |
| 特点     | 灵活，但行为不完全可预测         | 可预测、可控、好调试                   |

<aside>
✅

**本实现的边界**：首次路径仍由 LLM 自主选择，因此保留 Agentic 特征；一旦选择普通或多跳检索，后续评估、改写和重试由代码强制执行，因此 CRAG 闭环不会被跳过。

</aside>

---

## 9. 易混点：Self-RAG 在线评估 ≠ RAGAS 离线评估

两者**同源**（都是 LLM-as-Judge，维度都含 Faithfulness / Relevance），但岗位完全不同：

|              | RAGAS（Phase 2）                   | Self-RAG 闭环（Phase 3）                                      |
| ------------ | ---------------------------------- | ------------------------------------------------------------- |
| 时机         | 离线、事后，跑在评估数据集上       | 在线、实时，跑在每次真实查询里                                |
| 目的         | 衡量“优化到底有没有效、好了多少” | 改进“**当前这一次**回答”                              |
| 输出         | 一张分数报告（给开发者看）         | 一个决策（answer/refine/fallback），**改变 Agent 行为** |
| 要标准答案吗 | 部分指标要`ground_truth`         | 不需要，运行时哪有标准答案                                    |

> 一句话：**RAGAS 是“考试打分”（质检员写报告），Self-RAG 闭环是“边做边自检”（操作员现场纠错）。** 同一套评判技术，两个不同用途。

---

## 10. 已知局限 & 下一步

1. **多跳仍依赖评估模型质量**：执行器能局部重试和重规划，但错误的事实抽取仍可能沿依赖传播。
2. **只评估了检索、没评估生成**：完整闭环还可增加回答后的忠实度（groundedness）校验，不达标时重答。
3. **尚未封装成 MCP Tool**：当前使用本地 OpenAI function calling，后续可在不改变检索闭环的前提下增加 MCP 传输层。

---

## 验证方式（Manual）

运行 `phase3_main.py`，测试：

- **自适应检索**：问“1+1 等于几？” → Agent 应直接回答，不检索
- **正常检索**：问“Transformer 的注意力机制” → 路由到技术库并回答
- **Self-RAG**：问一个检索质量差的问题 → Agent 应改写查询重搜
- **自适应多跳**：观察单跳评估、局部重试、剩余计划重规划或提前结束
- **路由**：技术问题 vs 通用问题，观察路由到不同库
- **`/compare`**：同一问题对比传统 RAG vs Agentic RAG
- **`/steps`**：查看多跳子步骤、单跳评估、重试、重规划、提前结束和最终 CRAG
