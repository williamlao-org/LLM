# Phase 3: Agentic RAG 实现计划

## 背景

在 Phase 1（经典 RAG）和 Phase 2（进阶 RAG：混合检索、Reranking、Query Rewriting、RAGAS 评估）的基础上，Phase 3 的核心目标是：**把 RAG 检索从"无脑管线"变成 Agent 可自主控制的工具**。

传统 RAG 是一条固定的流水线——不管用户问什么都去检索。Agentic RAG 让 LLM 自己决定：
- 是否需要检索？（常识问题可以直接答）
- 检索结果质量够不够？（不够就改写重来）
- 需不需要多轮迭代检索？（多跳推理）
- 应该查哪个知识库？（路由选择）

## Proposed Changes

### 文件结构

在 [RAG/](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/RAG) 目录下新建 4 个文件：

```
RAG/
├── phase3_agentic_rag.py    # [NEW] 核心 Agent 循环 + 工具定义
├── phase3_self_rag.py       # [NEW] Self-RAG / CRAG 检索质量评估
├── phase3_router.py         # [NEW] 多知识库路由
├── phase3_main.py           # [NEW] 交互式入口
```

---

### 1. 核心 Agent 循环

#### [NEW] [phase3_agentic_rag.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/RAG/phase3_agentic_rag.py)

**核心思路**：用 OpenAI function calling 实现一个精简的 ReAct 循环，把检索封装为 Agent 可调用的工具。

```
用户问题
    │
    ▼
Agent 思考：需不需要检索？
    │
    ├── 不需要 → 直接回答（自适应检索）
    │
    └── 需要 → 调用 search_knowledge_base 工具
                │
                ▼
           Agent 评估检索质量（Self-RAG）
                │
                ├── 质量好 → 基于检索结果生成回答
                │
                └── 质量差 → 改写查询，再次检索（多跳）
```

**类设计**：

- `AgenticRAG` 类：
  - 复用已有的 `RAGChain` 的底层组件（embedder、store、retriever、reranker）
  - 不直接复用 `RAGChain.query()`，因为 Agent 需要拆解控制每个步骤
  - 定义 3 个 function calling 工具供 LLM 调用：
    1. **`search_knowledge_base`** — 从知识库中检索相关信息
    2. **`assess_retrieval_quality`** — 评估检索结果的质量（Self-RAG 的核心）
    3. **`direct_answer`** — 不需要检索，直接回答
  - Agent 循环：最多 `max_iterations` 轮（默认 5），每轮调用 LLM → 解析 function call → 执行工具 → 把结果反馈给 LLM
  - 自适应检索：通过 system prompt 引导 LLM 判断是否需要检索
  - 多跳检索：Agent 可以多次调用 `search_knowledge_base`，每次用不同的查询

**与 Phase 2 的关系**：
- 底层检索仍然用 Phase 2 的混合检索 + Reranking
- 但不再是"一条管线跑到底"，而是 Agent 按需调用

---

### 2. Self-RAG 质量评估

#### [NEW] [phase3_self_rag.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/RAG/phase3_self_rag.py)

**核心思路**：实现 Self-RAG 和 CRAG 的简化版——Agent 检索后，先让 LLM 评估检索结果的质量，再决定下一步动作。

包含两个评估维度：

1. **Relevance（相关性）**：检索到的内容和用户问题是否相关？
   - `relevant` — 相关，可以用于回答
   - `partially_relevant` — 部分相关，可以勉强用
   - `irrelevant` — 完全无关，需要改写查询重新检索

2. **Sufficiency（充分性）**：检索到的内容是否足够回答问题？
   - `sufficient` — 信息充分，可以直接回答
   - `insufficient` — 信息不足，需要补充检索
   - `conflicting` — 信息矛盾，需要更多来源验证

**实现**：用一次 LLM 调用，让模型以 JSON 输出评估结果 + 建议的下一步动作（直接回答 / 改写查询重搜 / 用当前结果勉强回答）。

---

### 3. 多知识库路由

#### [NEW] [phase3_router.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/RAG/phase3_router.py)

**核心思路**：不同类型的问题应该路由到不同的知识库。

- `KnowledgeRouter` 类：
  - 管理多个命名知识库（每个知识库有名称、描述、和一个已构建索引的检索器）
  - 路由方式：**LLM 分类** — 用一次 LLM 调用，根据知识库的描述判断应该查哪个库
  - 支持路由到多个库并合并结果

**学习目的**：
- 在 `phase3_main.py` 中，用 `docs/` 下的不同文件模拟不同的知识库
  - 例如：`技术文档库`（transformer.md, rag_overview.md, agent.md）和 `通用知识库`（纳瓦尔宝典.pdf, 数字化建设方案.docx）
  - 用户提问时，Agent 先路由到合适的库再检索

---

### 4. 交互式入口

#### [NEW] [phase3_main.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/RAG/phase3_main.py)

仿照 [main.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/RAG/main.py) 的交互式风格，但展示 Agentic RAG 的全流程：

- 支持命令：`/quit`、`/help`、`/rebuild`、`/compare`（对比 always-retrieve vs agentic）
- 默认模式下展示 Agent 的完整思考过程：是否检索 → 路由选择 → 质量评估 → 最终回答
- `/compare` 模式：同一个问题分别用传统 RAG 和 Agentic RAG 处理，对比效果

---

## Open Questions

> [!IMPORTANT]
> **知识库分组**：目前 `docs/` 下有 5 个文件，我计划按内容类型分成两个逻辑知识库来演示路由功能。但物理上仍共用同一个 `docs/` 目录——通过文件分组而非子目录来实现。是否需要创建子目录（如 `docs/tech/`、`docs/general/`）来物理分离？

## Verification Plan

### Manual Verification

1. 运行 `phase3_main.py`，测试以下场景：
   - **自适应检索**：问"1+1等于几？"（常识）→ Agent 应该直接回答，不检索
   - **正常检索**：问"Transformer 的注意力机制"→ Agent 检索并回答
   - **Self-RAG**：问一个检索质量差的问题 → Agent 应该改写查询重搜
   - **多跳检索**：问"ReAct Agent 如何结合 RAG 技术"→ 可能需要分别检索 Agent 和 RAG 的知识
   - **路由选择**：问技术问题和通用问题，观察路由到不同知识库
   - **`/compare`**：对比传统 RAG 和 Agentic RAG 的效果差异
