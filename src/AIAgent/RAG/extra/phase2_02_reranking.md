# Phase 2 · 第二块：重排序 Reranking（Cross-encoder 精排）

> 本文是学习笔记 + 「新对话快速启动器」。
> 新开会话学 Phase 2 时，把这份文件喂给我即可立刻接上下文，无需重新铺垫。

---

## 0. 我现在在哪（进度锚点）

```
Phase 1 ✅ 经典 RAG（document_loader / chunker / embedder / vector_store / rag_chain / main）
Phase 2 🚧 进阶 RAG
   ├─ 第一块 ✅ 混合检索 Hybrid Search（BM25 + RRF）
   ├─ 第二块 ✅ Reranking（Cross-encoder 二次精排）   ← 本文，已落地、已跑通
   ├─ 第三块 ⬜ Query Rewriting / HyDE / Multi-Query
   └─ 第四块 ⬜ 评估（RAGAS：Precision@K / Faithfulness ...）
Phase 3+ ⬜ Agentic RAG / 记忆系统 / GraphRAG（见 learning_roadmap.md）
```

---

## 1. 为什么需要 Reranking

Hybrid Search（Phase 2 第一块）解决了"召回"——Dense + BM25 + RRF 把可能相关的 chunk 捞出来。但这些候选仍然是**粗排**的：

- Dense（双塔）：query 和 doc **各自编码**成向量，比余弦相似度
- BM25：纯词频统计
- RRF：只看名次，丢弃原始分数

它们的共同问题：**query 和 doc 从未真正"见面"**——各自编码后只比一个数字。

Cross-encoder 解决这个问题：把 query 和 doc **拼在一起**送进 Transformer，让模型看到两者的词级交互，输出一个精确的相关性分数。

### 具体例子

```
query = "Python 怎么处理 JSON"

chunk A = "Python 的 json 模块提供 loads() 和 dumps() 方法"
chunk B = "Python 是一门流行的编程语言，支持多种数据格式"
```

- **双塔**：A 和 B 都和 "Python" 语义相关，分数可能接近
- **Cross-encoder**：看到 "处理 JSON" ↔ "json 模块 loads() dumps()" 的精确对应，给 A 打出远高于 B 的分数

### 双塔 vs Cross-encoder

```
双塔（Bi-encoder）               Cross-encoder

query → [模型A] → vec_q          query ─┐
                      ↘                  ├→ [模型] → 相关性分数
doc   → [模型B] → vec_d  cos(q,d)       │
                                 doc   ──┘

各自编码，快                      联合编码，准
适合初检百万候选                   适合精排几十条
```

**为什么不直接用 Cross-encoder 做初检？**
因为它要对每一对 (query, doc) 都过一遍模型。知识库 10 万个 chunk → 每次查询跑 10 万次推理 → 太慢。所以只拿来重排少量候选（10~30 条）。

---

## 2. 流水线位置

Reranking 插在 Hybrid 召回和 Prompt 构造之间，整体流水线变成：

```
用户问题
    │
    ▼
Hybrid 召回 Top-N（粗排，N = top_k × 3，快）
    │
    ▼
Cross-encoder Rerank（精排，对 N 条打分，取 top_k 条）   ← 新增
    │
    ▼
构造 Prompt（注入 top_k 条上下文）
    │
    ▼
LLM 生成回答
```

关键设计：**candidate_k**（送进 Reranker 的候选数）默认为 `top_k * 3`。如果只给 Reranker `top_k` 条，它能做的只是微调顺序，价值有限。多给一些候选，让那些"被粗排低估"的好 chunk 有机会被精排捞回来。

---

## 3. SiliconFlow Rerank API

SiliconFlow 有专用的 `/rerank` endpoint，和 Cohere 风格类似：

```bash
curl -X POST https://api.siliconflow.cn/v1/rerank \
  -H "Authorization: Bearer $SILICONFLOW_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Apple",
    "documents": ["apple", "banana", "fruit", "vegetable"],
    "return_documents": true,
    "top_n": 4
  }'
```

返回格式：
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95, "document": {"text": "apple"}},
    {"index": 2, "relevance_score": 0.82, "document": {"text": "fruit"}},
    ...
  ]
}
```

- `index`：原始 documents 数组中的下标
- `relevance_score`：Cross-encoder 打出的相关性分数
- 结果已按 `relevance_score` 降序排列
- `return_documents: false` 时不返回原文（我们已经有了，省流量）

模型：`BAAI/bge-reranker-v2-m3`，和我们 embedding 用的 `bge-m3` 同源。

---

## 4. 代码落地（文件 / 接口 / 跑法）

### 新增文件

| 文件 | 内容 | 关键接口 |
|:---|:---|:---|
| `phase2_02_reranker.py` | APIReranker，调用 SiliconFlow /rerank | `APIReranker(api_key, model).rerank(query, results, top_n)` |

### 修改文件

| 文件 | 改动 |
|:---|:---|
| `config.py` | 新增 `reranker_base_url` / `reranker_api_key` / `reranker_model`（复用 SiliconFlow 凭据） |
| `rag_chain.py` | `__init__` 新增 `use_reranker=False` 开关；`query()` 在检索和 Prompt 之间插入 rerank 步骤 |

### 接口设计

`APIReranker.rerank()` 接受 `list[SearchResult]` → 返回 `list[SearchResult]`：
- 输入：Hybrid 的粗排结果
- 输出：重排后的结果，`score` 更新为 reranker 分数，`ranks` 保留原始 Dense/Sparse 名次
- 不修改 `SearchResult` 数据结构

### 跑

```bash
uv run python phase2_02_reranker.py      # 单测 Reranker（mock 数据 + 真实 API）
uv run python rag_chain.py     # 全链路（Hybrid + Reranker）
```

---

## 5. 实测观察

### 5.1 单测（mock 数据）

查询："Embedding 模型是怎么把文本变成向量的"

| 粗排名次 | Reranker 精排 | 分数 | 内容 |
|:---|:---|:---|:---|
| 第 4 → | **第 1 ↑** | 0.9752 | "Embedding 模型把文本转成向量后，语义相近的句子在向量空间里距离也相近" |
| 第 2 → | 第 2 | 0.1966 | "BGE-M3 是一个多语言的 embedding 模型…" |
| 第 5 → | 第 3 ↑ | 0.0009 | "BM25 是一种经典的稀疏检索算法…" |

**关键观察**：
1. **最相关的 chunk 从第 4 被拉到第 1**——粗排时被 RRF 名次压住了，精排纠正了
2. **分数差距非常明显**：0.97 vs 0.19 vs 0.0009，Cross-encoder 的区分度远超粗排的 0.033~0.028
3. **天气那条（完全无关）被踢出 Top-3**——粗排时 RRF 第 3，精排时相关分太低直接淘汰

### 5.2 全链路（真实知识库）

查询："ReAct 是什么架构模式？"

Hybrid 粗排候选 9 条（`top_k * 3`），Reranker 精排取 Top-3：

| 粗排名次 | Reranker 精排 | 分数 | 内容 |
|:---|:---|:---|:---|
| 第 2 → | **第 1 ↑** | 0.9441 | `agent.md`："AI Agent 技术详解 → 什么是 AI Agent" |
| 第 1 → | 第 2 ↓ | 0.8337 | `agent.md`："ReAct 工作流程：Thought → Action → Observation" |
| 第 5 → | **第 3 ↑** | 0.0383 | `数字化建设方案.docx`：总体架构描述 |

**关键观察**：
1. **粗排第 1 和第 2 交换了位置**：粗排时 "ReAct 工作流程" 在第 1，但 Cross-encoder 认为 "什么是 AI Agent"（包含 ReAct 的整体介绍）更全面地回答了"什么架构模式"这个问题，排到了第 1
2. **分数拉开明显梯度**：0.94 和 0.83 都是高相关，但和第 3 名的 0.038 拉开了巨大差距——说明前两条确实最相关
3. **粗排第 5 跳到精排第 3**：数字化建设方案在粗排时被排到第 5（被无关 chunk 挤掉），精排时被捞回来

---

## 6. 核心收获

1. **双塔快但粗，Cross-encoder 准但慢** → 典型做法是两者级联：双塔初检 + Cross-encoder 精排
2. **candidate_k 要大于 top_k** → 否则精排无素材可调整
3. **Reranker 不做初检** → O(N) 复杂度，百万级候选跑不起
4. **分数区分度** → 粗排分数挤在很小的范围（0.028~0.033），精排分数跨越 0.0009~0.97，LLM 拿到的上下文质量显著提升
5. **API 调用简单** → SiliconFlow 的 /rerank 接口和 Cohere 风格一致，几行 httpx 就搞定

---

## 7. 待消化 / 可自己玩

- 把 `candidate_k` 从 `top_k * 3` 改成 `top_k * 5` 或更大，看精排结果有没有变化
- 查看 Reranker 模型的 token 限制——如果 chunk 太长，API 可能会截断
- 对比「只有 Hybrid 没有 Reranker」vs「Hybrid + Reranker」在你自己的知识库上的效果差异
- Cross-encoder 本地跑：用 `sentence-transformers` 的 `CrossEncoder("BAAI/bge-reranker-v2-m3")` 本地推理，和 API 结果对比

---

## 8. 下一步：Query Rewriting（Phase 2 第三块）

- **动机**：用户的问题经常"不适合检索"——太口语化、太模糊、或者一个问题包含多个子问题
- **Query Rewriting**：让 LLM 把用户问题改写成更适合检索的形式
- **HyDE**：先让 LLM 生成一个"假设答案"，用假设答案的 embedding 去检索（而不是用问题本身）
- **Multi-Query**：把一个复杂问题拆成多个子查询，分别检索后合并结果

> 新对话开场建议说：「我在学 Phase 2，前两块 Hybrid 和 Reranking 已落地（见 phase2_01_hybrid_search.md 和 phase2_02_reranking.md），现在要做第三块 Query Rewriting」。
