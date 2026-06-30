# Phase 2 · 第三块：查询优化 Query Rewriting / HyDE / Multi-Query

> 本文是学习笔记 + 「新对话快速启动器」。
> 新开会话学 Phase 2 时，把这份文件喂给我即可立刻接上下文，无需重新铺垫。

---

## 0. 我现在在哪（进度锚点）

```
Phase 1 ✅ 经典 RAG（document_loader / chunker / embedder / vector_store / rag_chain / main）
Phase 2 🚧 进阶 RAG
   ├─ 第一块 ✅ 混合检索 Hybrid Search（BM25 + RRF）
   ├─ 第二块 ✅ Reranking（Cross-encoder 二次精排）
   ├─ 第三块 ✅ Query Rewriting / HyDE / Multi-Query   ← 本文，已落地、已跑通
   └─ 第四块 ⬜ 评估（RAGAS：Precision@K / Faithfulness ...）
Phase 3+ ⬜ Agentic RAG / 记忆系统 / GraphRAG（见 learning_roadmap.md）
```

---

## 1. 为什么需要查询优化

Hybrid Search（第一块）解决了「召回覆盖」，Reranking（第二块）解决了「精排排序」。
但这两块都默认用户的原始问题就是好的检索查询——而事实上：

| 场景 | 用户原始问题 | 问题出在哪 |
|:---|:---|:---|
| 口语/模糊 | 「这个 transformer 注意力的那个公式咋来的」 | 和文档表达风格不匹配 |
| 问句 vs 陈述 | 「RAG 是什么？」 | 向量空间里问句和答案句不对齐 |
| 多维度 | 「A 和 B 有什么区别，各适合什么场景？」 | 单条查询只能命中一个维度 |

这就是**查询优化层**要解决的问题：在问题进入检索之前，先对它做处理。

---

## 2. 三种策略原理

### 2.1 Query Rewriting（直接改写）

**思路**：让 LLM 把用户的口语、模糊、含歧义的问题，改写成更适合向量检索的规范化表述。

```
用户问：「这个 transformer 注意力的那个公式咋来的」
         ↓ LLM 改写
检索用：「Transformer 自注意力机制 Scaled Dot-Product Attention 公式推导」
```

改写的核心操作：
- 去掉语气词、口语表达
- 补全缩写和代词（「它」→ 具体名词）
- 使用文档化的专业表述风格

**本质**：弥合「用户表达风格」和「文档表达风格」之间的沟。

---

### 2.2 HyDE（Hypothetical Document Embeddings）

**思路**：不用问题本身去检索，而是先让 LLM 生成一个「假设性答案文档」，再用这个假设文档的 embedding 去检索真实文档。

```
用户问：「Transformer 的自注意力有什么特点？」
         ↓ LLM 生成假设答案
假设文档：「Transformer 的自注意力机制具有以下特点：
           1. 并行计算：不同于 RNN 的顺序处理，自注意力可以并行...
           2. 全局感受野：每个 token 可以直接关注序列中任意位置...」
         ↓ 用假设文档的 embedding 去检索
真实文档：（和假设文档语义空间对齐的真实 chunks）
```

**为什么有效**？

```
问题的 embedding 空间：短疑问句 → 「自注意力有什么特点？」
文档的 embedding 空间：长陈述句 → 「自注意力的特点包括...并行计算...」

两者在向量空间里是不对齐的（一个是问号结尾的短句，一个是陈述段落）

假设答案的向量 ≈ 真实答案的向量（都是陈述风格，语义空间对齐）
→ 用假设答案检索，命中率更高
```

**潜在风险**：如果 LLM 生成的假设答案方向跑偏（知识幻觉），检索结果会更差。
适合知识域明确、LLM 有基础认知的问题。

---

### 2.3 Multi-Query（多查询分解）

**思路**：把一个复杂问题拆成 N 个独立子查询，每个子查询分别检索，最后用 RRF 把多路结果合并，取并集。

```
用户问：「ReAct 和普通 Agent 有什么区别，各适合什么场景？」
         ↓ LLM 分解成 3 个子查询
子查询1：「ReAct Agent 的原理和特点是什么」
子查询2：「普通 Agent（如基于规划的Agent）的原理和特点是什么」
子查询3：「ReAct Agent 和普通 Agent 的区别以及各自的最佳应用场景」
         ↓ 3路各自检索，RRF 合并
最终结果：三路的并集（去重），覆盖多个维度
```

**为什么 Multi-Query 用 RRF 合并而不是简单拼接**？
- 简单拼接会有大量重复 chunk
- RRF 去重 + 名次加权，在多路都靠前的 chunk 会获得更高分（强共识信号）

---

## 3. 三种策略对比

| 维度 | rewrite | hyde | multi_query |
|:---|:---|:---|:---|
| 解决的问题 | 表达风格不匹配 | 问句/答案向量空间不对齐 | 问题含多个子维度 |
| 改写结果 | 1条规范查询 | 1段假设答案文本 | N条子查询列表 |
| 检索次数 | 1次 | 1次 | N次（N默认3） |
| 合并方式 | 直接替换原始 query | 直接替换原始 query | RRF 合并 N 路结果 |
| API 调用 | 1次 LLM（改写）+ 1次检索 | 1次 LLM（生成答案）+ 1次检索 | 1次 LLM（分解）+ N次检索 |
| 适合场景 | 口语/缩写/歧义问题 | 知识型问答（答案空间清晰） | 多维度/对比/综合性问题 |
| 风险 | 低（改写偏差有限） | 中（LLM 幻觉会带偏检索） | 低（子查询独立，失败可降级） |

---

## 4. 关键设计决策

### 决策1：改写只影响检索，不影响 Prompt

原始问题始终保留，只有 `retrieval_query`（送进检索器的那个）被改写/替换。
LLM 生成答案时，Prompt 里用的仍然是用户原始问题：

```
原始问题：「这个 transformer 注意力的那个公式咋来的」
              │
              ├─ retrieval_query →「Transformer自注意力机制公式推导」→ 检索 → 找到相关 chunks
              │
              └─ question →「这个 transformer 注意力的那个公式咋来的」→ 注入 Prompt → LLM 理解用户意图
```

如果把改写后的查询送进 Prompt，LLM 可能会回答「改写版」问题而不是用户真正的意图。

### 决策2：Reranker 始终用原始问题打分

即使用了 HyDE（假设答案检索），Reranker 打分时也用原始问题，不用假设答案：
- 假设答案只是检索的「钥匙」，用来找门
- Reranker 评的是「这个 chunk 和用户真正想知道的内容有多相关」

### 决策3：Multi-Query 的 RRF 合并复用了 phase2_01_hybrid_retriever 里的 `reciprocal_rank_fusion`

这个函数在 Hybrid Search 里已经写好了，Multi-Query 的多路合并直接复用它。
不需要重写，接口完全一致（`list[list[SearchResult]]` → `list[SearchResult]`）。

---

## 5. 流水线终态

启用所有优化后的完整流水线：

```
用户原始问题
    │
    ▼  ← [phase2_03_query_rewriter.py] QueryRewriter
    │  rewrite:     改写成规范查询（1条）
    │  hyde:        生成假设答案（1段）
    │  multi_query: 分解子查询（N条）
    │
    ▼
Hybrid 召回（Dense + BM25 + RRF）
    │  multi_query: N路各自检索，再 RRF 合并
    │
    ▼
Cross-encoder Rerank 精排       [phase2_02_reranker.py]
    │  始终用原始问题打分
    │
    ▼
构造 Prompt（始终注入原始问题）  [rag_chain.py]
    │
    ▼
LLM 生成回答
```

---

## 6. 代码落地（文件 / 接口 / 跑法）

### 新增文件

| 文件 | 内容 | 关键接口 |
|:---|:---|:---|
| `phase2_03_query_rewriter.py` | 三种策略实现 | `QueryRewriter(llm_client, model).rewrite(q)` / `.hyde(q)` / `.multi_query(q, n)` |

### 修改文件

| 文件 | 改动 |
|:---|:---|
| `rag_chain.py` | `import reciprocal_rank_fusion` 从 phase2_01_hybrid_retriever；`__init__` 新增 `query_rewrite` 参数；`query()` Step 0 插入改写；`_build_response()` 抽出为独立方法（供 multi_query 路径复用） |

### 跑

```bash
uv run python phase2_03_query_rewriter.py              # 单测三种策略（纯 LLM，不需要知识库）

uv run python rag_chain.py                   # 全链路，默认 none（无改写）
uv run python rag_chain.py rewrite           # 全链路，策略1：直接改写
uv run python rag_chain.py hyde              # 全链路，策略2：HyDE
uv run python rag_chain.py multi_query       # 全链路，策略3：多子查询
```

---

## 7. 实测观察

### 7.1 单测（phase2_03_query_rewriter.py）

**策略1 rewrite**
```
原始：「这个 transformer 注意力的那个公式怎么来的」
改写：「Transformer注意力机制公式的推导方法」

原始：「RAG 是啥，怎么用？」
改写：「RAG的概念与用法」
```
口语表达被规范化，缩写（RAG）保留，语气词被清除。✅

**策略2 HyDE**
```
原始问题：「RAG 是啥，怎么用？」
假设答案：「RAG（检索增强生成）是一种结合信息检索与大语言模型的技术，
           旨在提升生成内容的准确性和时效性。其典型流程是：首先将用户查询
           转化为向量，从外部知识库中检索最相关的文档片段...」
```
生成了一段专业、陈述风格的假设答案，用它的 embedding 去检索文档会更准。✅

**策略3 Multi-Query**
```
原始问题：「ReAct 和普通 Agent 有啥区别，分别适合啥场景？」
分解为：
  [1] ReAct Agent 的原理和特点是什么
  [2] 普通 Agent（如基于规划的Agent）的原理和特点是什么
  [3] ReAct Agent 和普通 Agent 的区别以及各自的最佳应用场景
```
完整覆盖原始问题的三个角度（ReAct 自身、对比对象、场景对比）。✅

### 7.2 全链路（rag_chain.py rewrite 策略）

查询「ReAct 和普通 Agent 有什么区别，各适合什么场景？」（多维度问题）：

Reranker 精排结果：
| 粗排 | 精排 | 分数 | 内容 |
|:---|:---|:---|:---|
| 第 2 → | **第 1 ↑** | 0.6442 | `agent.md`：ReAct 工作流程（Thought→Action→Observation） |
| 第 1 → | 第 2 ↓ | 0.3967 | `agent.md`：什么是 AI Agent（整体介绍） |
| 第 3 → | 第 3 → | 0.0054 | `agent.md`：多 Agent 协作模式 |

LLM 回答质量良好：清晰区分 ReAct/Plan-and-Execute/Reflexion，给出具体适用场景。✅

---

## 8. 待消化 / 可自己玩

- **效果对比**：同一个问题用 `none` vs `rewrite` vs `hyde` vs `multi_query` 分别跑一遍，对比检索到的 chunks 和最终回答质量
- **HyDE 的风险边界**：找一个 LLM 不熟悉的领域，看假设答案是否方向跑偏，导致检索结果变差
- **Multi-Query 子查询数量**：把 `n=3` 改成 `n=5`，看覆盖是否更广，但 API 调用成本也会上升
- **联合策略**：先 Multi-Query 分解，每个子查询再 rewrite，理论上覆盖更全（但 API 成本 ×2）
- **子查询去噪**：LLM 有时生成相似度很高的子查询，可以加余弦相似度过滤，去掉语义重复的

---

## 9. 下一步：评估（Phase 2 第四块）

Phase 2 的最后一块——如果没有量化评估，就不知道这些优化到底有没有效果。

- **检索评估**：Precision@K、Recall@K、MRR、NDCG
- **生成评估**：Faithfulness（忠实度）、Relevance（相关性）、Groundedness（有据性）
- 工具：**RAGAS**（专门为 RAG 评估设计）

> 新对话开场建议说：「我在学 Phase 2，前三块 Hybrid / Reranking / Query Rewriting 已落地（见 phase2_01_hybrid_search.md / phase2_02_reranking.md / phase2_03_query_rewriting.md），现在要做第四块评估（RAGAS）」。
