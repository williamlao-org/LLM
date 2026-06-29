# Phase 2 · 第一块：混合检索 Hybrid Search（BM25 + RRF）

> 本文是学习笔记 + 「新对话快速启动器」。
> 新开会话学 Phase 2 时，把这份文件喂给我即可立刻接上下文，无需重新铺垫。

---

## 0. 我现在在哪（进度锚点）

```
Phase 1 ✅ 经典 RAG（document_loader / chunker / embedder / vector_store / rag_chain / main）
Phase 2 🚧 进阶 RAG
   ├─ 第一块 ✅ 混合检索 Hybrid Search（本文）   ← 已落地、已跑通
   ├─ 第二块 ✅ Reranking（Cross-encoder 二次精排）
   ├─ 第三块 ⬜ Query Rewriting / HyDE / Multi-Query
   └─ 第四块 ⬜ 评估（RAGAS：Precision@K / Faithfulness ...）
Phase 3+ ⬜ Agentic RAG / 记忆系统 / GraphRAG（见 learning_roadmap.md）
```

学习方式偏好：**不上来就用框架**，先手写理解原理；小步推进；对比驱动（A/B）。

---

## 1. 为什么需要 Hybrid

Phase 1 用的是 **Dense 检索**（向量/语义）：问题和 chunk 都转向量，比余弦相似度。
强在懂语义，弱在**精确词、专有名词、编号、罕见术语**。

| 维度     | Dense（向量）                   | Sparse（BM25）                   |
| :------- | :------------------------------ | :------------------------------- |
| 本质     | 语义相似度                      | 升级版关键词匹配 + 词频权重      |
| 强项     | 同义改写、语义联想              | 精确命中专名/型号/编号           |
| 盲区     | 型号词被稀释、编号无区分度      | 词不重叠 = 0 分，无语义联想      |
| 典型翻车 | "BGE-M3 的维度"、"错误码 40301" | "怎么训练神经网络" ↔ "反向传播" |

两者**互补**，所以两路各自检索再融合 → Hybrid Search，是生产级 RAG 性价比最高的升级之一。

---

## 2. BM25（Sparse）原理

词袋模型。对查询 Q 和文档 D：

```
                        f(qi, D) · (k1 + 1)
    score(Q, D) = Σ IDF(qi) · ──────────────────────────────────
                  qi∈Q        f(qi, D) + k1 · (1 - b + b · |D|/avgdl)
```

| 符号           | 含义                   | 直觉                                    |
| :------------- | :--------------------- | :-------------------------------------- |
| `f(qi, D)`   | 词 qi 在 D 中的词频 TF | 命中越多越相关                          |
| `IDF(qi)`    | 逆文档频率             | **罕见的词更值钱**（"的""是"≈0） |
| `\|D\|, avgdl` | 文档长度、全库平均长度 | 长文档天然词多                          |
| `k1≈1.5`    | 词频饱和               | 出现 10 次 ≠ 5 次的 2 倍重要           |
| `b≈0.75`    | 长度惩罚               | **长文档打折**，避免霸榜          |

三句话记住：**罕见词更值钱(IDF) + 命中越多越好但有上限(k1) + 长文档打折(b)**。

IDF 用带平滑版本，`+0.5` 防除零、`+1` 保证恒正（经典 BM25 对超高频词会出负分）：

```
IDF(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )      # df = 含该词的文档数，N = 总文档数
```

**中文要先分词**（BM25 是词袋）。代码做了双模式：有 `jieba` 走词级，没装退化到字符级（也能用，精度略逊）。装：`uv add jieba`。

---

## 3. RRF（融合）原理

难点：Dense 余弦 ∈ [-1,1]，BM25 ∈ [0,+∞)，**量纲不同**，直接相加会被 BM25 淹没，min-max 归一化又对离群值敏感。

**RRF（Reciprocal Rank Fusion）只看名次、丢弃原始分**：

```
                      1
    RRF(d) = Σ   ───────────────         k 惯例取 60
             各列表   k + rank(d)         rank 从 1 开始
```

例（k=60）：某 chunk 在 Dense 排第 1、Sparse 排第 3：
`RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0322`

- 因为只比名次，**余弦和 BM25 的量纲差异天然消失** → 比归一化稳。
- 两路都靠前 = 强共识 = 排最前；只在一路靠前的也能进榜（被"捞"回来）。

---

## 4. 代码落地（文件 / 接口 / 跑法）

### 新增文件

| 文件                    | 内容                | 关键接口                                                                                 |
| :---------------------- | :------------------ | :--------------------------------------------------------------------------------------- |
| `sparse_retriever.py` | 手写 BM25 + 分词    | `BM25Retriever().add(chunks)` / `.search(query_str, top_k)`                          |
| `hybrid_retriever.py` | RRF 纯函数 + 编排类 | `reciprocal_rank_fusion(lists, k, top_k)` / `HybridRetriever(dense_retriever, sparse_retriever)` |

### 接口对齐（刻意和 Phase 1 一致，便于融合）

- 两种检索器的 `search` 都返回 `list[SearchResult]`
- 区别：`SimpleVectorStore.search(向量)` vs `BM25Retriever.search(字符串)`
- `HybridRetriever.search` 额外返回 `"ranks"`（如 `[1,3]` = Dense第1/Sparse第3），纯为可解释性

### 接进 rag_chain.py

`RAGChain.__init__` 新增开关 `retriever_type="dense" | "hybrid"`（与 `store_type` 同风格）：

- `build_index`：Step 4 同一批 chunk 同时灌进向量库和 BM25
- `load_index`：BM25 **不入磁盘缓存**，从加载到的 chunks 现场重建
- `query`：hybrid 时走 `self.hybrid.search()`，dense 时维持原路径
- ⚠️ 已知小瑕疵：hybrid 下 `query(verbose=True)` 打印的 `(相似度: 0.07..)` 其实是 **RRF 分数**，标签未改，不影响功能

### 跑

```bash
uv run python sparse_retriever.py    # 单测 BM25
uv run python hybrid_retriever.py    # 单测 Dense+BM25+RRF（调真实 embedding API）
uv run python rag_chain.py           # 全链路（__main__ 已切到 retriever_type="hybrid"）
```

### 关键设计：candidate_k

每路先各取 `top_k * 3` 个候选再融合，而不是只取 top_k。
否则"单路靠前、另一路靠后"的好结果根本进不了融合池。

---

## 5. 实测观察（对比驱动）

用 4 条样本（含 "BGE-M3"、"语义相近的句子" 等）实测：

1. **量纲被抹平**：原始分 Dense `0.26~0.77` / Sparse `0~8.2`，融合后全挤在 `0.031~0.033`。
2. **共识稳居第一**：名次 `[1,1]` 的永远最前。
3. **捞回效应**：查询"怎么让语义相近的句子靠在一起"，BGE-M3 那条在 Dense 排第 4（差点出局）、Sparse 排第 2，RRF 把它捞回第 3 —— **单用 Dense 会漏掉**。这就是 Hybrid 的全部价值。
4. **Sparse 非黑即白**：词不重叠的 chunk BM25 直接 0 分；Dense 则给个非零弱相似度。

---

## 6. 待消化 / 可自己玩

- IDF 里 `+0.5` 和 `+1` 各自的作用（平滑 / 防负分）。
- 调 `BM25Retriever(k1=?, b=?)` 看排序变化：k1↑ 更看重高频词；b↑ 更狠惩罚长文档。
- 装 jieba 前后对比词表大小和排序稳定性。
- RRF 的 `k` 改小（如 10）会发生什么？（头部名次差距被放大，更"信任"第一名）

---

## 7. 下一步：Reranking（Phase 2 第二块）

- **动机**：Hybrid 召回的是"粗排"候选；Cross-encoder 把 query 和每个候选**拼在一起**送进模型，输出精确相关分，做二次精排。
- 和现在的区别：Dense/BM25 是 query 和 doc **各自编码**（双塔，快但粗）；Cross-encoder 是**联合编码**（准但慢），所以只用来重排 Top-N，不用来初检。
- 典型流水线终态：`Hybrid 召回 Top-20 → Reranker 精排 → 取 Top-3 → 注入 Prompt`。
- 候选模型：`BAAI/bge-reranker-v2-m3`（和现在的 bge-m3 同源，SiliconFlow 有 API）。

> 新对话开场建议说：「我在学 Phase 2，第一块 Hybrid 已落地（见 phase2_hybrid_search.md），现在要做第二块 Reranking」。
