"""
重排序模块（Reranker）—— Cross-encoder 二次精排

Phase 2 Hybrid Search 解决了"召回"，但召回结果仍然是粗排：
  - Dense（双塔）：query 和 doc 各自编码成向量，比余弦相似度
  - BM25：纯词频统计
  - RRF：只看名次，不看语义

这些方法的共同问题：query 和 doc 是**独立编码**的，无法捕获它们之间的
细粒度词级交互。比如：
  query = "Python 怎么处理 JSON"
  chunk A = "Python 的 json 模块提供 loads() 和 dumps() 方法"
  chunk B = "Python 是一门流行的编程语言，支持多种数据格式"

双塔模型可能觉得 A 和 B 都和 "Python" + "数据格式" 语义相关，分数接近。
但 Cross-encoder 把 query 和 chunk **拼在一起**送进 Transformer，
能看到 "处理 JSON" 和 "json 模块 loads() dumps()" 之间的精确对应关系，
从而给 A 打出远高于 B 的分数。

这就是 Reranking 的价值：**用更贵但更准的模型，对少量候选做二次精排**。

典型流水线：
  Hybrid 召回 Top-20（粗排，快）
      │
      ▼
  Cross-encoder Rerank（精排，只对 20 条打分，慢但准）
      │
      ▼
  取 Top-3 → 注入 Prompt → LLM 生成

为什么不直接用 Cross-encoder 做初检？
  因为它要对每一对 (query, doc) 都过一遍模型，复杂度 O(N)。
  如果知识库有 10 万个 chunk，每次查询都要跑 10 万次模型推理——太慢了。
  所以 Cross-encoder 只适合重排少量候选（比如 10~30 条），不适合初检。

双塔 vs Cross-encoder 对比：

  ┌──────────────────────────────────────────────────────┐
  │  双塔（Bi-encoder）           Cross-encoder          │
  │                                                      │
  │  query → [模型A] → vec_q     query ─┐               │
  │                        ↘       │     ├→ [模型] → 分数 │
  │  doc   → [模型B] → vec_d  cos(q,d)  │               │
  │                                doc ──┘               │
  │                                                      │
  │  各自编码，快                   联合编码，准           │
  │  适合初检百万候选               适合精排几十条         │
  └──────────────────────────────────────────────────────┘
"""

import httpx

from phase1_dense_retriever import SearchResult


class APIReranker:
    """
    通过 API 调用 Cross-encoder Reranker。

    使用 SiliconFlow 的 /rerank endpoint（兼容 Cohere 风格）。
    模型：BAAI/bge-reranker-v2-m3（和我们的 bge-m3 embedding 同源）。

    用法：
        reranker = APIReranker(api_key="...", model="BAAI/bge-reranker-v2-m3")
        reranked = reranker.rerank(query, hybrid_results, top_n=3)

    设计要点：
      - 输入是 Hybrid 召回的 list[SearchResult]
      - 输出也是 list[SearchResult]，score 更新为 reranker 的相关性分数
      - 不修改 SearchResult 的数据结构，保持简洁
    """

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 3,
        verbose: bool = False,
    ) -> list[SearchResult]:
        """
        对召回结果做 Cross-encoder 重排序。

        工作流程：
          1. 把每条 SearchResult 的 chunk.content 抽出来作为 documents
          2. 调用 SiliconFlow /rerank API，模型对每个 (query, doc) 对打分
          3. API 返回按相关性降序的结果（含 index 和 relevance_score）
          4. 按 reranker 分数重新排序，取 top_n 条返回

        Args:
            query: 用户原始查询
            results: Hybrid 召回的候选列表（粗排结果）
            top_n: 精排后保留多少条
            verbose: 是否打印重排前后的对比

        Returns:
            重排后的 SearchResult 列表，score 已更新为 reranker 分数
        """
        if not results:
            return []

        # 如果候选数不超过 top_n，还是要 rerank（排序可能变化），
        # 但 top_n 不能超过实际候选数
        top_n = min(top_n, len(results))

        # ---- Step 1：构造 API 请求 ----
        documents = [r.chunk.content for r in results]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "return_documents": False,  # 不需要返回原文，我们已经有了
            "top_n": top_n,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # ---- Step 2：调用 API ----
        response = httpx.post(
            f"{self.base_url}/rerank",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        # ---- Step 3：解析结果，重建 SearchResult ----
        # API 返回格式：
        # {
        #   "results": [
        #     {"index": 0, "relevance_score": 0.95},
        #     {"index": 2, "relevance_score": 0.82},
        #     ...
        #   ]
        # }
        reranked_items = data["results"]

        if verbose:
            print("  [Reranker 重排]")

        reranked: list[SearchResult] = []
        for new_rank, item in enumerate(reranked_items, start=1):
            original_idx = item["index"]
            reranker_score = item["relevance_score"]
            original_result = results[original_idx]

            reranked.append(
                SearchResult(
                    chunk=original_result.chunk,
                    score=reranker_score,
                    ranks=original_result.ranks,  # 保留原始的 Dense/Sparse 名次
                )
            )

            if verbose:
                old_rank = original_idx + 1  # 粗排时的位置（1-indexed）
                move = "→" if old_rank == new_rank else ("↑" if old_rank > new_rank else "↓")
                preview = original_result.chunk.content[:50].replace("\n", " ")
                print(
                    f"    {new_rank}. (rerank={reranker_score:.4f}) "
                    f"粗排第{old_rank} {move} "
                    f"{preview!r}..."
                )

        return reranked


# ===== 测试 =====
if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from chunker import Chunk
    from config import config

    print("=" * 60)
    print("测试 APIReranker（Cross-encoder 重排序）")
    print("=" * 60)

    # 构造一些候选 chunk，模拟 Hybrid 召回的粗排结果
    # 故意让顺序"不太对"，看 reranker 能不能纠正
    test_chunks = [
        Chunk(
            content="向量数据库通过近似最近邻搜索（ANN）来加速大规模相似度检索",
            metadata={"source": "doc1", "chunk_index": 0},
        ),
        Chunk(
            content="BGE-M3 是一个多语言的 embedding 模型，支持中英文，输出 1024 维向量",
            metadata={"source": "doc2", "chunk_index": 0},
        ),
        Chunk(
            content="今天天气非常好，适合出去散步晒太阳，公园里的花都开了",
            metadata={"source": "doc3", "chunk_index": 0},
        ),
        Chunk(
            content="Embedding 模型把文本转成向量后，语义相近的句子在向量空间里距离也相近",
            metadata={"source": "doc4", "chunk_index": 0},
        ),
        Chunk(
            content="BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率进行关键词匹配",
            metadata={"source": "doc5", "chunk_index": 0},
        ),
    ]

    from embedder import APIEmbedder
    from vector_store import SimpleVectorStore
    from retriever import DenseRetriever
    from phase2_01_sparse_retriever import BM25Retriever
    from phase2_01_hybrid_retriever import HybridRetriever

    # 1. 初始化 Dense 检索器
    embedder = APIEmbedder(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        model=config.embedding_model,
    )
    vectors = embedder.embed_texts([c.content for c in test_chunks])

    dense_store = SimpleVectorStore()
    dense_store.add(test_chunks, vectors)
    dense_retriever = DenseRetriever(embedder, dense_store)

    # 2. 初始化 Sparse 检索器
    sparse_retriever = BM25Retriever()
    sparse_retriever.add(test_chunks)

    # 3. 构造 Hybrid 检索器
    hybrid = HybridRetriever(dense_retriever, sparse_retriever)

    reranker = APIReranker(
        api_key=config.reranker_api_key,
        model=config.reranker_model,
        base_url=config.reranker_base_url,
    )

    query = "Embedding 模型是怎么把文本变成向量的"

    print(f"\n🔍 查询: {query}")
    
    # 真实进行 Hybrid 检索召回候选
    hybrid_results = hybrid.search(query, top_k=5, verbose=True)
    print(f"\n  [粗排顺序（Hybrid/RRF 融合结果）] 候选数: {len(hybrid_results)}")
    for i, r in enumerate(hybrid_results, 1):
        print(f"    {i}. (RRF={r.score:.4f}, ranks={r.ranks}) "
              f"{r.chunk.content[:50]!r}...")

    print()
    # 精排
    reranked = reranker.rerank(query, hybrid_results, top_n=3, verbose=True)

    print("\n  [精排结果（Reranker Top-3）]")
    for i, r in enumerate(reranked, 1):
        print(f"    {i}. (rerank={r.score:.4f}) {r.chunk.content[:60]!r}...")
