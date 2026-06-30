"""
混合检索模块（Hybrid Search）—— Dense + Sparse 融合

把两条腿合起来：
  - Dense（向量检索，Phase 1 的 SimpleVectorStore）：懂语义、抗改写
  - Sparse（BM25，phase2_01_sparse_retriever.py）：精确命中关键词、专名、编号

融合用 RRF（Reciprocal Rank Fusion，倒数排名融合）。

为什么不直接把两路分数相加？
  余弦相似度 ∈ [-1, 1]，BM25 ∈ [0, +∞)，量纲完全不同，
  相加会被 BM25 的大分淹没，min-max 归一化又对离群值敏感、不稳定。

RRF 的做法：只看每个 chunk 在各列表里的"名次"，丢弃原始分数。

                          1
    RRF(d) = Σ      ───────────────
             各列表       k + rank(d)

    rank(d) : d 在该列表中的排名（第 1 名 rank=1，以此类推）
    k       : 常数（惯例 60），压平名次差距、抑制头部垄断

  因为只比名次，余弦和 BM25 的量纲差异天然消失——这是它比归一化稳的关键。
"""

from phase1_chunker import Chunk
from phase1_dense_retriever import DenseRetriever, Retriever, SearchResult
from phase2_01_sparse_retriever import BM25Retriever


# ========== 核心：RRF 融合（纯函数，便于单测和复用） ==========


def _chunk_key(chunk: Chunk) -> str:
    """
    给 chunk 生成一个稳定的去重键，用来判断"两路检索是不是同一个 chunk"。

    PDF 等文档会被拆成多个 part（例如每页一个 part），而每个
    part 的 chunk_index 都从 0 开始，因此唯一键必须包含 part_index。
    没有完整元数据时退化到用内容本身。
    """
    document_id = (
        chunk.metadata.get("document_id")
        or chunk.metadata.get("filepath")
        or chunk.metadata.get("source")
    )
    part_idx = chunk.metadata.get("part_index")
    idx = chunk.metadata.get("chunk_index")
    if document_id is not None and idx is not None:
        if part_idx is not None:
            return f"{document_id}::part={part_idx}::chunk={idx}"
        return f"{document_id}::chunk={idx}"
    return chunk.content


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int = 3,
) -> list[SearchResult]:
    """
    把多条已排序的检索结果列表融合成一条。

    Args:
        result_lists: 多路已按各自相关度降序排好的 SearchResult 列表
        k: RRF 常数，越大则越淡化名次差距
        top_k: 融合后返回多少条

    Returns:
        融合后的 SearchResult 列表
        score 是 RRF 分数；ranks 记录它在每一路的名次（None 表示该路没检索到），
        方便你观察融合是怎么发生的。
    """
    # key -> 累加的 RRF 分数
    fused_scores: dict[str, float] = {}
    # key -> chunk 对象（去重后只留一份）
    key_to_chunk: dict[str, Chunk] = {}
    # key -> 它在每一路的名次，纯粹为了可解释性
    key_to_ranks: dict[str, list[int | None]] = {}

    n_lists = len(result_lists)

    for list_idx, results in enumerate(result_lists):
        # 同一路中的同一个 chunk 最多贡献一次，避免异常重复累加。
        seen_keys: set[str] = set()
        for rank, item in enumerate(results, start=1):  # rank 从 1 开始
            chunk = item.chunk
            key = _chunk_key(chunk)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # 第一次见到这个 chunk，初始化
            if key not in fused_scores:
                fused_scores[key] = 0.0
                key_to_chunk[key] = chunk
                key_to_ranks[key] = [None] * n_lists

            # RRF 核心累加：这一路贡献 1 / (k + rank)
            fused_scores[key] += 1.0 / (k + rank)
            key_to_ranks[key][list_idx] = rank

    # 按融合分数降序
    ranked_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    output: list[SearchResult] = []
    for key in ranked_keys[:top_k]:
        output.append(
            SearchResult(
                chunk=key_to_chunk[key],
                score=fused_scores[key],
                ranks=key_to_ranks[key],  # 例如 [1, 3] = Dense 第1、Sparse 第3
            )
        )
    return output


# ========== 编排层：Hybrid 检索器 ==========


class HybridRetriever:
    """
    混合检索器：编排 Dense + Sparse + RRF。

    用法（注入两个已经就绪的检索器）：
        hybrid = HybridRetriever(dense_retriever, sparse_retriever)
        results = hybrid.search("问题", top_k=3)

    设计要点：
      - Dense/Sparse 都实现 Retriever.search(query, top_k)
      - 本类不关心 Embedding、向量库和建索引，只负责召回与融合
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 3,
        candidate_k: int | None = None,
        verbose: bool = False,
    ) -> list[SearchResult]:
        """
        混合检索。

        Args:
            query: 原始查询字符串（Dense 这边内部转向量，Sparse 这边内部分词）
            top_k: 最终返回数量
            candidate_k: 每一路各取多少候选去融合（默认 top_k * 3）。
                         取大一点能让"只在某一路靠前"的好结果有机会进入融合。
            verbose: 打印两路各自的命中，方便对比观察

        Returns:
            RRF 融合后的 SearchResult 列表
        """
        candidate_k = candidate_k or top_k * 3

        # ---- 第 1 路：Dense（向量检索）----
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)

        # ---- 第 2 路：Sparse（BM25）----
        sparse_results = self.sparse_retriever.search(query, top_k=candidate_k)

        if verbose:
            print("  [Dense 召回]")
            for i, r in enumerate(dense_results, 1):
                print(f"    {i}. ({r.score:.4f}) {r.chunk.content[:50]!r}...")
            print("  [Sparse 召回]")
            for i, r in enumerate(sparse_results, 1):
                print(f"    {i}. ({r.score:.4f}) {r.chunk.content[:50]!r}...")

        # ---- 融合 ----
        fused = reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            top_k=top_k,
        )
        return fused


# ===== 测试 =====
if __name__ == "__main__":
    from embedder import APIEmbedder
    from vector_store import SimpleVectorStore
    from config import config

    print("=" * 60)
    print("测试 HybridRetriever（Dense + BM25 + RRF）")
    print("=" * 60)

    chunks = [
        Chunk(content="BGE-M3 是一个开源的中文 embedding 模型，输出 1024 维向量",
              metadata={"source": "doc1", "chunk_index": 0}),
        Chunk(content="向量数据库通过近似最近邻搜索（ANN）来加速相似度检索",
              metadata={"source": "doc2", "chunk_index": 0}),
        Chunk(content="把文本转成向量后，语义相近的句子在空间里距离也相近",
              metadata={"source": "doc3", "chunk_index": 0}),
        Chunk(content="今天天气非常好，适合出去散步晒太阳",
              metadata={"source": "doc4", "chunk_index": 0}),
    ]

    embedder = APIEmbedder(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        model=config.embedding_model,
    )
    vectors = embedder.embed_texts([c.content for c in chunks])

    dense_store = SimpleVectorStore()
    dense_store.add(chunks, vectors)
    dense_retriever = DenseRetriever(embedder, dense_store)

    sparse_retriever = BM25Retriever()
    sparse_retriever.add(chunks)

    hybrid = HybridRetriever(dense_retriever, sparse_retriever)

    # 这个查询故意用"语义相关但没有精确关键词"和"精确关键词"混合的问法
    for q in ["BGE-M3 的维度是多少", "怎么让语义相近的句子靠在一起"]:
        print(f"\n🔍 查询: {q}")
        results = hybrid.search(q, top_k=3, verbose=True)
        print("  [RRF 融合后]")
        for i, r in enumerate(results, 1):
            print(f"    {i}. (RRF={r.score:.4f}, 名次 Dense/Sparse={r.ranks}) "
                  f"{r.chunk.content[:50]!r}...")
