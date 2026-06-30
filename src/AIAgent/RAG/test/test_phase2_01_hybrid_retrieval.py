from phase1_chunker import Chunk
from phase2_01_hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from phase1_dense_retriever import DenseRetriever, SearchResult
from phase2_01_sparse_retriever import BM25Retriever
from phase1_vector_store import SimpleVectorStore


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        assert query == "语义查询"
        return [1.0, 0.0]


def test_dense_and_sparse_return_search_results():
    chunks = [Chunk("向量检索"), Chunk("关键词检索")]

    vector_store = SimpleVectorStore()
    vector_store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    dense_results = DenseRetriever(FakeEmbedder(), vector_store).search("语义查询")

    sparse_retriever = BM25Retriever()
    sparse_retriever.add(chunks)
    sparse_results = sparse_retriever.search("关键词")

    assert all(isinstance(result, SearchResult) for result in dense_results)
    assert all(isinstance(result, SearchResult) for result in sparse_results)
    assert dense_results[0].chunk == chunks[0]


def test_rrf_returns_ranked_search_results():
    shared = Chunk("两路共同命中", {"source": "doc", "chunk_index": 0})
    dense_only = Chunk("仅 Dense 命中", {"source": "dense", "chunk_index": 0})

    results = reciprocal_rank_fusion(
        [
            [SearchResult(shared, 0.9), SearchResult(dense_only, 0.8)],
            [SearchResult(shared, 4.2)],
        ],
        top_k=2,
    )

    assert results[0].chunk == shared
    assert results[0].ranks == [1, 1]
    assert results[1].ranks == [2, None]


def test_hybrid_accepts_retrievers_instead_of_embedder_and_store():
    chunks = [Chunk("两路检索")]
    vector_store = SimpleVectorStore()
    vector_store.add(chunks, [[1.0, 0.0]])
    dense_retriever = DenseRetriever(FakeEmbedder(), vector_store)

    sparse_retriever = BM25Retriever()
    sparse_retriever.add(chunks)

    hybrid = HybridRetriever(dense_retriever, sparse_retriever)
    results = hybrid.search("语义查询", top_k=1)

    assert results[0].chunk == chunks[0]
    assert results[0].ranks == [1, 1]


def test_rrf_does_not_merge_different_pages_with_same_chunk_index():
    page_1 = Chunk(
        "PDF 第 1 页",
        {
            "document_id": "pdf-id",
            "source": "book.pdf",
            "part_index": 0,
            "chunk_index": 0,
        },
    )
    page_105 = Chunk(
        "PDF 第 105 页",
        {
            "document_id": "pdf-id",
            "source": "book.pdf",
            "part_index": 104,
            "chunk_index": 0,
        },
    )

    results = reciprocal_rank_fusion(
        [
            [SearchResult(page_1, 0.9), SearchResult(page_105, 0.8)],
            [SearchResult(page_105, 5.0), SearchResult(page_1, 4.0)],
        ],
        top_k=2,
    )

    assert {result.chunk.content for result in results} == {
        "PDF 第 1 页",
        "PDF 第 105 页",
    }
    assert all(result.score <= 2 / 61 for result in results)


def test_rrf_counts_duplicate_chunk_only_once_per_result_list():
    chunk = Chunk("same", {"document_id": "doc", "part_index": 0, "chunk_index": 0})

    results = reciprocal_rank_fusion(
        [[SearchResult(chunk, 1.0), SearchResult(chunk, 0.5)]],
        top_k=1,
    )

    assert results[0].score == 1 / 61
    assert results[0].ranks == [1]
