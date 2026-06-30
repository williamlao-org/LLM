"""检索器的公共数据结构和接口。"""

from dataclasses import dataclass
from typing import Protocol

from phase1_chunker import Chunk


@dataclass(slots=True)
class SearchResult:
    """一条统一的检索结果。"""

    chunk: Chunk
    score: float
    # 仅融合结果会填写，例如 [1, 3] 表示 Dense 第 1、Sparse 第 3。
    ranks: list[int | None] | None = None


class Retriever(Protocol):
    """所有面向文本查询的检索器都遵循这个接口。"""

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]: ...


class Embedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class VectorStore(Protocol):
    def search(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[SearchResult]: ...


class DenseRetriever:
    """把文本向量化和向量库搜索组合成统一的文本检索接口。"""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.search(query_vector, top_k=top_k)
