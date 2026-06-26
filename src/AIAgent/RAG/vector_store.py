"""
向量存储模块

职责：存储和检索文本块的向量表示

本模块实现了两个版本：
1. SimpleVectorStore —— 纯 Python 手写版，帮助你理解底层原理
2. ChromaVectorStore —— 使用 Chroma 向量数据库，适合实际使用

为什么先手写一个？
  向量数据库的核心逻辑其实很简单：
  1. 存：保存向量和对应的文本
  2. 查：计算查询向量和所有存储向量的相似度，返回 Top-K

  手写一遍你就会发现，向量数据库并没有那么神秘。
  Chroma/FAISS 等工具只是在此基础上做了大规模优化（索引结构、批量计算等）。
"""

import json
import os
import numpy as np
from chunker import Chunk
from embedder import cosine_similarity


class SimpleVectorStore:
    """
    手写的简易向量存储

    底层就是两个列表：
    - vectors: 所有向量
    - chunks: 所有对应的文本块

    检索就是暴力遍历计算相似度。
    数据量小（< 10000 条）时完全够用。
    """

    def __init__(self):
        self.vectors: list[list[float]] = []
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]):
        """
        添加数据

        Args:
            chunks: 文本块列表
            vectors: 对应的向量列表（顺序要一致）
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) 和 vectors ({len(vectors)}) 数量不一致"
            )

        self.vectors.extend(vectors)
        self.chunks.extend(chunks)
        print(f"  💾 已存入 {len(chunks)} 条数据，总计 {len(self.vectors)} 条")

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """
        搜索最相似的文本块

        核心逻辑（暴力搜索）：
        1. 计算查询向量和每个存储向量的余弦相似度
        2. 按相似度降序排序
        3. 返回 Top-K 个结果

        Args:
            query_vector: 查询的 embedding 向量
            top_k: 返回的结果数量

        Returns:
            [{"chunk": Chunk, "score": float}, ...]
        """
        if not self.vectors:
            return []

        # 计算和所有向量的相似度
        scores = []
        for i, vec in enumerate(self.vectors):
            score = cosine_similarity(query_vector, vec)
            scores.append((i, score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 Top-K
        results = []
        for i, score in scores[:top_k]:
            results.append(
                {
                    "chunk": self.chunks[i],
                    "score": score,
                }
            )

        return results

    def save(self, filepath: str):
        """
        持久化到磁盘

        保存为 JSON 格式（简单但不高效，大数据量应该用更好的格式）
        """
        data = {
            "vectors": self.vectors,
            "chunks": [
                {"content": c.content, "metadata": c.metadata} for c in self.chunks
            ],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  💾 已保存到 {filepath}")

    def load(self, filepath: str):
        """从磁盘加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vectors = data["vectors"]
        self.chunks = [
            Chunk(content=c["content"], metadata=c["metadata"]) for c in data["chunks"]
        ]
        print(f"  📂 已加载 {len(self.vectors)} 条数据")

    def clear(self):
        """清空所有数据"""
        self.vectors = []
        self.chunks = []
        print("  🗑️ 已清空数据")

    def __len__(self):
        return len(self.vectors)


class ChromaVectorStore:
    """
    基于 Chroma 的向量存储

    Chroma 是一个轻量级向量数据库，特点：
    - 像 SQLite 一样简单，嵌入式使用
    - 自动处理向量索引和检索优化
    - 支持持久化到磁盘
    - 支持元数据过滤

    适合学习和中小规模应用。生产环境可以换成 Milvus/Pinecone 等。
    """

    def __init__(
        self, collection_name: str = "rag_collection", persist_dir: str = "chroma_db"
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("需要安装 chromadb: pip install chromadb")

        self.persist_dir = persist_dir
        # 创建/连接持久化的 Chroma 客户端
        self.client = chromadb.PersistentClient(path=persist_dir)
        # 获取或创建 collection（类似数据库中的表）
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦距离
        )
        print(
            f"  🗄️ Chroma collection '{collection_name}' 就绪 (现有 {self.collection.count()} 条数据)"
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]):
        """添加数据到 Chroma"""
        if not chunks:
            return

        # Chroma 要求每条数据有唯一 ID
        existing_count = self.collection.count()
        ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]

        # 准备元数据（Chroma 要求元数据值是基本类型）
        metadatas = []
        for chunk in chunks:
            meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            documents=[c.content for c in chunks],
            embeddings=vectors,
            metadatas=metadatas,
        )
        print(f"  💾 已存入 {len(chunks)} 条数据，总计 {self.collection.count()} 条")

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """搜索最相似的文本块"""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
        )

        # 转换成统一的返回格式
        output = []
        for i in range(len(results["ids"][0])):
            chunk = Chunk(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            # Chroma 返回的是距离，需要转成相似度
            # 余弦距离 = 1 - 余弦相似度
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1 - distance
            output.append({"chunk": chunk, "score": score})

        return output

    def clear(self):
        """清空所有数据"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  🗑️ 已清空数据")

    def __len__(self):
        return self.collection.count()


# ===== 测试 =====
if __name__ == "__main__":
    # 测试 SimpleVectorStore
    print("=" * 50)
    print("测试 SimpleVectorStore")
    print("=" * 50)

    store = SimpleVectorStore()

    # 造一些测试数据
    chunks = [
        Chunk(content="机器学习是人工智能的一个分支", metadata={"source": "test"}),
        Chunk(content="深度学习使用多层神经网络", metadata={"source": "test"}),
        Chunk(content="今天的天气非常好", metadata={"source": "test"}),
    ]

    from embedder import APIEmbedder
    from config import config

    embedder = APIEmbedder(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        model=config.embedding_model,
    )
    vectors = embedder.embed_texts([c.content for c in chunks])
    query = embedder.embed_query("机器学习")
    store.add(chunks, vectors)

    # 搜索一个和 "机器学习" 相似的向量
    results = store.search(query, top_k=3)
    print("\n搜索结果:")
    for r in results:
        print(f"  Score: {r['score']:.4f} | {r['chunk'].content}")
