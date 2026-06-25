"""
Embedding 模块

职责：把文本转成向量（一组浮点数）

核心原理：
  Embedding 模型在大量文本对上训练，学会了把"语义相似"的文本
  映射到向量空间中距离相近的位置。

  "今天天气真好" → [0.12, -0.34, 0.78, ...]
  "天气不错"     → [0.11, -0.32, 0.80, ...]   ← 语义相似，向量接近
  "量子力学导论" → [0.89, 0.45, -0.23, ...]   ← 语义不同，向量远离

本模块设计了两种 Embedding 方式：
  1. API Embedding：调用 OpenAI 兼容格式的 embedding 接口
  2. 本地 Sentence-Transformers：用 HuggingFace 模型在本地计算

为什么需要两种？
  - 有些国内 API 服务商没有专门的 embedding 接口
  - 本地模型免费且隐私安全，适合学习和小规模使用
"""

import numpy as np
from openai import OpenAI


class APIEmbedder:
    """
    通过 API 计算 Embedding

    兼容所有 OpenAI 格式的 embedding 接口，包括：
    - OpenAI text-embedding-3-small/large
    - 通义千问 text-embedding-v2
    - 智谱 embedding-2
    - 等等
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self._dim = None  # 惰性获取维度

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        批量计算文本的 embedding

        Args:
            texts: 文本列表

        Returns:
            embedding 向量列表，每个向量是一个 float 列表
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )

        # 按 index 排序确保顺序一致
        embeddings = sorted(response.data, key=lambda x: x.index)
        vectors = [e.embedding for e in embeddings]

        # 记录维度
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
            print(f"  📐 Embedding 维度: {self._dim}")

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """
        计算单个查询的 embedding

        为什么单独一个方法？
        有些 embedding 模型区分"文档"和"查询"的 embedding 方式
        （比如 BGE 模型需要给查询加前缀 "Represent this sentence:"）
        这里统一接口，方便后续扩展

        Args:
            query: 查询文本

        Returns:
            embedding 向量
        """
        return self.embed_texts([query])[0]

    @property
    def dim(self) -> int:
        """向量维度"""
        if self._dim is None:
            # 用一个测试文本获取维度
            test_vec = self.embed_query("test")
            self._dim = len(test_vec)
        return self._dim


class LocalEmbedder:
    """
    使用本地 Sentence-Transformers 模型计算 Embedding

    优势：
    - 免费，不需要 API key
    - 隐私安全，数据不出本地
    - 中文模型（如 BAAI/bge-small-zh-v1.5）效果好

    劣势：
    - 需要下载模型（通常几百MB）
    - 第一次加载较慢
    - 大批量计算需要 GPU
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        Args:
            model_name: HuggingFace 模型名称
                推荐中文模型：
                - "BAAI/bge-small-zh-v1.5" (小模型，~90MB)
                - "BAAI/bge-base-zh-v1.5"  (中等模型，~400MB)
                - "BAAI/bge-large-zh-v1.5" (大模型，~1.3GB)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers: pip install sentence-transformers"
            )

        print(f"  🔄 正在加载本地 Embedding 模型: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"  ✅ 模型加载完成，维度: {self._dim}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量计算 embedding"""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """计算单个查询的 embedding"""
        return self.embed_texts([query])[0]

    @property
    def dim(self) -> int:
        return self._dim


# ========== 工具函数 ==========


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度

    公式:
                 A · B          向量点积
    cos(θ) = ────────── = ──────────────────────
              |A| × |B|    两个向量的模的乘积

    返回值范围 [-1, 1]：
       1 = 完全相同方向（最相似）
       0 = 正交（无关）
      -1 = 完全相反方向

    手写这个函数是为了让你理解向量检索的底层原理。
    实际向量数据库内部也是用类似的计算。
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)  # 点积
    norm_a = np.linalg.norm(a)  # A 的模
    norm_b = np.linalg.norm(b)  # B 的模

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


# ===== 测试 =====
if __name__ == "__main__":
    # 测试余弦相似度
    print("余弦相似度测试:")
    print(f"  相同向量: {cosine_similarity([1, 0, 0], [1, 0, 0]):.4f}")  # 1.0
    print(f"  正交向量: {cosine_similarity([1, 0, 0], [0, 1, 0]):.4f}")  # 0.0
    print(f"  相反向量: {cosine_similarity([1, 0, 0], [-1, 0, 0]):.4f}")  # -1.0
    print(f"  相似向量: {cosine_similarity([1, 2, 3], [1, 2, 4]):.4f}")  # ~0.99

    from config import config

    embedder = APIEmbedder(
        base_url=config.embedding_base_url,
        api_key=config.embedding_api_key,
        model=config.embedding_model,
    )
    texts = ["今天天气真好", "天气不错", "量子力学导论"]
    vecs = embedder.embed_texts(texts)
    print(f"\n语义相似度测试:")
    print(f"  '天气真好' vs '天气不错': {cosine_similarity(vecs[0], vecs[1]):.4f}")
    print(f"  '天气真好' vs '量子力学': {cosine_similarity(vecs[0], vecs[2]):.4f}")
