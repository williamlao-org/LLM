"""
稀疏检索模块（BM25）—— Hybrid Search 的另一条腿

Phase 1 的向量检索（Dense）懂语义，但对"精确关键词、专有名词、编号"不敏感。
BM25 是一种 Sparse（稀疏）检索：本质是"升级版的关键词匹配 + 词频权重"，
正好补上 Dense 的短板。两者融合就是生产级 RAG 标配的 Hybrid Search。

为什么先手写一个？
  和你 Phase 1 手写 SimpleVectorStore 的思路一样：
  BM25 听起来高大上，拆开看就三件事——
    1. 罕见的词更值钱        （IDF：逆文档频率）
    2. 命中越多越相关，但有上限（k1：词频饱和）
    3. 长文档要打折          （b：长度归一化）
  手写一遍，rank_bm25 这类库你就知道它在算什么了。

BM25 评分公式（对查询 Q 和文档 D）：

                        f(qi, D) · (k1 + 1)
    score(Q, D) = Σ IDF(qi) · ──────────────────────────────────
                  qi∈Q        f(qi, D) + k1 · (1 - b + b · |D|/avgdl)

    f(qi, D) : 词 qi 在文档 D 中的出现次数（TF，词频）
    IDF(qi)  : 逆文档频率，越罕见权重越高（"的""是"几乎为 0）
    |D|      : 文档长度（词数）;  avgdl : 全库平均文档长度
    k1 (~1.5): 控制词频饱和——出现 10 次不该比 5 次重要整整 2 倍
    b  (~0.75): 控制长度惩罚——长文档天然词多，打折避免霸榜
"""

import math
from collections import Counter
from phase1_chunker import Chunk
from phase1_dense_retriever import SearchResult


# ========== 分词 ==========
# BM25 是"词袋"模型，第一步必须把文本切成词。
# 中文没有空格，需要分词工具；这里优先用 jieba，没装就退化到字符级。
# 字符级（按单字切）对中文也意外地能用，只是精度略逊于 jieba 词级。

try:
    import jieba

    # 关掉 jieba 启动时的日志，保持输出干净
    jieba.setLogLevel("ERROR")
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def tokenize(text: str) -> list[str]:
    """
    把文本切成 token 列表。

    - 有 jieba：中文按词切，更准（"向量数据库" → ["向量", "数据库"]）
    - 没 jieba：退化到字符级（"向量" → ["向", "量"]），并保留连续的英文/数字串

    返回的 token 已统一小写，便于英文大小写无关匹配。
    """
    text = text.lower()

    if _HAS_JIEBA:
        # jieba 切出来会带空格和标点，过滤掉
        return [t for t in jieba.lcut(text) if t.strip()]

    # ---- 字符级退化方案 ----
    # 中文逐字切；连续的英文字母/数字当成一个整体 token（不然 "bge" 被拆成 b/g/e）
    tokens: list[str] = []
    buf = ""
    for ch in text:
        if ch.isascii() and (ch.isalnum()):
            buf += ch  # 累积英文/数字
        else:
            if buf:
                tokens.append(buf)
                buf = ""
            if ch.strip():  # 非空白的中文字符等，逐个加入
                tokens.append(ch)
    if buf:
        tokens.append(buf)
    return tokens


# ========== BM25 检索器 ==========


class BM25Retriever:
    """
    手写的 BM25 稀疏检索器。

    接口刻意和 SimpleVectorStore 对齐，方便后面融合：
        retriever = BM25Retriever()
        retriever.add(chunks)                  # 建索引（不需要向量！）
        results = retriever.search(query, top_k=3)
        # results: [SearchResult(chunk=Chunk, score=float), ...]

    注意：BM25 不需要 Embedding，纯靠词频统计，所以 add() 只吃 chunks。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # 每个文档的 token 列表，与 chunks 一一对应
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []
        # 每个文档的词频表：Counter({token: 次数})
        self.doc_freqs: list[Counter] = []
        # 每个文档的长度（token 数）
        self.doc_lens: list[int] = []
        # 全库平均文档长度
        self.avgdl: float = 0.0
        # 每个 token 的 IDF 值：{token: idf}
        self.idf: dict[str, float] = {}

    def add(self, chunks: list[Chunk]):
        """
        建立 BM25 索引。

        步骤：
        1. 对每个 chunk 分词、统计词频、记录长度
        2. 统计每个词出现在多少篇文档里（文档频率 DF）
        3. 由 DF 算出每个词的 IDF
        """
        if not chunks:
            return

        self.chunks.extend(chunks)

        # ---- Step 1: 分词 + 词频 + 长度 ----
        for chunk in chunks:
            tokens = tokenize(chunk.content)
            self.doc_tokens.append(tokens)
            self.doc_freqs.append(Counter(tokens))
            self.doc_lens.append(len(tokens))

        # 平均文档长度（注意：用全库重新算，支持多次 add）
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

        # ---- Step 2: 文档频率 DF ----
        # df[token] = 含有该 token 的文档数（每篇只算一次，所以遍历 set）
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            for token in freqs.keys():
                df[token] += 1

        # ---- Step 3: IDF ----
        # 用带平滑的 BM25 IDF 公式：
        #   IDF(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )
        # +1 保证 IDF 永远 > 0（经典 BM25 对超高频词会出负数，加 1 规避）
        N = len(self.chunks)
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

        print(
            f"  📚 BM25 索引就绪：{len(self.chunks)} 篇文档，词表 {len(self.idf)} 个 token"
        )

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """计算查询对单篇文档的 BM25 分数（就是上面公式的 Σ）"""
        freqs = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        score = 0.0
        for token in query_tokens:
            if token not in freqs:
                continue  # 文档里没这个词，贡献 0
            tf = freqs[token]
            idf = self.idf.get(token, 0.0)

            # 公式的分子和分母
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        检索最相关的文档。

        注意：参数是原始的查询字符串（不是向量！），内部会自己分词。
        返回格式和 SimpleVectorStore.search 一致，方便融合。
        """
        if not self.chunks:
            return []

        query_tokens = tokenize(query)

        # 给每篇文档打分
        scored = []
        for i in range(len(self.chunks)):
            score = self._score(query_tokens, i)
            scored.append((i, score))

        # 按分数降序，取 Top-K
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for i, score in scored[:top_k]:
            results.append(SearchResult(chunk=self.chunks[i], score=score))
        return results

    def clear(self):
        """清空索引"""
        self.__init__(k1=self.k1, b=self.b)
        print("  🗑️ 已清空 BM25 索引")

    def __len__(self):
        return len(self.chunks)


# ===== 测试 =====
if __name__ == "__main__":
    print("=" * 50)
    print(
        f"测试 BM25Retriever（分词模式：{'jieba 词级' if _HAS_JIEBA else '字符级退化'}）"
    )
    print("=" * 50)

    chunks = [
        Chunk(
            content="BGE-M3 是一个开源的中文 embedding 模型，输出 1024 维向量",
            metadata={"source": "doc1"},
        ),
        Chunk(
            content="向量数据库通过近似最近邻搜索来加速相似度检索",
            metadata={"source": "doc2"},
        ),
        Chunk(content="今天的天气非常好，适合出去散步", metadata={"source": "doc3"}),
        Chunk(content="机器学习是人工智能的一个重要分支", metadata={"source": "doc4"}),
    ]

    retriever = BM25Retriever()
    retriever.add(chunks)

    for q in ["BGE-M3 的维度", "向量检索是怎么加速的", "天气怎么样"]:
        print(f"\n🔍 查询: {q}")
        for r in retriever.search(q, top_k=2):
            print(f"  Score: {r.score:.4f} | {r.chunk.content}")
