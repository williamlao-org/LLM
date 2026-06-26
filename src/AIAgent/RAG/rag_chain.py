"""
RAG Chain —— 串联完整的 RAG 流程

这是整个 RAG 系统的核心编排层。
它把前面所有模块串联起来，完成：

  用户问题
      │
      ▼
  1. Embedding（把问题转向量）
      │
      ▼
  2. 检索（在向量库中找最相关的 chunks）
      │
      ▼
  3. 构造 Prompt（把检索结果 + 问题拼成 Prompt）
      │
      ▼
  4. LLM 生成（调用 LLM 基于上下文回答）
      │
      ▼
  返回答案 + 来源引用
"""

from pathlib import Path
from openai import OpenAI
from embedder import APIEmbedder, LocalEmbedder
from vector_store import SimpleVectorStore, ChromaVectorStore
from hybrid_retriever import HybridRetriever
from document_loader import load_documents
from chunker import chunk_documents, Chunk
from config import config


# ========== RAG Prompt 模板 ==========

RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。你的任务是基于提供的参考资料来回答用户的问题。

请遵循以下规则：
1. 只基于参考资料中的信息来回答，不要编造或推测
2. 如果参考资料中没有足够的信息来回答问题，请明确说明"根据现有资料，我无法回答这个问题"
3. 在回答中适当引用来源（如"根据[文档名]..."）
4. 保持回答清晰、准确、有条理"""

RAG_USER_PROMPT_TEMPLATE = """参考资料：
{context}

---

用户问题：{question}

请基于以上参考资料回答问题。"""


class RAGChain:
    """
    RAG 主流程

    使用方式：
        rag = RAGChain()
        rag.build_index()                      # 构建索引（离线，只需一次）
        answer = rag.query("什么是 Transformer？") # 查询（在线，可以多次）
    """

    def __init__(
        self,
        # Embedding 配置
        embedder_type: str = "api",  # "api" 或 "local"
        # 向量库配置
        store_type: str = "simple",  # "simple" 或 "chroma"
        # 检索方式
        retriever_type: str = "dense",  # "dense"（纯向量）或 "hybrid"（向量 + BM25）
    ):
        # ===== 初始化 Embedder =====
        if embedder_type == "api":
            print("🔧 初始化 API Embedder...")
            self.embedder = APIEmbedder(
                base_url=config.embedding_base_url,
                api_key=config.embedding_api_key or config.llm_api_key,
                model=config.embedding_model,
            )
        elif embedder_type == "local":
            print("🔧 初始化本地 Embedder...")
            self.embedder = LocalEmbedder()
        else:
            raise ValueError(f"未知的 embedder_type: {embedder_type}")

        # ===== 初始化 Vector Store =====
        if store_type == "simple":
            print("🔧 初始化 SimpleVectorStore...")
            self.store = SimpleVectorStore()
        elif store_type == "chroma":
            print("🔧 初始化 ChromaVectorStore...")
            self.store = ChromaVectorStore(persist_dir=config.db_dir)
        else:
            raise ValueError(f"未知的 store_type: {store_type}")

        # ===== 初始化 LLM 客户端 =====
        print("🔧 初始化 LLM 客户端...")
        self.llm_client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.llm_model = config.llm_model

        # ===== 初始化检索器 =====
        # dense：查询时直接用向量库检索
        # hybrid：在向量库之上再挂一个 BM25，查询时两路融合（RRF）
        self.retriever_type = retriever_type
        if retriever_type == "hybrid":
            print("🔧 初始化 HybridRetriever (Dense + BM25)...")
            self.hybrid = HybridRetriever(self.embedder, self.store)
        elif retriever_type == "dense":
            self.hybrid = None
        else:
            raise ValueError(f"未知的 retriever_type: {retriever_type}")

        print("✅ RAG Chain 初始化完成\n")

    # ========== 离线阶段：构建索引 ==========

    def load_index(self, index_file: str | Path | None = None) -> bool:
        """
        从磁盘加载 SimpleVectorStore 索引

        Args:
            index_file: 索引文件路径，默认使用 config.simple_index_file

        Returns:
            是否成功加载到非空索引
        """
        index_file = Path(index_file or config.simple_index_file)

        if not hasattr(self.store, "load"):
            print("⚠️ 当前向量库不支持手动加载索引。")
            return False

        if not index_file.exists():
            print(f"📭 未找到本地索引文件: {index_file}")
            return False

        try:
            self.store.load(str(index_file))
        except Exception as e:
            print(f"⚠️ 本地索引加载失败，将重新构建: {e}")
            return False

        if len(self.store) == 0:
            print("⚠️ 本地索引为空，将重新构建。")
            return False

        # Hybrid 模式：BM25 索引没存进磁盘缓存，用加载到的 chunks 现场重建
        loaded_chunks = getattr(self.store, "chunks", None)
        if self.hybrid is not None and loaded_chunks:
            self.hybrid.sparse_store.clear()
            self.hybrid.sparse_store.add(loaded_chunks)

        return True

    def save_index(self, index_file: str | Path | None = None):
        """
        保存 SimpleVectorStore 索引到磁盘

        Args:
            index_file: 索引文件路径，默认使用 config.simple_index_file
        """
        index_file = Path(index_file or config.simple_index_file)

        if not hasattr(self.store, "save"):
            raise TypeError("当前向量库不支持手动保存索引。")

        self.store.save(str(index_file))

    def build_index(
        self,
        docs_dir: str | Path | None = None,
        clear_existing: bool = True,
    ) -> int:
        """
        构建知识库索引

        完整流程：
        1. 加载文档
        2. 文本分块
        3. 计算 Embedding
        4. 存入向量库

        Args:
            docs_dir: 文档目录路径，默认使用 config 中的配置
            clear_existing: 入库前是否清空已有索引，避免重复追加

        Returns:
            本次构建并入库的 chunk 数量
        """
        docs_dir = docs_dir or config.docs_dir

        # Step 1: 加载文档
        print("📂 Step 1: 加载文档...")
        documents = load_documents(docs_dir)
        if not documents:
            print("⚠️ 没有找到文档！")
            return 0

        # Step 2: 文本分块
        print("\n🔪 Step 2: 文本分块...")
        chunks = chunk_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            strategy="recursive",
        )

        # Step 3: 计算 Embedding
        print("\n🧮 Step 3: 计算 Embedding...")
        texts = [chunk.content for chunk in chunks]

        # 分批计算（API 通常有批量大小限制）
        batch_size = 20
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.embedder.embed_texts(batch)
            all_vectors.extend(vectors)
            print(
                f"  ✅ 已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个 chunks"
            )

        # Step 4: 存入向量库
        print("\n💾 Step 4: 存入向量库...")
        if clear_existing and hasattr(self.store, "clear"):
            self.store.clear()
        self.store.add(chunks, all_vectors)

        # Hybrid 模式：同一批 chunk 再灌进 BM25 稀疏索引（不需要向量）
        if self.hybrid is not None:
            if clear_existing:
                self.hybrid.sparse_store.clear()
            self.hybrid.sparse_store.add(chunks)

        print(f"\n🎉 索引构建完成！共 {len(chunks)} 个 chunks 已入库")
        return len(chunks)

    # ========== 在线阶段：查询回答 ==========

    def query(
        self, question: str, top_k: int | None = None, verbose: bool = True
    ) -> dict:
        """
        对用户问题进行 RAG 查询

        完整流程：
        1. 把问题转成向量
        2. 在向量库中检索最相关的 chunks
        3. 构造包含上下文的 Prompt
        4. 调用 LLM 生成回答

        Args:
            question: 用户的问题
            top_k: 检索返回的 chunk 数量
            verbose: 是否打印中间过程

        Returns:
            {
                "answer": "LLM 生成的回答",
                "sources": [检索到的源文档信息],
                "context": "实际注入 Prompt 的上下文"
            }
        """
        top_k = top_k or config.top_k

        # Step 1 + 2: 检索（dense 走向量库；hybrid 走 Dense + BM25 + RRF）
        if verbose:
            print(f"\n🔍 查询: {question}")

        if self.hybrid is not None:
            if verbose:
                print(f"  1️⃣+2️⃣ Hybrid 检索 Top-{top_k}（Dense + BM25 融合）...")
            results = self.hybrid.search(question, top_k=top_k, verbose=verbose)
        else:
            if verbose:
                print("  1️⃣ 计算查询向量...")
            query_vector = self.embedder.embed_query(question)
            if verbose:
                print(f"  2️⃣ 检索 Top-{top_k} 相关文档...")
            results = self.store.search(query_vector, top_k=top_k)

        if not results:
            return {
                "answer": "知识库中没有找到相关信息。",
                "sources": [],
                "context": "",
            }

        # 打印检索结果
        if verbose:
            print("  📋 检索结果:")
            for i, r in enumerate(results):
                source = r["chunk"].metadata.get("source", "?")
                score = r["score"]
                preview = r["chunk"].content[:80].replace("\n", " ")
                print(f"     [{i + 1}] (相似度: {score:.4f}) [{source}] {preview}...")

        # Step 3: 构造 Prompt
        if verbose:
            print("  3️⃣ 构造 Prompt...")

        # 把检索到的 chunks 拼成上下文
        context_parts = []
        for i, r in enumerate(results):
            source = r["chunk"].metadata.get("source", "未知来源")
            context_parts.append(f"[来源: {source}]\n{r['chunk'].content}")

        context = "\n\n---\n\n".join(context_parts)

        # 构造最终的 user prompt
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        # Step 4: LLM 生成
        if verbose:
            print("  4️⃣ LLM 生成回答...")

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # RAG 场景用较低温度，保持忠实度
        )

        answer = response.choices[0].message.content

        # 整理来源信息
        sources = [
            {
                "source": r["chunk"].metadata.get("source", "?"),
                "score": r["score"],
                "content_preview": r["chunk"].content[:100],
            }
            for r in results
        ]

        if verbose:
            print(f"\n💡 回答:\n{answer}")

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
        }


# ===== 测试 =====
if __name__ == "__main__":
    # 快速测试整个 RAG 流程（切到 hybrid 演示混合检索）
    rag = RAGChain(embedder_type="api", store_type="simple", retriever_type="hybrid")

    # 构建索引
    rag.build_index()

    # 测试几个问题
    print("\n" + "=" * 60)
    rag.query("什么是自注意力机制？")

    print("\n" + "=" * 60)
    rag.query("RAG 解决了什么问题？")

    print("\n" + "=" * 60)
    rag.query("ReAct 是什么架构模式？")
