"""
Agentic RAG —— 让 Agent 自主控制检索行为

Phase 1-2 的 RAG 是一条固定管线：
    问题 → 检索 → Rerank → 生成
不管问什么都走一遍，不管检索质量好不好都直接用。

Phase 3 的 Agentic RAG 把检索变成 Agent 的「工具」：
    Agent 自己决定——
    1. 需不需要检索？（自适应检索）
    2. 该查哪个知识库？（路由选择）
    3. 检索结果质量行不行？（Self-RAG 质量评估）
    4. 不行的话怎么改？（改写查询、多跳检索）

实现方式：
    用 OpenAI function calling 实现精简的 ReAct 循环。
    定义工具 → LLM 决定调哪个 → 执行 → 结果反馈给 LLM → 循环。
    ┌─────────────────────────────────────────────────────────┐
    │                    Agent 循环                            │
    │                                                         │
    │   用户问题                                               │
    │       │                                                 │
    │       ▼                                                 │
    │   LLM 思考 ──────┐                                      │
    │       │          │                                      │
    │       │   ┌──────┴──────┐                               │
    │       │   │ function    │                               │
    │       │   │ calling     │                               │
    │       │   └──────┬──────┘                               │
    │       │          │                                      │
    │       │    ┌─────┼───────────┐                          │
    │       │    ▼     ▼           ▼                          │
    │       │  search  assess   direct                        │
    │       │  _kb     _quality  _answer                      │
    │       │    │     │           │                          │
    │       │    └─────┼───────────┘                          │
    │       │          │                                      │
    │       │          ▼                                      │
    │       │   工具结果反馈给 LLM                              │
    │       │          │                                      │
    │       ▼          ▼                                      │
    │   最终回答（text response 或 direct_answer）              │
    └─────────────────────────────────────────────────────────┘

与 Phase 2 的关系：
    底层检索仍然复用 Phase 2 的混合检索 + Reranking，
    但不再是「一条管线跑到底」，而是 Agent 按需、按次调用。
"""

import json
from pathlib import Path
from openai import OpenAI

from phase1_embedder import APIEmbedder
from phase1_vector_store import SimpleVectorStore
from phase1_dense_retriever import DenseRetriever, SearchResult
from phase2_01_sparse_retriever import BM25Retriever
from phase2_01_hybrid_retriever import HybridRetriever
from phase2_02_reranker import APIReranker
from phase1_document_loader import load_documents
from phase1_chunker import chunk_documents
from phase3_self_rag import SelfRAGAssessor
from phase3_router import KnowledgeRouter, KnowledgeBase
from config import config


# ========== Agent System Prompt ==========

AGENT_SYSTEM_PROMPT = """你是一个智能知识库问答助手，拥有检索知识库的能力。

## 你的工具
你有以下工具可以使用：
1. **search_knowledge_base**: 在知识库中检索相关信息
2. **assess_retrieval_quality**: 评估检索结果的质量，决定是否需要重新检索
3. **direct_answer**: 当你确信不需要检索就能回答时使用

## 工作流程
对于每个用户问题，按以下逻辑决策：

1. **判断是否需要检索**：
   - 如果是常识问题（如"1+1=?"、"什么是 Python？"）→ 用 direct_answer 直接回答
   - 如果需要专业/具体知识 → 调用 search_knowledge_base 检索

2. **检索后评估质量**：
   - 调用 assess_retrieval_quality 评估检索结果
   - 如果质量好 → 基于检索结果回答
   - 如果质量差 → 用建议的查询重新检索（最多重试 2 次）

3. **生成回答**：
   - 基于检索结果生成回答时，引用来源
   - 如果检索结果不够，坦诚说明

## 重要规则
- 不要在一次回复中调用超过 3 次工具
- 如果检索了两次还是质量差，就用现有结果尽力回答
- 回答要清晰、准确、有条理"""


# ========== Function Calling 工具定义 ==========

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中检索与查询相关的信息。返回最相关的文档片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询。应该是清晰、具体的问题或关键词组合。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_retrieval_quality",
            "description": "评估最近一次检索结果的质量。判断检索结果是否与用户问题相关、是否足够回答问题，并建议下一步动作（直接回答/改写重搜/放弃检索）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的原始问题",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "direct_answer",
            "description": "当问题不需要检索知识库时使用。适用于常识问题、简单计算、闲聊等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "直接回答的内容",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ========== Agentic RAG ==========


class AgenticRAG:
    """
    Agentic RAG：让 Agent 自主控制检索行为。

    使用方式：
        agentic = AgenticRAG()
        agentic.build_all_indexes()   # 构建索引（一次性）
        result = agentic.query("Transformer 注意力机制是什么？")

    对比传统 RAG：
        # 传统 RAG: 无脑检索
        rag = RAGChain(); rag.query("1+1=?")  # 也会去检索，浪费

        # Agentic RAG: 智能决策
        agentic.query("1+1=?")  # Agent 判断不需要检索，直接回答
    """

    def __init__(
        self,
        top_k: int | None = None,
        use_reranker: bool = False,
        use_router: bool = True,
        max_iterations: int = 5,
    ):
        self.top_k = top_k or config.top_k
        self.use_reranker = use_reranker
        self.use_router = use_router
        self.max_iterations = max_iterations

        # ===== LLM 客户端 =====
        print("🔧 初始化 LLM 客户端...")
        self.llm_client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.llm_model = config.llm_model

        # ===== Embedder =====
        print("🔧 初始化 Embedder...")
        self.embedder = APIEmbedder(
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key or config.llm_api_key,
            model=config.embedding_model,
        )

        # ===== Self-RAG 评估器 =====
        print("🔧 初始化 Self-RAG 评估器...")
        self.assessor = SelfRAGAssessor(self.llm_client, self.llm_model)

        # ===== Router =====
        if use_router:
            print("🔧 初始化知识库路由器...")
            self.router = KnowledgeRouter(self.llm_client, self.llm_model)
        else:
            self.router = None

        # ===== Reranker =====
        if use_reranker:
            print("🔧 初始化 Reranker...")
            self.reranker = APIReranker(
                api_key=config.reranker_api_key,
                model=config.reranker_model,
                base_url=config.reranker_base_url,
            )
        else:
            self.reranker = None

        # 知识库存储（后续通过 build 方法填充）
        self._knowledge_bases: dict[str, dict] = {}
        # 最近一次检索结果（供 assess_retrieval_quality 使用）
        self._last_results: list[SearchResult] = []

        print("✅ Agentic RAG 初始化完成\n")

    # ========== 知识库构建 ==========

    def _build_kb(
        self,
        name: str,
        description: str,
        docs_dir: str | Path,
        file_filter: list[str] | None = None,
    ) -> int:
        """
        构建一个命名知识库的索引。

        Args:
            name: 知识库名称
            description: 知识库描述（给路由器看的）
            docs_dir: 文档目录
            file_filter: 只加载这些文件名（None=全部加载）

        Returns:
            入库的 chunk 数量
        """
        print(f"\n📦 构建知识库 [{name}]: {description}")

        # 加载文档
        documents = load_documents(docs_dir)
        if file_filter:
            documents = [
                doc
                for doc in documents
                if any(
                    f in doc.metadata.get("source", "")
                    for f in file_filter
                )
            ]

        if not documents:
            print(f"  ⚠️ [{name}] 没有找到文档")
            return 0

        file_names = [doc.metadata.get("source", "?") for doc in documents]
        print(f"  📂 加载了 {len(documents)} 个文档: {file_names}")

        # 分块
        chunks = chunk_documents(
            documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            strategy="recursive",
        )

        # Embedding
        print(f"  🧮 计算 Embedding ({len(chunks)} chunks)...")
        texts = [chunk.content for chunk in chunks]
        batch_size = 20
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = self.embedder.embed_texts(batch)
            all_vectors.extend(vectors)

        # 向量库
        store = SimpleVectorStore()
        store.add(chunks, all_vectors)

        # 检索器
        dense = DenseRetriever(self.embedder, store)
        sparse = BM25Retriever()
        sparse.add(chunks)
        hybrid = HybridRetriever(dense, sparse)

        # 存储
        self._knowledge_bases[name] = {
            "store": store,
            "dense": dense,
            "sparse": sparse,
            "hybrid": hybrid,
            "description": description,
            "file_names": file_names,
            "chunks": chunks,
        }

        # 注册到 Router
        if self.router:
            self.router.add_kb(
                KnowledgeBase(
                    name=name,
                    description=description,
                    retriever=hybrid,
                    file_list=[Path(f).name for f in file_names],
                )
            )

        print(f"  ✅ [{name}] 构建完成: {len(chunks)} chunks")
        return len(chunks)

    def build_default_indexes(self, docs_dir: str | Path | None = None) -> int:
        """
        用默认分组构建知识库索引。

        将 docs/ 目录下的文件分为两个逻辑知识库：
        - tech_docs: 技术文档（AI/ML 相关）
        - general_kb: 通用知识库（其他文档）

        Returns:
            总 chunk 数量
        """
        docs_dir = docs_dir or config.docs_dir

        total = 0
        total += self._build_kb(
            name="tech_docs",
            description="AI/ML 技术文档库，包含 Transformer 架构、RAG 技术、Agent 系统等深度学习和自然语言处理领域的技术文档",
            docs_dir=docs_dir,
            file_filter=["transformer", "rag_overview", "agent"],
        )

        total += self._build_kb(
            name="general_kb",
            description="通用知识库，包含商业思维、个人成长、企业数字化等非技术领域的书籍和方案文档",
            docs_dir=docs_dir,
            file_filter=["纳瓦尔", "数字化"],
        )

        print(f"\n🎉 全部知识库构建完成，共 {total} chunks")
        return total

    def build_single_index(self, docs_dir: str | Path | None = None) -> int:
        """
        不分组，把所有文档放进一个知识库（禁用路由时使用）。

        Returns:
            总 chunk 数量
        """
        docs_dir = docs_dir or config.docs_dir
        return self._build_kb(
            name="all",
            description="综合知识库",
            docs_dir=docs_dir,
        )

    # ========== 索引持久化 ==========

    def save_indexes(self, base_dir: str | Path | None = None):
        """保存所有知识库索引到磁盘"""
        base_dir = Path(base_dir or Path(__file__).resolve().parent)
        for name, kb in self._knowledge_bases.items():
            path = base_dir / f"phase3_index_{name}.json"
            kb["store"].save(str(path))
            print(f"💾 [{name}] 索引已保存到 {path}")

    def load_indexes(self, base_dir: str | Path | None = None) -> bool:
        """
        从磁盘加载知识库索引。

        如果任何一个索引文件不存在，返回 False。

        Returns:
            是否成功加载
        """
        base_dir = Path(base_dir or Path(__file__).resolve().parent)

        # 检查默认分组的索引文件是否存在
        if self.use_router:
            names = ["tech_docs", "general_kb"]
        else:
            names = ["all"]

        for name in names:
            path = base_dir / f"phase3_index_{name}.json"
            if not path.exists():
                print(f"📭 [{name}] 索引文件不存在: {path}")
                return False

        # 全部存在，开始加载
        for name in names:
            path = base_dir / f"phase3_index_{name}.json"
            store = SimpleVectorStore()
            store.load(str(path))

            if len(store) == 0:
                print(f"⚠️ [{name}] 索引为空")
                return False

            chunks = store.chunks
            dense = DenseRetriever(self.embedder, store)
            sparse = BM25Retriever()
            sparse.add(chunks)
            hybrid = HybridRetriever(dense, sparse)

            # 从索引反推描述（简化处理）
            desc_map = {
                "tech_docs": "AI/ML 技术文档库，包含 Transformer 架构、RAG 技术、Agent 系统等技术文档",
                "general_kb": "通用知识库，包含商业思维、个人成长、企业数字化等书籍和方案文档",
                "all": "综合知识库",
            }
            desc = desc_map.get(name, name)
            file_names = list({c.metadata.get("source", "?") for c in chunks})

            self._knowledge_bases[name] = {
                "store": store,
                "dense": dense,
                "sparse": sparse,
                "hybrid": hybrid,
                "description": desc,
                "file_names": file_names,
                "chunks": chunks,
            }

            if self.router:
                self.router.add_kb(
                    KnowledgeBase(
                        name=name,
                        description=desc,
                        retriever=hybrid,
                        file_list=[Path(f).name for f in file_names],
                    )
                )

            print(f"✅ [{name}] 已加载 {len(store)} chunks")

        return True

    # ========== 工具执行 ==========

    def _exec_search(self, query: str, verbose: bool = True) -> str:
        """执行检索工具"""
        if verbose:
            print(f"\n  🔍 search_knowledge_base(query=\"{query}\")")

        candidate_k = self.top_k * 3 if self.reranker else self.top_k

        # 路由 + 检索
        if self.router and len(self._knowledge_bases) > 1:
            results, route_decision = self.router.route_and_search(
                query, top_k=candidate_k, verbose=verbose
            )
        else:
            # 无路由，直接用第一个知识库
            kb_name = list(self._knowledge_bases.keys())[0]
            kb = self._knowledge_bases[kb_name]
            results = kb["hybrid"].search(query, top_k=candidate_k)

        # Reranking（如果启用）
        if self.reranker and results:
            if verbose:
                print(f"     🔄 Reranker 精排 Top-{candidate_k} → Top-{self.top_k}...")
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_n=self.top_k,
                verbose=False,
            )
        else:
            results = results[: self.top_k]

        # 保存最近一次检索结果（供 assess 使用）
        self._last_results = results

        if not results:
            return "未找到相关信息。"

        # 格式化结果
        parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("source", "?")
            kb_name = r.chunk.metadata.get("knowledge_base", "default")
            score = r.score
            parts.append(
                f"[{i}] (相关度: {score:.4f}) [来源: {source}] [知识库: {kb_name}]\n"
                f"{r.chunk.content}"
            )

        result_text = "\n\n---\n\n".join(parts)

        if verbose:
            print(f"     📋 检索到 {len(results)} 条结果:")
            for i, r in enumerate(results, 1):
                source = r.chunk.metadata.get("source", "?")
                preview = r.chunk.content[:80].replace("\n", " ")
                print(f"        [{i}] ({r.score:.4f}) [{source}] {preview}...")

        return result_text

    def _exec_assess(self, question: str, verbose: bool = True) -> str:
        """执行检索质量评估工具"""
        if verbose:
            print(f"\n  📊 assess_retrieval_quality(question=\"{question}\")")

        if not self._last_results:
            return json.dumps(
                {
                    "relevance": "irrelevant",
                    "sufficiency": "insufficient",
                    "action": "refine",
                    "reason": "没有检索结果可供评估",
                    "suggested_query": question,
                },
                ensure_ascii=False,
            )

        assessment = self.assessor.assess(
            question=question,
            results=self._last_results,
            verbose=verbose,
        )

        return json.dumps(
            {
                "relevance": assessment.relevance,
                "sufficiency": assessment.sufficiency,
                "action": assessment.action,
                "reason": assessment.reason,
                "suggested_query": assessment.suggested_query,
            },
            ensure_ascii=False,
        )

    def _exec_direct_answer(self, answer: str, verbose: bool = True) -> str:
        """执行直接回答工具"""
        if verbose:
            print(f"\n  💡 direct_answer: Agent 判断不需要检索，直接回答")
        return answer

    # ========== Agent 主循环 ==========

    def query(self, question: str, verbose: bool = True) -> dict:
        """
        Agentic RAG 查询。

        Agent 循环：
        1. 把用户问题和工具定义发给 LLM
        2. LLM 决定调用哪个工具（或直接回答）
        3. 执行工具，把结果反馈给 LLM
        4. 重复，直到 LLM 给出最终文本回答

        Args:
            question: 用户问题
            verbose: 是否打印思考过程

        Returns:
            {
                "answer": "最终回答",
                "steps": [每一步的工具调用记录],
                "iterations": 循环了几轮,
                "used_retrieval": 是否使用了检索,
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤖 Agentic RAG 查询: {question}")
            print(f"{'='*60}")

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        steps = []
        used_retrieval = False

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- 第 {iteration + 1} 轮 ---")

            # 调用 LLM（带 function calling）
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=TOOLS,
                temperature=0.3,
            )

            msg = response.choices[0].message

            # 情况 1：LLM 直接给出文本回答（没有调用工具）
            if not msg.tool_calls:
                answer = msg.content or ""
                if verbose:
                    print(f"\n💡 Agent 最终回答:\n{answer}")

                return {
                    "answer": answer,
                    "steps": steps,
                    "iterations": iteration + 1,
                    "used_retrieval": used_retrieval,
                }

            # 情况 2：LLM 调用了工具
            # 先把 assistant message 加到对话历史
            messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                if verbose:
                    print(f"\n  🔧 Agent 调用工具: {fn_name}")

                # 执行工具
                if fn_name == "search_knowledge_base":
                    result = self._exec_search(fn_args.get("query", question), verbose)
                    used_retrieval = True
                elif fn_name == "assess_retrieval_quality":
                    result = self._exec_assess(fn_args.get("question", question), verbose)
                elif fn_name == "direct_answer":
                    direct = fn_args.get("answer", "")
                    result = self._exec_direct_answer(direct, verbose)
                    # direct_answer 是最终回答，记录后直接返回
                    steps.append({
                        "tool": fn_name,
                        "args": fn_args,
                        "result_preview": direct[:200],
                    })
                    return {
                        "answer": direct,
                        "steps": steps,
                        "iterations": iteration + 1,
                        "used_retrieval": used_retrieval,
                    }
                else:
                    result = f"未知工具: {fn_name}"

                steps.append({
                    "tool": fn_name,
                    "args": fn_args,
                    "result_preview": result[:200] if result else "",
                })

                # 把工具结果反馈给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # 超过最大轮数
        if verbose:
            print(f"\n⚠️ 达到最大轮数 ({self.max_iterations})，强制输出")

        # 最后一次调用 LLM 强制生成回答（不给工具）
        messages.append({
            "role": "user",
            "content": "请根据目前收集到的信息，直接给出最终回答。",
        })
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or "无法生成回答。"

        if verbose:
            print(f"\n💡 Agent 最终回答:\n{answer}")

        return {
            "answer": answer,
            "steps": steps,
            "iterations": self.max_iterations,
            "used_retrieval": used_retrieval,
        }


# ========== 对比模式：传统 RAG vs Agentic RAG ==========

def compare_with_naive(
    agentic: AgenticRAG,
    question: str,
) -> dict:
    """
    对比传统 RAG（always-retrieve）和 Agentic RAG 的效果。

    传统 RAG：不管问什么都检索。
    Agentic RAG：Agent 自主决策。

    Returns:
        {
            "question": ...,
            "naive": {"answer": ..., ...},
            "agentic": {"answer": ..., "steps": ..., ...},
        }
    """
    print(f"\n{'='*60}")
    print(f"⚔️  对比模式: {question}")
    print(f"{'='*60}")

    # --- 传统 RAG：无脑检索 + 生成 ---
    print(f"\n{'─'*40}")
    print("📌 传统 RAG（always-retrieve）")
    print(f"{'─'*40}")

    # 用第一个知识库的 hybrid retriever 做简单检索
    kb_name = list(agentic._knowledge_bases.keys())[0]
    kb = agentic._knowledge_bases[kb_name]
    results = kb["hybrid"].search(question, top_k=agentic.top_k)

    if results:
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("source", "?")
            context_parts.append(f"[来源: {source}]\n{r.chunk.content}")
        context = "\n\n---\n\n".join(context_parts)

        print(f"  📋 检索到 {len(results)} 条结果:")
        for i, r in enumerate(results, 1):
            source = r.chunk.metadata.get("source", "?")
            preview = r.chunk.content[:80].replace("\n", " ")
            print(f"     [{i}] ({r.score:.4f}) [{source}] {preview}...")

        naive_prompt = f"参考资料：\n{context}\n\n---\n\n用户问题：{question}\n\n请基于以上参考资料回答问题。"
    else:
        naive_prompt = f"用户问题：{question}\n\n（没有找到相关参考资料，请尽力回答）"

    response = agentic.llm_client.chat.completions.create(
        model=agentic.llm_model,
        messages=[
            {"role": "system", "content": "你是一个知识库问答助手。基于参考资料回答问题。"},
            {"role": "user", "content": naive_prompt},
        ],
        temperature=0.3,
    )
    naive_answer = response.choices[0].message.content

    print(f"\n  💡 传统 RAG 回答:\n{naive_answer}")

    # --- Agentic RAG ---
    print(f"\n{'─'*40}")
    print("🤖 Agentic RAG（智能决策）")
    print(f"{'─'*40}")

    agentic_result = agentic.query(question, verbose=True)

    # --- 对比摘要 ---
    print(f"\n{'─'*40}")
    print("📊 对比摘要")
    print(f"{'─'*40}")
    print(f"  传统 RAG:  始终检索（{len(results)} 条结果）→ 直接回答")
    print(f"  Agentic:   {agentic_result['iterations']} 轮迭代, "
          f"{'使用' if agentic_result['used_retrieval'] else '未使用'}检索, "
          f"{len(agentic_result['steps'])} 次工具调用")

    return {
        "question": question,
        "naive": {"answer": naive_answer, "num_results": len(results)},
        "agentic": agentic_result,
    }


# ===== 快速测试 =====
if __name__ == "__main__":
    print("🚀 Agentic RAG 快速测试\n")

    agentic = AgenticRAG(use_router=True, use_reranker=False)

    # 构建或加载索引
    if not agentic.load_indexes():
        agentic.build_default_indexes()
        agentic.save_indexes()

    # 测试 1：常识问题（应该直接回答，不检索）
    print("\n" + "=" * 60)
    agentic.query("1 + 1 等于几？")

    # 测试 2：知识库问题（应该检索）
    print("\n" + "=" * 60)
    agentic.query("Transformer 的自注意力机制是怎么工作的？")

    # 测试 3：跨领域问题（测试路由）
    print("\n" + "=" * 60)
    agentic.query("纳瓦尔对财富的看法是什么？")

    # 测试 4：对比模式
    compare_with_naive(agentic, "RAG 解决了什么问题？")
