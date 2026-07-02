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
from phase3_self_rag import RetrievalAssessment, SelfRAGAssessor
from phase3_router import KnowledgeRouter, KnowledgeBase
from phase3_query_decomposer import QueryDecomposer, QueryStep
from phase3_hop_assessor import HopAssessment, HopAssessor
from phase4_working_memory import WorkingMemory
from config import config


# ========== Agent System Prompt ==========

AGENT_SYSTEM_PROMPT = """你是一个智能知识库问答助手，拥有检索知识库的能力。

## 你的工具
你有以下工具可以使用：
- **search_knowledge_base**: 在知识库中检索相关信息
{multi_hop_tool}
- **direct_answer**: 当你确信不需要检索就能回答时使用

## 工作流程
对于每个用户问题，按以下逻辑决策：

1. **判断是否需要检索**：
   - 如果是常识问题（如"1+1=?"、"什么是 Python？"）→ 用 direct_answer 直接回答
{multi_hop_rule}
   - 如果是一般的简单知识查询 → 调用 search_knowledge_base 检索

2. **检索后评估质量**：
   - 系统会自动评估检索结果，不需要你额外调用评估工具
   - 质量不足时，系统会自动改写查询并有限重试
   - 你会在检索工具结果中同时收到最终证据和质量评估

3. **生成回答**：
   - 基于检索结果生成回答时，引用来源
   - 如果检索结果不够，坦诚说明

## 重要规则
- 每轮只选择一个最合适的工具
- 系统最多允许 {max_tool_calls} 次 Agent 工具调用、{max_retrieval_retries} 次额外检索重试
- 达到重试上限后，用现有结果尽力回答并明确说明不足
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
            "name": "multi_hop_search",
            "description": "处理需要多个关联步骤的复杂问题。先生成粗计划，再逐跳检索和评估；证据不足时会局部重试或重规划剩余步骤。",
            "parameters": {
                "type": "object",
                "properties": {
                    "complex_query": {
                        "type": "string",
                        "description": "用户的复杂查询"
                    }
                },
                "required": ["complex_query"]
            }
        }
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
        use_multi_hop: bool = True,
        max_iterations: int = 5,
        max_tool_calls: int = 3,
        max_retrieval_retries: int = 2,
        max_hop_retries: int = 1,
        max_replans: int = 1,
        max_multi_hop_steps: int = 6,
    ):
        self.top_k = config.top_k if top_k is None else top_k
        self.use_reranker = use_reranker
        self.use_router = use_router
        self.use_multi_hop = use_multi_hop
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_retrieval_retries = max_retrieval_retries
        self.max_hop_retries = max_hop_retries
        self.max_replans = max_replans
        self.max_multi_hop_steps = max_multi_hop_steps

        if self.top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations 必须大于 0")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls 必须大于 0")
        if self.max_retrieval_retries < 0:
            raise ValueError("max_retrieval_retries 不能小于 0")
        if self.max_hop_retries < 0:
            raise ValueError("max_hop_retries 不能小于 0")
        if self.max_replans < 0:
            raise ValueError("max_replans 不能小于 0")
        if self.max_multi_hop_steps <= 0:
            raise ValueError("max_multi_hop_steps 必须大于 0")

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

        # ===== Query Decomposer =====
        if use_multi_hop:
            print("🔧 初始化多跳查询拆分器...")
            self.decomposer = QueryDecomposer(self.llm_client, self.llm_model)
            self.hop_assessor = HopAssessor(self.llm_client, self.llm_model)
        else:
            self.decomposer = None
            self.hop_assessor = None

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

        print("✅ Agentic RAG 初始化完成\n")

    def _available_tools(self) -> list[dict]:
        """根据功能开关返回当前可用的 function-calling 工具。"""
        disabled = set()
        if not self.use_multi_hop:
            disabled.add("multi_hop_search")
        return [
            tool
            for tool in TOOLS
            if tool["function"]["name"] not in disabled
        ]

    def _system_prompt(self) -> str:
        """让 Prompt 与功能开关和代码级限制保持一致。"""
        if self.use_multi_hop:
            multi_hop_tool = (
                "- **multi_hop_search**: 面对需要多步查证或实体关联的复杂问题时使用"
            )
            multi_hop_rule = (
                '   - 如果问题复杂、涉及关联跳跃（如"某某的作者的导师是谁"）'
                " → 调用 multi_hop_search"
            )
        else:
            multi_hop_tool = ""
            multi_hop_rule = ""

        return AGENT_SYSTEM_PROMPT.format(
            multi_hop_tool=multi_hop_tool,
            multi_hop_rule=multi_hop_rule,
            max_tool_calls=self.max_tool_calls,
            max_retrieval_retries=self.max_retrieval_retries,
        )

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

    def _retrieve_results(
        self,
        query: str,
        verbose: bool = True,
        use_routing: bool = True,
    ) -> list[SearchResult]:
        """执行底层检索并返回结构化结果。"""
        if not self._knowledge_bases:
            if verbose:
                print("     ⚠️ 知识库尚未加载，无法执行检索")
            return []

        candidate_k = self.top_k * 3 if self.reranker else self.top_k

        if self.router and len(self._knowledge_bases) > 1:
            if use_routing:
                results, _ = self.router.route_and_search(
                    query, top_k=candidate_k, verbose=verbose
                )
            else:
                results = self.router.search(
                    question=query,
                    kb_names=list(self._knowledge_bases),
                    top_k=candidate_k,
                    verbose=verbose,
                )
        else:
            kb = next(iter(self._knowledge_bases.values()))
            results = kb["hybrid"].search(query, top_k=candidate_k)

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

        return results

    @staticmethod
    def _format_results(results: list[SearchResult]) -> str:
        """把结构化检索结果格式化为给 LLM 的上下文。"""
        if not results:
            return "未找到相关信息。"

        parts = []
        for i, result in enumerate(results, 1):
            source = result.chunk.metadata.get("source", "?")
            kb_name = result.chunk.metadata.get("knowledge_base", "default")
            parts.append(
                f"[{i}] (相关度: {result.score:.4f}) "
                f"[来源: {source}] [知识库: {kb_name}]\n"
                f"{result.chunk.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _exec_search(
        self,
        query: str,
        verbose: bool = True,
        use_routing: bool = True,
    ) -> tuple[str, list[SearchResult]]:
        """执行检索工具，返回格式化上下文和本次查询的结果。"""
        if verbose:
            print(f"\n  🔍 search_knowledge_base(query=\"{query}\")")

        results = self._retrieve_results(
            query,
            verbose=verbose,
            use_routing=use_routing,
        )
        result_text = self._format_results(results)

        if verbose:
            print(f"     📋 检索到 {len(results)} 条结果:")
            for i, r in enumerate(results, 1):
                source = r.chunk.metadata.get("source", "?")
                preview = r.chunk.content[:80].replace("\n", " ")
                print(f"        [{i}] ({r.score:.4f}) [{source}] {preview}...")

        return result_text, results

    def _exec_assess(
        self,
        question: str,
        results: list[SearchResult],
        verbose: bool = True,
    ) -> tuple[str, RetrievalAssessment]:
        """评估显式传入的本次检索结果，避免跨查询共享状态。"""
        if verbose:
            print(f"\n  📊 assess_retrieval_quality(question=\"{question}\")")

        assessment = self.assessor.assess(
            question=question,
            results=results,
            verbose=verbose,
        )

        payload = json.dumps(
            {
                "relevance": assessment.relevance,
                "sufficiency": assessment.sufficiency,
                "action": assessment.action,
                "reason": assessment.reason,
                "suggested_query": assessment.suggested_query,
            },
            ensure_ascii=False,
        )
        return payload, assessment

    def _exec_direct_answer(self, answer: str, verbose: bool = True) -> str:
        """执行直接回答工具"""
        if verbose:
            print(f"\n  💡 direct_answer: Agent 判断不需要检索，直接回答")
        return answer

    @staticmethod
    def _merge_unique_results(
        target: list[SearchResult],
        seen: set[tuple[str, str]],
        incoming: list[SearchResult],
    ) -> None:
        """按来源和内容去重合并证据。"""
        for result in incoming:
            key = (
                str(result.chunk.metadata.get("source", "?")),
                result.chunk.content,
            )
            if key not in seen:
                seen.add(key)
                target.append(result)

    def _resolve_multi_hop_query(
        self,
        step: QueryStep,
        dependency_records: list[dict],
    ) -> str:
        """只使用显式依赖步骤的结构化事实和实体解析当前查询。"""
        if not dependency_records:
            return step.query

        payload = {
            "query": step.query,
            "dependencies": [
                {
                    "step_id": record["step_id"],
                    "facts": record.get("facts", []),
                    "entities": record.get("entities", {}),
                    "fallback_evidence": record.get("fallback_evidence", ""),
                }
                for record in dependency_records
            ],
        }
        prompt = (
            "请仅根据下面 JSON 中声明的依赖事实和实体，将 query 里的代词或占位符"
            "替换为具体实体。直接输出可检索的纯文本查询，不要输出解释或思维过程。\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or step.query

    @staticmethod
    def _completed_step_payload(completed: dict[int, dict]) -> list[dict]:
        return [
            {
                "step_id": record["step_id"],
                "query": record["query"],
                "facts": record.get("facts", []),
                "entities": record.get("entities", {}),
            }
            for record in completed.values()
        ]

    def _exec_multi_hop_search(
        self,
        complex_query: str,
        verbose: bool = True,
        step_log: list[dict] | None = None,
    ) -> tuple[str, list[SearchResult]]:
        """执行 Planner + Adaptive Executor 多跳检索。"""
        if verbose:
            print(f"\n  🕵️‍♂️ multi_hop_search(complex_query=\"{complex_query}\")")

        if not getattr(self, "decomposer", None) or not getattr(
            self, "hop_assessor", None
        ):
            return "错误：未开启自适应多跳检索功能。", []

        trace = step_log if step_log is not None else []
        plan = self.decomposer.decompose(complex_query, verbose)
        pending_steps = list(plan.steps)
        completed: dict[int, dict] = {}
        aggregated_results: list[SearchResult] = []
        seen_chunks: set[tuple[str, str]] = set()
        history_context: list[str] = []
        known_step_ids = {step.step_id for step in pending_steps}
        executed_steps = 0
        replan_count = 0
        stop_reason = ""

        while pending_steps and executed_steps < self.max_multi_hop_steps:
            step = pending_steps.pop(0)
            executed_steps += 1

            if verbose:
                print(f"\n     ➡️ 执行 Step {step.step_id}: {step.query}")

            missing_dependencies = [
                dependency_id
                for dependency_id in (step.depends_on or [])
                if dependency_id not in completed
            ]
            if missing_dependencies:
                stop_reason = f"Step {step.step_id} 缺少依赖 {missing_dependencies}"
                trace.append({
                    "tool": "assess_multi_hop_step",
                    "args": {
                        "step_id": step.step_id,
                        "status": "invalid_dependency",
                    },
                    "result_preview": stop_reason,
                })
                if replan_count >= self.max_replans:
                    break

                next_step_id = max(known_step_ids | set(completed) | {0}) + 1
                try:
                    new_plan = self.decomposer.replan(
                        original_question=complex_query,
                        completed_steps=self._completed_step_payload(completed),
                        failed_step=step,
                        remaining_steps=pending_steps,
                        next_step_id=next_step_id,
                        failed_context={
                            "error": stop_reason,
                            "missing_dependencies": missing_dependencies,
                        },
                        verbose=verbose,
                    )
                except Exception as error:
                    stop_reason += f"；重规划失败: {error}"
                    trace.append({
                        "tool": "replan_multi_hop",
                        "args": {
                            "failed_step_id": step.step_id,
                            "status": "failed",
                        },
                        "result_preview": stop_reason[:200],
                    })
                    break

                pending_steps = list(new_plan.steps)
                known_step_ids.update(item.step_id for item in pending_steps)
                replan_count += 1
                stop_reason = ""
                trace.append({
                    "tool": "replan_multi_hop",
                    "args": {
                        "failed_step_id": step.step_id,
                        "status": "success",
                        "replan_count": replan_count,
                        "replacement_steps": [
                            item.model_dump() for item in pending_steps
                        ],
                    },
                    "result_preview": f"替换为 {len(pending_steps)} 个新步骤",
                })
                continue

            dependency_records = [
                completed[dependency_id]
                for dependency_id in (step.depends_on or [])
            ]
            dependency_facts = [
                fact
                for record in dependency_records
                for fact in record.get("facts", [])
            ]
            accumulated_facts = [
                fact
                for record in completed.values()
                for fact in record.get("facts", [])
            ]

            try:
                actual_query = self._resolve_multi_hop_query(
                    step,
                    dependency_records,
                )
            except Exception as error:
                actual_query = step.query
                if verbose:
                    print(f"     ⚠️ 依赖查询解析失败，使用原查询: {error}")

            retry_count = 0
            accepted = False
            finish_early = False
            assessment_failed = False
            hop_facts: list[str] = []
            hop_entities: dict[str, str] = {}
            hop_results: list[SearchResult] = []
            hop_seen: set[tuple[str, str]] = set()
            last_context = "未找到相关信息。"
            last_assessment: HopAssessment | None = None

            while True:
                last_context, results = self._exec_search(
                    actual_query,
                    verbose=False,
                )
                self._merge_unique_results(hop_results, hop_seen, results)
                first_chunk = last_context.split("\n\n---\n\n")[0]
                trace.append({
                    "tool": "multi_hop_step",
                    "args": {
                        "step_id": step.step_id,
                        "query": actual_query,
                        "depends_on": step.depends_on or [],
                        "attempt": retry_count + 1,
                    },
                    "result_preview": first_chunk[:200],
                })

                try:
                    last_assessment = self.hop_assessor.assess(
                        original_question=complex_query,
                        step=step,
                        executed_query=actual_query,
                        dependency_facts=dependency_facts,
                        accumulated_facts=accumulated_facts,
                        results=results,
                        verbose=verbose,
                    )
                except Exception as error:
                    assessment_failed = True
                    message = f"Step {step.step_id} 评估失败: {error}"
                    trace.append({
                        "tool": "assess_multi_hop_step",
                        "args": {
                            "step_id": step.step_id,
                            "status": "assessment_error",
                        },
                        "result_preview": message[:200],
                    })
                    if results:
                        accepted = True
                        stop_reason = message
                    else:
                        stop_reason = message + "，且没有可用证据"
                    break

                hop_facts.extend(
                    fact
                    for fact in last_assessment.extracted_facts
                    if fact not in hop_facts
                )
                hop_entities.update(last_assessment.resolved_entities)
                trace.append({
                    "tool": "assess_multi_hop_step",
                    "args": {
                        "step_id": step.step_id,
                        "relevance": last_assessment.relevance,
                        "sufficiency": last_assessment.sufficiency,
                        "can_answer_question": last_assessment.can_answer_question,
                    },
                    "result_preview": json.dumps(
                        {
                            "facts": last_assessment.extracted_facts,
                            "entities": last_assessment.resolved_entities,
                            "reason": last_assessment.reason,
                        },
                        ensure_ascii=False,
                    )[:200],
                })

                if last_assessment.relevance != "irrelevant":
                    self._merge_unique_results(
                        aggregated_results,
                        seen_chunks,
                        results,
                    )

                if last_assessment.can_answer_question:
                    accepted = True
                    finish_early = True
                    break

                if (
                    last_assessment.relevance != "irrelevant"
                    and last_assessment.sufficiency == "sufficient"
                ):
                    accepted = True
                    break

                suggested_query = (
                    last_assessment.suggested_query or ""
                ).strip()
                if (
                    retry_count < self.max_hop_retries
                    and suggested_query
                    and suggested_query != actual_query
                ):
                    retry_count += 1
                    trace.append({
                        "tool": "retry_multi_hop_step",
                        "args": {
                            "step_id": step.step_id,
                            "from_query": actual_query,
                            "to_query": suggested_query,
                            "attempt": retry_count + 1,
                        },
                        "result_preview": last_assessment.reason[:200],
                    })
                    actual_query = suggested_query
                    continue
                break

            if accepted:
                if assessment_failed:
                    self._merge_unique_results(
                        aggregated_results,
                        seen_chunks,
                        hop_results,
                    )
                first_chunk = last_context.split("\n\n---\n\n")[0]
                sources = sorted({
                    str(result.chunk.metadata.get("source", "?"))
                    for result in hop_results
                })
                record = {
                    "step_id": step.step_id,
                    "query": actual_query,
                    "facts": hop_facts,
                    "entities": hop_entities,
                    "sources": sources,
                    "fallback_evidence": first_chunk if not hop_facts else "",
                }
                completed[step.step_id] = record
                facts_text = "；".join(hop_facts) or first_chunk
                history_context.append(
                    f"[Step {step.step_id} 查询: {actual_query}]\n"
                    f"确认事实: {facts_text}\n"
                    f"来源: {', '.join(sources) or '?'}\n"
                    f"证据摘录: {first_chunk}"
                )

                if finish_early:
                    stop_reason = f"Step {step.step_id} 的累计证据已足以回答原问题"
                    trace.append({
                        "tool": "finish_multi_hop",
                        "args": {"step_id": step.step_id},
                        "result_preview": stop_reason,
                    })
                    break
                continue

            if assessment_failed:
                break

            if replan_count >= self.max_replans:
                stop_reason = (
                    f"Step {step.step_id} 证据不足，且已达到重规划上限 "
                    f"({self.max_replans})"
                )
                break

            next_step_id = max(known_step_ids | set(completed) | {0}) + 1
            failed_context = {
                "query": actual_query,
                "facts": hop_facts,
                "entities": hop_entities,
                "evidence_preview": last_context[:500],
                "assessment": (
                    last_assessment.model_dump() if last_assessment else {}
                ),
            }
            try:
                new_plan = self.decomposer.replan(
                    original_question=complex_query,
                    completed_steps=self._completed_step_payload(completed),
                    failed_step=step,
                    remaining_steps=pending_steps,
                    next_step_id=next_step_id,
                    failed_context=failed_context,
                    verbose=verbose,
                )
            except Exception as error:
                stop_reason = f"Step {step.step_id} 重规划失败: {error}"
                trace.append({
                    "tool": "replan_multi_hop",
                    "args": {
                        "failed_step_id": step.step_id,
                        "status": "failed",
                    },
                    "result_preview": stop_reason[:200],
                })
                break

            pending_steps = list(new_plan.steps)
            known_step_ids.update(item.step_id for item in pending_steps)
            replan_count += 1
            trace.append({
                "tool": "replan_multi_hop",
                "args": {
                    "failed_step_id": step.step_id,
                    "status": "success",
                    "replan_count": replan_count,
                    "replacement_steps": [
                        item.model_dump() for item in pending_steps
                    ],
                },
                "result_preview": f"替换为 {len(pending_steps)} 个新步骤",
            })

        if pending_steps and executed_steps >= self.max_multi_hop_steps:
            stop_reason = (
                f"达到多跳步骤上限 ({self.max_multi_hop_steps})，"
                f"仍有 {len(pending_steps)} 个步骤未执行"
            )

        status = stop_reason or "计划中的步骤已全部完成"
        final_output = (
            "【自适应多跳检索汇总】\n\n"
            + ("\n\n".join(history_context) or "没有确认到可用事实。")
            + f"\n\n【执行状态】{status}"
        )
        return final_output, aggregated_results

    def _run_retrieval_cycle(
        self,
        question: str,
        initial_query: str,
        tool_name: str,
        steps: list[dict],
        verbose: bool,
    ) -> str:
        """执行检索 → 强制评估 → 必要时改写重试的可靠闭环。"""
        if tool_name == "multi_hop_search":
            return self._run_multi_hop_retrieval_cycle(
                question=question,
                initial_query=initial_query,
                steps=steps,
                verbose=verbose,
            )

        current_query = initial_query
        final_context = ""
        assessment_payload = ""
        assessment = None

        for attempt in range(self.max_retrieval_retries + 1):
            retrieval_step = {
                "tool": tool_name,
                "args": {"query": current_query, "attempt": attempt + 1},
                "result_preview": "",
            }
            steps.append(retrieval_step)

            final_context, results = self._exec_search(
                current_query,
                verbose=verbose,
            )
            retrieval_step["result_preview"] = final_context[:200]

            assessment_payload, assessment = self._exec_assess(
                question,
                results,
                verbose=verbose,
            )
            steps.append({
                "tool": "assess_retrieval_quality",
                "args": {
                    "question": question,
                    "query": current_query,
                    "attempt": attempt + 1,
                },
                "result_preview": assessment_payload[:200],
            })

            can_refine = (
                assessment.action == "refine"
                and bool(assessment.suggested_query)
                and attempt < self.max_retrieval_retries
            )
            if not can_refine:
                break

            current_query = assessment.suggested_query.strip()
            if verbose:
                print(f"\n  🔄 CRAG 自动改写并重试: {current_query}")

        action_hint = {
            "answer": "检索证据通过质量评估，请基于证据回答并引用来源。",
            "fallback": "知识库没有可靠证据；如使用模型自身知识，请明确说明。",
            "refine": "已达到重试上限，请仅使用现有证据并明确说明不足。",
        }.get(assessment.action if assessment else "", "请谨慎使用现有证据。")

        return (
            f"{final_context}\n\n"
            f"【自动质量评估】\n{assessment_payload}\n\n"
            f"【后续要求】{action_hint}"
        )

    def _run_multi_hop_retrieval_cycle(
        self,
        question: str,
        initial_query: str,
        steps: list[dict],
        verbose: bool,
    ) -> str:
        """运行一次自适应计划，最终 CRAG refine 仅执行定向补检。"""
        retrieval_step = {
            "tool": "multi_hop_search",
            "args": {"complex_query": initial_query, "attempt": 1},
            "result_preview": "",
        }
        steps.append(retrieval_step)
        final_context, results = self._exec_multi_hop_search(
            initial_query,
            verbose=verbose,
            step_log=steps,
        )
        retrieval_step["result_preview"] = final_context[:200]

        combined_results = list(results)
        seen = {
            (
                str(result.chunk.metadata.get("source", "?")),
                result.chunk.content,
            )
            for result in combined_results
        }
        assessment_payload, assessment = self._exec_assess(
            question,
            combined_results,
            verbose=verbose,
        )
        steps.append({
            "tool": "assess_retrieval_quality",
            "args": {
                "question": question,
                "query": initial_query,
                "attempt": 1,
                "mode": "multi_hop_final",
            },
            "result_preview": assessment_payload[:200],
        })

        supplemental_contexts: list[str] = []
        for attempt in range(self.max_retrieval_retries):
            suggested_query = (assessment.suggested_query or "").strip()
            if assessment.action != "refine" or not suggested_query:
                break

            if verbose:
                print(f"\n  🎯 CRAG 定向补检: {suggested_query}")
            context, supplemental_results = self._exec_search(
                suggested_query,
                verbose=verbose,
            )
            self._merge_unique_results(
                combined_results,
                seen,
                supplemental_results,
            )
            supplemental_contexts.append(
                f"【定向补检 {attempt + 1}: {suggested_query}】\n{context}"
            )
            steps.append({
                "tool": "search_knowledge_base",
                "args": {
                    "query": suggested_query,
                    "attempt": attempt + 1,
                    "mode": "multi_hop_targeted_refine",
                },
                "result_preview": context[:200],
            })

            assessment_payload, assessment = self._exec_assess(
                question,
                combined_results,
                verbose=verbose,
            )
            steps.append({
                "tool": "assess_retrieval_quality",
                "args": {
                    "question": question,
                    "query": suggested_query,
                    "attempt": attempt + 2,
                    "mode": "multi_hop_final",
                },
                "result_preview": assessment_payload[:200],
            })

        if supplemental_contexts:
            final_context += "\n\n" + "\n\n".join(supplemental_contexts)

        action_hint = {
            "answer": "检索证据通过质量评估，请基于证据回答并引用来源。",
            "fallback": "知识库没有可靠证据；如使用模型自身知识，请明确说明。",
            "refine": "定向补检已达上限，请仅使用现有证据并明确说明不足。",
        }.get(assessment.action, "请谨慎使用现有证据。")
        return (
            f"{final_context}\n\n"
            f"【自动质量评估】\n{assessment_payload}\n\n"
            f"【后续要求】{action_hint}"
        )

    # ========== Agent 主循环 ==========

    def _force_final_answer(
        self,
        messages: list,
        reason: str,
        verbose: bool,
    ) -> str:
        """关闭工具后，基于当前上下文强制生成最终回答。"""
        if verbose:
            print(f"\n⚠️ {reason}，强制输出最终回答")

        messages.append({
            "role": "user",
            "content": "请根据目前收集到的信息，直接给出最终回答，不要再调用工具。",
        })
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content or "无法生成回答。"

    def query(
        self,
        question: str,
        verbose: bool = True,
        memory: WorkingMemory | None = None,
    ) -> dict:
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
            memory: 可选的滑动窗口短期记忆；不传时保持单轮无状态

        Returns:
            {
                "answer": "最终回答",
                "steps": [工具调用与内部评估/重试/重规划记录],
                "iterations": 循环了几轮,
                "used_retrieval": 是否使用了检索,
            }
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤖 Agentic RAG 查询: {question}")
            print(f"{'='*60}")

        messages = [
            {"role": "system", "content": self._system_prompt()},
        ]
        if memory is not None:
            messages.extend(memory.get_context_messages())
        messages.append({"role": "user", "content": question})

        steps = []
        used_retrieval = False
        tool_call_count = 0

        def finish(answer: str, iterations: int) -> dict:
            """统一组装结果，并且只在成功得到最终答案后写入记忆。"""
            if memory is not None:
                memory.add_turn(question, answer)
            return {
                "answer": answer,
                "steps": steps,
                "iterations": iterations,
                "used_retrieval": used_retrieval,
            }

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- 第 {iteration + 1} 轮 ---")

            # 调用 LLM（带 function calling）
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=self._available_tools(),
                temperature=0.3,
            )

            msg = response.choices[0].message

            # 情况 1：LLM 直接给出文本回答（没有调用工具）
            if not msg.tool_calls:
                answer = msg.content or ""
                if verbose:
                    print(f"\n💡 Agent 最终回答:\n{answer}")

                return finish(answer, iteration + 1)

            # 情况 2：LLM 调用了工具
            # 先把 assistant message 加到对话历史
            messages.append(msg)
            sole_tool_call = len(msg.tool_calls) == 1
            force_final = False

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name

                if tool_call_count >= self.max_tool_calls:
                    result = (
                        f"工具调用上限为 {self.max_tool_calls}，"
                        "本次调用未执行。请根据已有信息回答。"
                    )
                    steps.append({
                        "tool": fn_name,
                        "args": {},
                        "result_preview": result,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                    force_final = True
                    continue

                tool_call_count += 1
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                if verbose:
                    print(f"\n  🔧 Agent 调用工具: {fn_name}")

                # 执行工具
                if fn_name == "search_knowledge_base":
                    result = self._run_retrieval_cycle(
                        question=question,
                        initial_query=fn_args.get("query") or question,
                        tool_name=fn_name,
                        steps=steps,
                        verbose=verbose,
                    )
                    used_retrieval = True
                elif fn_name == "multi_hop_search":
                    result = self._run_retrieval_cycle(
                        question=question,
                        initial_query=fn_args.get("complex_query") or question,
                        tool_name=fn_name,
                        steps=steps,
                        verbose=verbose,
                    )
                    used_retrieval = True
                elif fn_name == "direct_answer":
                    direct = str(fn_args.get("answer") or "").strip()
                    result = self._exec_direct_answer(direct, verbose)
                    steps.append({
                        "tool": fn_name,
                        "args": fn_args,
                        "result_preview": direct[:200],
                    })
                else:
                    result = f"未知工具: {fn_name}"
                    steps.append({
                        "tool": fn_name,
                        "args": fn_args,
                        "result_preview": result[:200],
                    })

                # 把工具结果反馈给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                if fn_name == "direct_answer" and sole_tool_call and result:
                    return finish(result, iteration + 1)

            if tool_call_count >= self.max_tool_calls:
                force_final = True

            if force_final:
                answer = self._force_final_answer(
                    messages,
                    reason=f"达到工具调用上限 ({self.max_tool_calls})",
                    verbose=verbose,
                )
                if verbose:
                    print(f"\n💡 Agent 最终回答:\n{answer}")
                return finish(answer, iteration + 1)

        # 超过最大轮数
        answer = self._force_final_answer(
            messages,
            reason=f"达到最大轮数 ({self.max_iterations})",
            verbose=verbose,
        )

        if verbose:
            print(f"\n💡 Agent 最终回答:\n{answer}")

        return finish(answer, self.max_iterations)


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

    # 使用与 Agentic RAG 相同的检索组件，但固定搜索全部知识库，
    # 只比较“是否自主检索”，避免知识库范围和 Reranker 配置造成偏差。
    context, results = agentic._exec_search(
        question,
        verbose=False,
        use_routing=False,
    )

    if results:
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
          f"{len(agentic_result['steps'])} 个执行步骤（含自动评估）")

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

    # 测试 3：复杂多跳问题（应该触发 multi_hop_search）
    print("\n" + "=" * 60)
    agentic.query("Transformer提出者之一Ashish Vaswani，其提出该架构时所在的机构是做什么的？")

    # 测试 3：跨领域问题（测试路由）
    print("\n" + "=" * 60)
    agentic.query("纳瓦尔对财富的看法是什么？")

    # 测试 4：对比模式
    compare_with_naive(agentic, "RAG 解决了什么问题？")
