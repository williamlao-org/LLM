"""
Phase 5 — Hybrid GraphRAG（向量检索 + 图检索融合）

职责：把向量 RAG 和 GraphRAG 融合成一个统一系统

为什么需要 Hybrid？
  向量 RAG 擅长语义匹配：  "如何配置 BGE-M3？"
  Local Search 擅长关系追溯："A 和 B 有什么关系？"
  Global Search 擅长全局概览："这些文档的主要趋势是什么？"

  没有银弹——每种检索方式都有盲区。
  Hybrid 的做法：用 Router 判断问题类型，分发到最合适的检索路径。

架构：
  用户问题
      │
      ▼
  Router（LLM 判断问题类型）
      │
      ├─ "vector"  → 向量检索（Phase 1 的 RAGChain）
      ├─ "local"   → Local Search（实体追溯）
      ├─ "global"  → Global Search（社区摘要 Map-Reduce）
      └─ "hybrid"  → 向量 + Local 双路检索，合并上下文
      │
      ▼
  LLM 生成回答

复用的已有模块：
  - config.py         → DeepSeek API 配置
  - phase3_router.py  → Router 设计思路
  - rag_chain.py      → 向量 RAG 检索链
"""

import json
from pathlib import Path

from openai import OpenAI
from config import config
from phase5_knowledge_graph import KnowledgeGraph
from phase5_community import Community, CommunityDetector
from phase5_local_search import LocalSearch
from phase5_global_search import GlobalSearch


# ========== Router Prompt ==========

_ROUTE_PROMPT = """分析以下用户问题，判断最佳检索策略：

1. "vector" — 事实性问题，找最相关的文本片段即可
   例：如何配置某个参数？某个概念是什么意思？操作步骤是什么？

2. "local" — 关系性问题，需要追溯实体之间的联系
   例：A 和 B 有什么关系？谁发明了 X？X 用了什么技术？X 有哪些组件？

3. "global" — 全局性问题，需要纵览全局
   例：主要趋势是什么？有哪些共同特点？总结一下所有...

4. "hybrid" — 不确定，或问题同时涉及具体事实和关系
   例：比较 A 和 B 的优劣（需要各自事实 + 相互关系）

用户问题：{query}

只返回一个词：vector / local / global / hybrid"""


# ========== 生成回答 Prompt ==========

_HYBRID_ANSWER_PROMPT = """你是一个知识库问答助手。以下是从两种来源检索到的信息：

## 向量检索结果（文本片段）
{vector_context}

## 知识图谱检索结果（结构化信息）
{graph_context}

---

用户问题：{query}

请综合以上两种来源的信息来回答问题。优先使用最直接相关的信息。
如果两种来源有互补信息，请整合在一起。
如果信息不足，请明确说明。"""


# ========== Hybrid GraphRAG ==========


class HybridGraphRAG:
    """
    Hybrid GraphRAG：融合向量检索和图检索的完整 RAG 系统

    这是 Phase 5 的终极产出——把 Phase 1 的向量 RAG 和 Phase 5 的 GraphRAG
    融合成一个统一入口。

    使用方式：
        hybrid = HybridGraphRAG()
        hybrid.load_graph_index()   # 加载知识图谱和社区
        answer = hybrid.query("Transformer 的核心创新是什么？")
    """

    def __init__(
        self,
        llm_client: OpenAI | None = None,
        model: str | None = None,
        kg_path: str | Path | None = None,
        communities_path: str | Path | None = None,
    ):
        """
        Args:
            llm_client: LLM 客户端
            model: 模型名称
            kg_path: 知识图谱 JSON 路径
            communities_path: 社区 JSON 路径
        """
        self.client = llm_client or OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        self.model = model or config.llm_model

        # 默认路径
        base_dir = Path(__file__).parent
        self.kg_path = Path(kg_path or base_dir / "phase5_knowledge_graph.json")
        self.communities_path = Path(
            communities_path or base_dir / "phase5_communities.json"
        )

        # 延迟初始化的组件
        self.kg: KnowledgeGraph | None = None
        self.communities: list[Community] | None = None
        self.local_search: LocalSearch | None = None
        self.global_search: GlobalSearch | None = None
        self.vector_rag = None  # 可选：Phase 1 的 RAGChain

    # ========== 初始化 ==========

    def load_graph_index(self, verbose: bool = True):
        """加载知识图谱索引（知识图谱 + 社区）"""
        if verbose:
            print("📂 加载 GraphRAG 索引...")

        # 加载知识图谱
        if self.kg_path.exists():
            self.kg = KnowledgeGraph.load_from_json(self.kg_path)
        else:
            print(f"  ⚠️  知识图谱文件不存在: {self.kg_path}")
            print("  请先运行: uv run python phase5_knowledge_graph.py")
            return

        # 加载社区
        if self.communities_path.exists():
            self.communities = CommunityDetector.load_communities(self.communities_path)
        else:
            print(f"  ⚠️  社区文件不存在: {self.communities_path}")
            print("  请先运行: uv run python phase5_community.py")
            return

        # 初始化搜索器
        self.local_search = LocalSearch(
            self.kg, self.communities,
            llm_client=self.client, model=self.model,
        )
        self.global_search = GlobalSearch(
            self.communities,
            llm_client=self.client, model=self.model,
        )

        if verbose:
            print("✅ GraphRAG 索引加载完成")

    def load_vector_index(self, verbose: bool = True):
        """
        可选：加载向量索引（Phase 1 的 RAGChain）

        如果不加载，hybrid 模式退化为只用 graph 检索。
        """
        try:
            from rag_chain import RAGChain
            self.vector_rag = RAGChain(
                store_type="simple",
                embedder_type="api",
                build_on_init=False,
            )
            # 尝试加载已有索引
            self.vector_rag.load_index()
            if verbose:
                print("✅ 向量索引加载完成")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  向量索引加载失败: {e}")
                print("  Hybrid 模式将只使用 Graph 检索")

    # ========== 路由 ==========

    def _route(self, query: str, verbose: bool = True) -> str:
        """
        判断问题类型，路由到最佳检索策略。

        返回: "vector" / "local" / "global" / "hybrid"
        """
        prompt = _ROUTE_PROMPT.format(query=query)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        route = response.choices[0].message.content.strip().lower()

        # 确保返回有效值
        valid_routes = {"vector", "local", "global", "hybrid"}
        if route not in valid_routes:
            route = "hybrid"  # 默认走 hybrid

        # 如果没有向量索引，vector/hybrid 退化
        if route == "vector" and self.vector_rag is None:
            route = "local"  # 退化为 local
        if route == "hybrid" and self.vector_rag is None:
            route = "local"  # 退化为 local

        if verbose:
            route_emoji = {
                "vector": "📄", "local": "🔗",
                "global": "🌐", "hybrid": "🔀",
            }
            print(f"  {route_emoji.get(route, '❓')} Router → {route}")

        return route

    # ========== 查询 ==========

    def query(
        self,
        question: str,
        force_route: str | None = None,
        verbose: bool = True,
    ) -> str:
        """
        统一查询入口。

        Args:
            question: 用户问题
            force_route: 强制指定路由（跳过 Router），用于 A/B 对比
            verbose: 是否打印过程

        Returns:
            LLM 生成的回答
        """
        if not self.kg:
            return "❌ 请先调用 load_graph_index() 加载索引"

        if verbose:
            print(f"\n{'='*60}")
            print(f"❓ 问题: {question}")

        # 路由
        route = force_route or self._route(question, verbose)

        # 分发到对应的检索策略
        if route == "vector":
            return self._vector_search(question, verbose)
        elif route == "local":
            return self.local_search.search(question, verbose)
        elif route == "global":
            return self.global_search.search(question, verbose)
        elif route == "hybrid":
            return self._hybrid_search(question, verbose)
        else:
            return self.local_search.search(question, verbose)

    def _vector_search(self, query: str, verbose: bool) -> str:
        """向量检索路径"""
        if self.vector_rag is None:
            if verbose:
                print("  ⚠️  向量索引未加载，退回 Local Search")
            return self.local_search.search(query, verbose)

        if verbose:
            print("  📄 向量检索...")

        result = self.vector_rag.query(query)
        return result.get("answer", "向量检索未返回结果")

    def _hybrid_search(self, query: str, verbose: bool) -> str:
        """
        Hybrid 检索：同时走向量和图两条路，合并上下文

        这是最强的检索模式——向量提供精确的文本片段，图提供结构化关系，互补。
        """
        if verbose:
            print("  🔀 Hybrid: 向量 + 图 双路检索")

        # 图检索上下文
        graph_context_obj = self.local_search.search_context_only(query, verbose)
        graph_context = graph_context_obj.formatted_context

        # 向量检索上下文
        vector_context = "（向量索引未加载）"
        if self.vector_rag:
            try:
                # 只做检索，不生成回答
                results = self.vector_rag.retrieve(query)
                vector_context = "\n\n".join(
                    f"[{r.get('source', '?')}] {r.get('content', '')}"
                    for r in results
                )
            except Exception:
                vector_context = "（向量检索失败）"

        # 合并上下文，让 LLM 综合回答
        prompt = _HYBRID_ANSWER_PROMPT.format(
            vector_context=vector_context,
            graph_context=graph_context,
            query=query,
        )

        if verbose:
            print("  📝 LLM 综合两路上下文生成回答...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    # ========== A/B 对比 ==========

    def compare(
        self,
        question: str,
        routes: list[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, str]:
        """
        A/B 对比：同一个问题分别走不同路径，打印对比结果。

        Args:
            question: 用户问题
            routes: 要对比的路由列表，默认 ["local", "global"]

        Returns:
            {route: answer} 字典
        """
        if routes is None:
            routes = ["local", "global"]

        results = {}
        for route in routes:
            if verbose:
                print(f"\n{'─'*60}")
                print(f"📊 [{route.upper()}] 路径:")
            answer = self.query(question, force_route=route, verbose=verbose)
            results[route] = answer
            if verbose:
                print(f"\n📝 [{route.upper()}] 回答:")
                print(answer[:300] + ("..." if len(answer) > 300 else ""))

        return results


# ========== 独立运行 Demo ==========


def demo():
    """
    Hybrid GraphRAG demo：展示路由 + 多路检索

    用法: uv run python phase5_hybrid_graphrag.py
    """
    print("=" * 60)
    print("Phase 5 — Hybrid GraphRAG Demo")
    print("=" * 60)

    # 初始化
    hybrid = HybridGraphRAG()
    hybrid.load_graph_index()

    # 测试三种类型的问题
    test_cases = [
        ("关系问题（→ Local）", "Transformer 和 BERT 有什么关系？"),
        ("全局问题（→ Global）", "这些文档的核心技术主题有哪些？"),
        ("具体问题（→ Local/Vector）", "ReAct 架构的工作流程是什么？"),
    ]

    for label, query in test_cases:
        print(f"\n{'='*60}")
        print(f"🏷️  {label}")
        answer = hybrid.query(query)
        print(f"\n📝 回答:\n{answer}")

    # A/B 对比：同一问题走不同路径
    print(f"\n{'='*60}")
    print("📊 A/B 对比: 同一问题走 Local vs Global")
    hybrid.compare(
        "AI Agent 有哪些核心能力？",
        routes=["local", "global"],
    )

    print(f"\n{'='*60}")
    print("✅ Demo 完成")


if __name__ == "__main__":
    demo()
