"""
Phase 5 — 主入口

提供三种运行模式：
  1. build  — 构建索引（抽取 → 建图 → 社区检测 → 摘要）
  2. query  — 交互式查询
  3. compare — A/B 对比（同一问题走不同路径）

用法：
  uv run python phase5_main.py build     # 构建索引
  uv run python phase5_main.py query     # 交互式查询
  uv run python phase5_main.py compare   # A/B 对比
"""

import sys
from pathlib import Path

from config import config


# ========== 路径常量 ==========

BASE_DIR = Path(__file__).parent
KG_PATH = BASE_DIR / "phase5_knowledge_graph.json"
COMMUNITIES_PATH = BASE_DIR / "phase5_communities.json"


# ========== 1. 构建索引 ==========


def build_index():
    """
    完整的索引构建流程：

      文档 → 分块 → LLM 抽取实体/关系 → 合并去重建图
                                          → 社区检测 → LLM 社区摘要
    """
    from phase1_document_loader import Document
    from phase1_chunker import chunk_documents
    from phase5_entity_extractor import EntityExtractor
    from phase5_knowledge_graph import KnowledgeGraph
    from phase5_community import CommunityDetector

    print("=" * 60)
    print("Phase 5 — 构建 GraphRAG 索引")
    print("=" * 60)

    # ---- Step 1: 加载文档 ----
    print("\n📚 Step 1: 加载文档...")
    docs_dir = BASE_DIR / "docs"
    md_files = sorted(docs_dir.glob("*.md"))

    md_docs = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        md_docs.append(Document(content=text, metadata={"source": md_file.name}))
        print(f"  ✅ {md_file.name} ({len(text)} 字符)")

    chunks = chunk_documents(md_docs, chunk_size=1200, chunk_overlap=100)
    print(f"  → {len(chunks)} 个 chunk")

    # ---- Step 2: 实体/关系抽取 ----
    print("\n🔍 Step 2: LLM 实体/关系抽取...")
    extractor = EntityExtractor(gleaning_rounds=1)
    results = extractor.extract_from_chunks(chunks)

    # ---- Step 3: 构建知识图谱 ----
    print("\n🏗️  Step 3: 构建知识图谱（合并去重）...")
    kg = KnowledgeGraph()
    kg.build_from_extractions(results, use_llm_merge=True)
    kg.save_to_json(KG_PATH)

    # ---- Step 4: 社区检测 + 摘要 ----
    print("\n🔬 Step 4: 社区检测 + 摘要...")
    detector = CommunityDetector(resolution=1.0)
    communities = detector.detect_and_summarize(kg)
    detector.save_communities(communities, COMMUNITIES_PATH)

    print(f"\n{'='*60}")
    print("✅ 索引构建完成！")
    print(f"  知识图谱: {KG_PATH}")
    print(f"  社区:     {COMMUNITIES_PATH}")


# ========== 2. 交互式查询 ==========


def interactive_query():
    """
    交互式查询：输入问题，Router 自动选择最佳路径
    """
    from phase5_hybrid_graphrag import HybridGraphRAG

    print("=" * 60)
    print("Phase 5 — GraphRAG 交互式查询")
    print("=" * 60)

    # 加载索引
    hybrid = HybridGraphRAG()
    hybrid.load_graph_index()

    print("\n📝 输入问题开始查询（输入 q 退出, 输入 local/global/hybrid 切换模式）")
    print("   默认模式: auto（Router 自动判断）\n")

    force_route = None

    while True:
        try:
            user_input = input("❓ 你的问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break

        # 模式切换
        if user_input.lower() in ("auto", "local", "global", "hybrid", "vector"):
            if user_input.lower() == "auto":
                force_route = None
                print("  🔄 切换到 auto 模式（Router 自动判断）")
            else:
                force_route = user_input.lower()
                print(f"  🔄 切换到 {force_route} 模式")
            continue

        # 查询
        answer = hybrid.query(user_input, force_route=force_route)
        print(f"\n📝 回答:\n{answer}\n")

    print("\n👋 再见！")


# ========== 3. A/B 对比 ==========


def ab_compare():
    """
    A/B 对比模式：同一问题分别走不同路径，直观对比效果
    """
    from phase5_hybrid_graphrag import HybridGraphRAG

    print("=" * 60)
    print("Phase 5 — A/B 对比模式")
    print("=" * 60)

    hybrid = HybridGraphRAG()
    hybrid.load_graph_index()

    # 预设对比用例
    test_cases = [
        {
            "label": "关系问题（Local 应该更好）",
            "query": "Transformer 和 BERT 有什么关系？",
            "routes": ["local", "global"],
        },
        {
            "label": "全局问题（Global 应该更好）",
            "query": "这些文档涵盖了哪些核心技术主题？",
            "routes": ["local", "global"],
        },
        {
            "label": "具体问题",
            "query": "自注意力机制的计算过程是什么？",
            "routes": ["local", "global"],
        },
    ]

    for case in test_cases:
        print(f"\n{'='*60}")
        print(f"🏷️  {case['label']}")
        print(f"❓  {case['query']}")

        results = hybrid.compare(
            case["query"],
            routes=case["routes"],
            verbose=False,
        )

        for route, answer in results.items():
            print(f"\n📝 [{route.upper()}] 回答:")
            print(f"  {answer[:400]}{'...' if len(answer) > 400 else ''}")

    print(f"\n{'='*60}")
    print("✅ 对比完成")


# ========== 入口 ==========


def main():
    if len(sys.argv) < 2:
        print("""
Phase 5 — GraphRAG 主入口

用法：
  uv run python phase5_main.py build     # 构建索引（抽取→建图→社区→摘要）
  uv run python phase5_main.py query     # 交互式查询
  uv run python phase5_main.py compare   # A/B 对比
        """)
        return

    command = sys.argv[1].lower()

    if command == "build":
        build_index()
    elif command == "query":
        interactive_query()
    elif command == "compare":
        ab_compare()
    else:
        print(f"未知命令: {command}")
        print("可用命令: build / query / compare")


if __name__ == "__main__":
    main()
