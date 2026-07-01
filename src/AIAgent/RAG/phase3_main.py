"""
Phase 3: Agentic RAG 交互式入口

启动后可以：
1. 自动构建或加载多知识库索引
2. 交互式提问，观察 Agent 的完整决策过程：
   - 是否需要检索？
   - 路由到哪个知识库？
   - 检索质量如何？需不需要重搜？
3. /compare  对比传统 RAG vs Agentic RAG
4. /steps    查看上次回答的 Agent 决策步骤
5. /rebuild  重建所有知识库索引
6. /help     显示帮助
7. /quit     退出
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3_agentic_rag import AgenticRAG, compare_with_naive
from config import config


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║             🤖 Agentic RAG 智能问答系统                   ║
║          Phase 3: Agent 自主控制检索行为                   ║
║                                                          ║
║   ┌─────────────────────────────────────────────────┐    ║
║   │  传统 RAG:  问题 → 检索 → 生成（无脑管线）       │    ║
║   │  Agentic:   问题 → Agent 决策 → 按需检索/路由   │    ║
║   │                   → 质量评估 → 迭代优化 → 生成   │    ║
║   └─────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════╝
    """)


def print_help():
    print("""
📖 可用命令:
  直接输入问题      → Agentic RAG 智能问答（观察 Agent 决策过程）
  /compare <问题>  → 对比传统 RAG vs Agentic RAG
  /steps           → 查看上次回答的 Agent 决策步骤详情
  /rebuild         → 重建所有知识库索引
  /help            → 显示帮助
  /quit            → 退出程序

💡 试试这些问题来体验 Agentic RAG 的不同行为：
  • "1+1 等于几？"             → Agent 应该直接回答（自适应检索）
  • "Transformer 注意力机制"    → 路由到技术文档库
  • "纳瓦尔对财富的看法"        → 路由到通用知识库
  • "RAG 和 Agent 有什么关系？" → 可能触发多跳检索
""")


def ensure_index(agentic: AgenticRAG) -> bool:
    """优先加载本地索引，加载不到时构建并保存。"""
    print("📦 准备知识库索引...\n")

    if agentic.load_indexes():
        print("\n✅ 已加载本地索引。\n")
        return True

    print("\n📦 本地索引不可用，开始构建知识库索引...\n")

    if agentic.use_router:
        total = agentic.build_default_indexes()
    else:
        total = agentic.build_single_index()

    if total == 0:
        print("❌ 索引为空，无法启动问答。")
        return False

    agentic.save_indexes()
    print("\n✅ 索引已构建并保存到本地缓存。\n")
    return True


def format_steps(steps: list[dict]) -> str:
    """格式化 Agent 的决策步骤"""
    if not steps:
        return "  （没有工具调用记录）"

    parts = []
    for i, step in enumerate(steps, 1):
        tool = step["tool"]
        args = step.get("args", {})
        preview = step.get("result_preview", "")

        icon = {
            "search_knowledge_base": "🔍",
            "assess_retrieval_quality": "📊",
            "direct_answer": "💡",
        }.get(tool, "🔧")

        parts.append(f"  [{i}] {icon} {tool}")
        if args:
            for k, v in args.items():
                v_str = str(v)
                if len(v_str) > 80:
                    v_str = v_str[:80] + "..."
                parts.append(f"      {k}: {v_str}")
        if preview:
            preview_short = preview[:100].replace("\n", " ")
            parts.append(f"      → {preview_short}...")

    return "\n".join(parts)


def main():
    print_banner()

    # 检查 API Key
    if not config.llm_api_key:
        print("⚠️  未设置 LLM API Key！请先设置环境变量：")
        print("  export LLM_API_KEY=your_deepseek_api_key")
        print()
        key = input("或者直接在这里输入你的 DeepSeek API Key: ").strip()
        if key:
            config.llm_api_key = key
        else:
            print("❌ 没有 LLM API Key，无法启动。")
            return

    if not config.embedding_api_key:
        print("⚠️  未设置 Embedding API Key！")
        key = input("输入你的 SiliconFlow API Key: ").strip()
        if key:
            config.embedding_api_key = key
        else:
            print("❌ 没有 Embedding API Key，无法启动。")
            return

    # 初始化 Agentic RAG
    print("🚀 初始化 Agentic RAG 系统...\n")
    try:
        agentic = AgenticRAG(
            use_router=True,      # 启用多知识库路由
            use_reranker=False,   # 学习阶段先不用 reranker，观察 Agent 行为
            max_iterations=5,
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 加载或构建索引
    try:
        if not ensure_index(agentic):
            return
    except Exception as e:
        print(f"❌ 索引准备失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print_help()

    # 交互式问答循环
    last_result = None

    while True:
        try:
            question = input("\n❓ 你的问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 再见！")
            break

        if not question:
            continue

        # 处理命令
        if question.startswith("/"):
            parts = question.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print("👋 再见！")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/steps":
                if last_result:
                    print(f"\n📋 上次回答的 Agent 决策步骤:")
                    print(f"  总轮数: {last_result['iterations']}")
                    print(f"  使用检索: {'是' if last_result['used_retrieval'] else '否'}")
                    print(f"\n  工具调用记录:")
                    print(format_steps(last_result["steps"]))
                else:
                    print("还没有查询记录。")

            elif cmd == "/compare":
                compare_question = parts[1] if len(parts) > 1 else None
                if not compare_question:
                    compare_question = input("  输入要对比的问题: ").strip()
                if compare_question:
                    try:
                        compare_with_naive(agentic, compare_question)
                    except Exception as e:
                        print(f"\n❌ 对比出错: {e}")
                        import traceback
                        traceback.print_exc()

            elif cmd == "/rebuild":
                print("🔄 重建所有知识库索引...\n")
                try:
                    if agentic.use_router:
                        total = agentic.build_default_indexes()
                    else:
                        total = agentic.build_single_index()
                    if total == 0:
                        print("❌ 没有构建出任何索引。")
                        continue
                    agentic.save_indexes()
                    print("✅ 索引重建完成！")
                except Exception as e:
                    print(f"❌ 重建失败: {e}")

            else:
                print(f"未知命令: {question}，输入 /help 查看帮助")

            continue

        # 正常的 Agentic RAG 查询
        try:
            result = agentic.query(question, verbose=True)
            last_result = result
        except Exception as e:
            print(f"\n❌ 查询出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
