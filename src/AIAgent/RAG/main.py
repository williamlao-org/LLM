"""
RAG 交互式问答入口

启动后可以：
1. 优先加载本地索引，缺失时自动构建
2. 交互式提问，实时看到检索和生成过程
3. 输入 /sources 查看最近一次回答的来源
4. 输入 /rebuild 重建索引
5. 输入 /quit 退出
"""
import sys
import os

# 把当前目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_chain import RAGChain
from config import config


def print_banner():
    print("""
╔══════════════════════════════════════════╗
║         📚 RAG 知识库问答系统             ║
║     Phase 1: 经典 RAG 学习项目            ║
╚══════════════════════════════════════════╝
    """)


def print_help():
    print("""
📖 可用命令:
  直接输入问题  → 进行 RAG 检索问答
  /sources     → 查看上次回答的来源详情
  /rebuild     → 重建知识库索引并覆盖本地缓存
  /help        → 显示帮助
  /quit        → 退出程序
""")


def ensure_index(rag: RAGChain) -> bool:
    """优先加载本地索引，加载不到时构建并保存。"""
    print("📦 准备知识库索引...\n")

    if rag.load_index():
        print(f"✅ 已加载本地索引，共 {len(rag.store)} 个 chunks。\n")
        return True

    print("\n📦 本地索引不可用，开始构建知识库索引...\n")
    built_count = rag.build_index()

    if built_count == 0 or len(rag.store) == 0:
        print("❌ 索引为空，无法启动问答。")
        return False

    rag.save_index()
    print("✅ 索引已构建并保存到本地缓存。\n")
    return True


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
        print("⚠️  未设置 Embedding API Key！请先设置环境变量：")
        print("  export SILICONFLOW_API_KEY=your_siliconflow_api_key")
        print()
        key = input("或者直接在这里输入你的 SiliconFlow API Key: ").strip()
        if key:
            config.embedding_api_key = key
        else:
            print("❌ 没有 Embedding API Key，无法启动。")
            return

    # 初始化 RAG Chain
    print("🚀 初始化 RAG 系统...\n")
    try:
        rag = RAGChain(
            embedder_type="api",
            store_type="simple",  # 学习阶段用 simple，理解原理
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 加载或构建索引
    try:
        if not ensure_index(rag):
            return
    except Exception as e:
        print(f"❌ 索引准备失败: {e}")
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
            cmd = question.lower()

            if cmd == "/quit" or cmd == "/exit":
                print("👋 再见！")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/sources":
                if last_result and last_result["sources"]:
                    print("\n📋 上次回答的来源:")
                    for i, src in enumerate(last_result["sources"]):
                        print(f"\n  [{i+1}] 来源: {src['source']}")
                        print(f"      相似度: {src['score']:.4f}")
                        print(f"      内容: {src['content_preview']}...")
                else:
                    print("还没有查询记录。")

            elif cmd == "/rebuild":
                print("🔄 重建索引...\n")
                built_count = rag.build_index()
                if built_count == 0:
                    print("❌ 没有构建出任何索引，已保留当前缓存。")
                    continue
                rag.save_index()
                print("✅ 索引重建完成！")

            else:
                print(f"未知命令: {question}，输入 /help 查看帮助")

            continue

        # 正常的 RAG 查询
        try:
            result = rag.query(question)
            last_result = result
        except Exception as e:
            print(f"\n❌ 查询出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
