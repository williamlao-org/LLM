"""
Phase 4: 短期记忆交互实验。

运行：
    uv run python src/AIAgent/RAG/phase4_main.py
    uv run python src/AIAgent/RAG/phase4_main.py --strategy tokens --token-budget 120
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from phase3_agentic_rag import AgenticRAG
from phase3_main import ensure_index
from phase4_token_memory import TokenBudgetMemory
from phase4_working_memory import ConversationWindowMemory, WorkingMemory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 短期记忆实验")
    parser.add_argument(
        "--strategy",
        choices=("turns", "tokens"),
        default="turns",
        help="记忆裁剪策略（默认: turns）",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="turns 策略保留的最大轮数（默认: 3）",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=1200,
        help="tokens 策略的历史 Token 预算（默认: 120）",
    )
    return parser.parse_args(argv)


def build_memory(args: argparse.Namespace) -> WorkingMemory:
    if args.strategy == "tokens":
        return TokenBudgetMemory(max_tokens=args.token_budget)
    return ConversationWindowMemory(max_turns=args.max_turns)


def print_banner(memory: WorkingMemory) -> None:
    if isinstance(memory, TokenBudgetMemory):
        strategy = f"Token 预算（{memory.max_tokens} tokens）"
    else:
        strategy = f"轮数窗口（{memory.max_turns} 轮）"

    print(f"""
╔════════════════════════════════════════════════════════╗
║                 🧠 Phase 4：短期记忆                  ║
║                                                        ║
║   当前策略: {strategy:<41}║
╚════════════════════════════════════════════════════════╝
""")


def print_help() -> None:
    print("""
📖 可用命令:
  直接输入问题  → 使用当前短期记忆继续对话
  /memory       → 查看当前窗口中的完整问答
  /clear        → 清空短期记忆
  /help         → 显示帮助
  /quit         → 退出

💡 滑动窗口实验:
  1. 告诉 Agent：“我叫小林”
  2. 在 3 轮窗口内追问：“我叫什么？”
  3. 再进行几轮无关对话，用 /memory 观察最早信息被淘汰
""")


def print_memory(memory: WorkingMemory) -> None:
    if isinstance(memory, TokenBudgetMemory):
        usage = (
            f"{len(memory)} 轮，"
            f"{memory.current_tokens}/{memory.max_tokens} 估算 tokens"
        )
    else:
        usage = f"{len(memory)}/{memory.max_turns} 轮"

    print(f"\n🧠 当前记忆: {usage}")
    if not memory.turns:
        print("  （空）")
        return

    for index, turn in enumerate(memory.turns, 1):
        print(f"  [{index}] 用户: {turn.user}")
        print(f"      助手: {turn.assistant}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        memory = build_memory(args)
    except ValueError as error:
        raise SystemExit(f"参数错误: {error}") from error
    print_banner(memory)

    if not config.llm_api_key or not config.embedding_api_key:
        print("❌ 请先在环境变量中配置 LLM_API_KEY 和 SILICONFLOW_API_KEY。")
        return

    try:
        agent = AgenticRAG(use_router=True, use_reranker=False)
        if not ensure_index(agent):
            return
    except Exception as error:
        print(f"❌ 初始化失败: {error}")
        return

    print_help()

    while True:
        try:
            question = input("\n❓ 你的问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if not question:
            continue
        if question in ("/quit", "/exit"):
            print("👋 再见！")
            break
        if question == "/help":
            print_help()
            continue
        if question == "/memory":
            print_memory(memory)
            continue
        if question == "/clear":
            memory.clear()
            print("✅ 短期记忆已清空。")
            continue
        if question.startswith("/"):
            print(f"未知命令: {question}，输入 /help 查看帮助。")
            continue

        try:
            agent.query(question, verbose=True, memory=memory)
        except Exception as error:
            print(f"\n❌ 查询出错: {error}")


if __name__ == "__main__":
    main()
