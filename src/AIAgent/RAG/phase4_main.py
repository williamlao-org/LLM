"""
Phase 4: 短期记忆交互实验。

运行：
    uv run python src/AIAgent/RAG/phase4_main.py
    uv run python src/AIAgent/RAG/phase4_main.py --strategy tokens --token-budget 120
    uv run python src/AIAgent/RAG/phase4_main.py \
  --strategy summary \
  --token-budget 1200 \
  --summary-token-budget 800
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from phase3_agentic_rag import AgenticRAG
from phase3_main import ensure_index
from phase4_summary_memory import (
    ConversationSummarizer,
    LLMConversationSummarizer,
    SummaryBufferMemory,
)
from phase4_token_memory import (
    DeepSeekV4TokenCounter,
    TokenBudgetMemory,
    TokenCounter,
    TurnTokenCounter,
)
from phase4_working_memory import ConversationWindowMemory, WorkingMemory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 短期记忆实验")
    parser.add_argument(
        "--strategy",
        choices=("turns", "tokens", "summary"),
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
        help="tokens/summary 策略的近期原文预算（默认: 1200）",
    )
    parser.add_argument(
        "--summary-token-budget",
        type=int,
        default=400,
        help="summary 策略的滚动摘要预算（默认: 400）",
    )
    return parser.parse_args(argv)


def build_memory(
    args: argparse.Namespace,
    summarizer: ConversationSummarizer | None = None,
    token_counter: TokenCounter | None = None,
    turn_token_counter: TurnTokenCounter | None = None,
) -> WorkingMemory:
    if args.strategy == "summary":
        if summarizer is None:
            raise ValueError("summary 策略需要 summarizer")
        return SummaryBufferMemory(
            max_recent_tokens=args.token_budget,
            max_summary_tokens=args.summary_token_budget,
            summarizer=summarizer,
            token_counter=token_counter,
            turn_token_counter=turn_token_counter,
        )
    if args.strategy == "tokens":
        return TokenBudgetMemory(
            max_tokens=args.token_budget,
            token_counter=token_counter,
            turn_token_counter=turn_token_counter,
        )
    return ConversationWindowMemory(max_turns=args.max_turns)


def print_banner(memory: WorkingMemory) -> None:
    if isinstance(memory, SummaryBufferMemory):
        strategy = (
            f"摘要缓冲（原文 {memory.max_recent_tokens} + "
            f"摘要 {memory.max_summary_tokens} tokens）"
        )
    elif isinstance(memory, TokenBudgetMemory):
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


def print_help(memory: WorkingMemory) -> None:
    common = """
📖 可用命令:
  直接输入问题  → 使用当前短期记忆继续对话
  /memory       → 查看当前窗口中的完整问答
  /clear        → 清空短期记忆
  /help         → 显示帮助
  /quit         → 退出
"""

    if isinstance(memory, SummaryBufferMemory):
        experiment = """
💡 摘要缓冲实验:
  1. 用小 Token 预算启动，告诉 Agent 你的名字和偏好
  2. 继续多轮对话，直到早期原文被淘汰
  3. 观察自动打印的滚动摘要，或用 /memory 查看完整记忆
"""
    else:
        experiment = """

💡 滑动窗口实验:
  1. 告诉 Agent：“我叫小林”
  2. 在 3 轮窗口内追问：“我叫什么？”
  3. 再进行几轮无关对话，用 /memory 观察最早信息被淘汰
"""
    print(common + experiment)


def print_memory(memory: WorkingMemory) -> None:
    if isinstance(memory, SummaryBufferMemory):
        usage = (
            f"{len(memory)} 轮近期原文，"
            f"{memory.recent_tokens}/{memory.max_recent_tokens} DeepSeek V4 tokens；"
            f"摘要 {memory.summary_tokens}/{memory.max_summary_tokens} "
            "DeepSeek V4 tokens"
        )
    elif isinstance(memory, TokenBudgetMemory):
        usage = (
            f"{len(memory)} 轮，{memory.current_tokens}/{memory.max_tokens} "
            "DeepSeek V4 tokens"
        )
    else:
        usage = f"{len(memory)}/{memory.max_turns} 轮"

    print(f"\n🧠 当前记忆: {usage}")
    if isinstance(memory, SummaryBufferMemory):
        print("  📝 历史摘要:")
        print(f"     {memory.summary or '（空）'}")
        if memory.last_summary_error:
            print(f"  ⚠️ 最近摘要失败: {memory.last_summary_error}")
        print("  💬 近期原文:")
    if not memory.turns:
        print("  （空）")
        return

    for index, turn in enumerate(memory.turns, 1):
        print(f"  [{index}] 用户: {turn.user}")
        print(f"      助手: {turn.assistant}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not config.llm_api_key or not config.embedding_api_key:
        print("❌ 请先在环境变量中配置 LLM_API_KEY 和 SILICONFLOW_API_KEY。")
        return

    try:
        deepseek_counter = None
        if args.strategy in ("tokens", "summary"):
            print(f"🔢 加载 Tokenizer: {config.llm_tokenizer_model} ...")
            deepseek_counter = DeepSeekV4TokenCounter.from_pretrained(
                config.llm_tokenizer_model
            )

        agent = AgenticRAG(use_router=True, use_reranker=False)
        summarizer = (
            LLMConversationSummarizer(agent.llm_client, agent.llm_model)
            if args.strategy == "summary"
            else None
        )
        memory = build_memory(
            args,
            summarizer=summarizer,
            token_counter=(
                deepseek_counter.count_text if deepseek_counter else None
            ),
            turn_token_counter=(
                deepseek_counter.count_turn if deepseek_counter else None
            ),
        )
        print_banner(memory)
        if not ensure_index(agent):
            return
    except Exception as error:
        print(f"❌ 初始化失败: {error}")
        return

    print_help(memory)

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
            print_help(memory)
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
            previous_summary = (
                memory.summary
                if isinstance(memory, SummaryBufferMemory)
                else None
            )
            previous_summary_error = (
                memory.last_summary_error
                if isinstance(memory, SummaryBufferMemory)
                else None
            )
            agent.query(question, verbose=True, memory=memory)
            if isinstance(memory, SummaryBufferMemory):
                if memory.summary != previous_summary:
                    print(f"\n📝 历史摘要已更新:\n{memory.summary}")
                if (
                    memory.last_summary_error
                    and memory.last_summary_error != previous_summary_error
                ):
                    print(
                        "\n⚠️ 摘要失败，已保持上下文预算: "
                        f"{memory.last_summary_error}"
                    )
        except Exception as error:
            print(f"\n❌ 查询出错: {error}")


if __name__ == "__main__":
    main()
