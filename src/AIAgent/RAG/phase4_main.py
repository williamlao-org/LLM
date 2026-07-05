"""
Phase 4: 短期记忆 + 长期情景记忆交互实验。

运行：
    uv run python src/AIAgent/RAG/phase4_main.py
    uv run python src/AIAgent/RAG/phase4_main.py --strategy tokens --token-budget 120
    uv run python src/AIAgent/RAG/phase4_main.py \
  --strategy summary \
  --structured-state \
  --episodic-memory-file episodic_memory.json \
  --token-budget 1200 \
  --summary-token-budget 400
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(filepath: str | None) -> str | None:
    """将相对路径解析为相对于脚本所在目录的绝对路径。"""
    if filepath is None:
        return None
    p = Path(filepath)
    if not p.is_absolute():
        p = _SCRIPT_DIR / p
    return str(p)

from config import config
from phase3_agentic_rag import AgenticRAG
from phase3_main import ensure_index
from phase4_episodic_memory import (
    EpisodicAgent,
    EpisodicMemory,
    LLMEpisodeReflector,
    RecalledEpisode,
)
from phase4_structured_memory import (
    LLMWorkingStateExtractor,
    StructuredWorkingMemory,
    WorkingStateExtractor,
)
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
    parser = argparse.ArgumentParser(description="Phase 4 Agent 记忆实验")
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
    parser.add_argument(
        "--structured-state",
        action="store_true",
        help="启用门控批处理的结构化工作记忆",
    )
    parser.add_argument(
        "--structured-state-file",
        type=str,
        default=None,
        help="结构化记忆的持久化存储文件路径（JSON格式）",
    )
    parser.add_argument(
        "--episodic-memory-file",
        type=str,
        default=None,
        help="启用情景记忆并指定 JSON 经验库路径",
    )
    parser.add_argument(
        "--episodic-top-k",
        type=int,
        default=3,
        help="每次召回的最大经验数（默认: 3）",
    )
    parser.add_argument(
        "--episodic-min-score",
        type=float,
        default=0.35,
        help="情景记忆最低余弦相似度（默认: 0.35）",
    )
    parser.add_argument(
        "--episodic-max-episodes",
        type=int,
        default=200,
        help="经验库最大条数（默认: 200）",
    )
    return parser.parse_args(argv)


def build_memory(
    args: argparse.Namespace,
    summarizer: ConversationSummarizer | None = None,
    state_extractor: WorkingStateExtractor | None = None,
    token_counter: TokenCounter | None = None,
    turn_token_counter: TurnTokenCounter | None = None,
) -> WorkingMemory:
    if args.strategy == "summary":
        if summarizer is None:
            raise ValueError("summary 策略需要 summarizer")
        base_memory: WorkingMemory = SummaryBufferMemory(
            max_recent_tokens=args.token_budget,
            max_summary_tokens=args.summary_token_budget,
            summarizer=summarizer,
            token_counter=token_counter,
            turn_token_counter=turn_token_counter,
        )
    elif args.strategy == "tokens":
        base_memory = TokenBudgetMemory(
            max_tokens=args.token_budget,
            token_counter=token_counter,
            turn_token_counter=turn_token_counter,
        )
    else:
        base_memory = ConversationWindowMemory(max_turns=args.max_turns)

    if args.structured_state:
        if state_extractor is None:
            raise ValueError("--structured-state 需要 state_extractor")
        return StructuredWorkingMemory(
            base_memory=base_memory,
            extractor=state_extractor,
            filepath=_resolve_path(args.structured_state_file),
        )
    return base_memory


def get_base_memory(memory: WorkingMemory) -> WorkingMemory:
    if isinstance(memory, StructuredWorkingMemory):
        return memory.base_memory
    return memory


def print_banner(
    memory: WorkingMemory,
    episodic_memory: EpisodicMemory | None = None,
) -> None:
    base_memory = get_base_memory(memory)
    if isinstance(base_memory, SummaryBufferMemory):
        strategy = (
            f"摘要缓冲（原文 {base_memory.max_recent_tokens} + "
            f"摘要 {base_memory.max_summary_tokens} tokens）"
        )
    elif isinstance(base_memory, TokenBudgetMemory):
        strategy = f"Token 预算（{base_memory.max_tokens} tokens）"
    else:
        strategy = f"轮数窗口（{base_memory.max_turns} 轮）"
    if isinstance(memory, StructuredWorkingMemory):
        strategy = "结构化状态 + " + strategy
    if episodic_memory is not None:
        strategy += f" + 情景记忆({len(episodic_memory)} 条)"

    print(f"""
╔════════════════════════════════════════════════════════╗
║                 🧠 Phase 4：Agent 记忆                 ║
║                                                        ║
║   当前策略: {strategy:<41}║
╚════════════════════════════════════════════════════════╝
""")


def print_help(
    memory: WorkingMemory,
    episodic_memory: EpisodicMemory | None = None,
) -> None:
    common = """
📖 可用命令:
  直接输入问题  → 使用当前短期记忆继续对话
  /memory       → 查看当前窗口中的完整问答
  /clear        → 清空短期记忆
  /help         → 显示帮助
  /quit         → 退出
"""

    if isinstance(memory, StructuredWorkingMemory):
        common += """  /state        → 查看结构化状态和 pending 批次
  /extract      → 手动抽取尚未达到门槛的 pending 回合
  /forget <category> <key> → 确定性删除一个条目
"""

    if episodic_memory is not None:
        common += """  /episodes     → 查看已持久化的历史经验
  /recall <query> → 手动召回相似经验
  /forget-episode <id> → 删除指定经验
  /clear-episodes → 清空长期情景记忆
"""

    base_memory = get_base_memory(memory)
    if isinstance(memory, StructuredWorkingMemory):
        experiment = """
💡 结构化工作记忆实验:
  1. 说“我喜欢蓝色”，观察显式信号如何立即更新状态
  2. 进行普通对话，观察每 5 轮的兜底批处理
  3. 用 /state、/extract 和 /forget 检查门控与更新
"""
    elif isinstance(base_memory, SummaryBufferMemory):
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


def print_recalled(recalled: tuple[RecalledEpisode, ...]) -> None:
    if not recalled:
        print("  （没有达到相似度门槛的历史经验）")
        return
    for index, item in enumerate(recalled, 1):
        episode = item.episode
        print(
            f"  [{index}] score={item.score:.4f} "
            f"id={episode.id} outcome={episode.outcome}"
        )
        print(f"      任务: {episode.task}")
        print(f"      反思: {episode.reflection.summary}")


def print_episodes(memory: EpisodicMemory) -> None:
    print(f"\n📚 情景记忆: {len(memory)}/{memory.max_episodes} 条")
    if not memory.episodes:
        print("  （空）")
    for index, episode in enumerate(reversed(memory.episodes), 1):
        print(
            f"  [{index}] {episode.created_at} "
            f"id={episode.id} outcome={episode.outcome}"
        )
        print(f"      任务: {episode.task}")
        print(f"      经验: {episode.reflection.summary}")
    if memory.last_load_error:
        print(f"  ⚠️ 经验库加载失败: {memory.last_load_error}")


def print_memory(memory: WorkingMemory) -> None:
    if isinstance(memory, StructuredWorkingMemory):
        print_state(memory)
    base_memory = get_base_memory(memory)

    if isinstance(base_memory, SummaryBufferMemory):
        usage = (
            f"{len(base_memory)} 轮近期原文，"
            f"{base_memory.recent_tokens}/{base_memory.max_recent_tokens} "
            "DeepSeek V4 tokens；"
            f"摘要 {base_memory.summary_tokens}/{base_memory.max_summary_tokens} "
            "DeepSeek V4 tokens"
        )
    elif isinstance(base_memory, TokenBudgetMemory):
        usage = (
            f"{len(base_memory)} 轮，"
            f"{base_memory.current_tokens}/{base_memory.max_tokens} "
            "DeepSeek V4 tokens"
        )
    else:
        usage = f"{len(base_memory)}/{base_memory.max_turns} 轮"

    print(f"\n🧠 当前记忆: {usage}")
    if isinstance(base_memory, SummaryBufferMemory):
        print("  📝 历史摘要:")
        print(f"     {base_memory.summary or '（空）'}")
        if base_memory.last_summary_error:
            print(f"  ⚠️ 最近摘要失败: {base_memory.last_summary_error}")
        print("  💬 近期原文:")
    if not base_memory.turns:
        print("  （空）")
        return

    for index, turn in enumerate(base_memory.turns, 1):
        print(f"  [{index}] 用户: {turn.user}")
        print(f"      助手: {turn.assistant}")


def print_state(memory: StructuredWorkingMemory) -> None:
    print(
        f"\n📌 结构化状态: {len(memory.entries)}/{memory.max_entries} 项，"
        f"pending={len(memory.pending_turns)}，version={memory.state_version}"
    )
    if not memory.entries:
        print("  （空）")
    for entry in memory.entries:
        print(
            f"  [{entry.category}] {entry.key} = {entry.value} "
            f"(created={entry.created_turn}, updated={entry.updated_turn})"
        )
    if memory.last_extraction_error:
        print(f"  ⚠️ 最近抽取失败: {memory.last_extraction_error}")


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
        state_extractor = (
            LLMWorkingStateExtractor(agent.llm_client, agent.llm_model)
            if args.structured_state
            else None
        )
        memory = build_memory(
            args,
            summarizer=summarizer,
            state_extractor=state_extractor,
            token_counter=(
                deepseek_counter.count_text if deepseek_counter else None
            ),
            turn_token_counter=(
                deepseek_counter.count_turn if deepseek_counter else None
            ),
        )
        episodic_memory = None
        query_agent = agent
        if args.episodic_memory_file:
            episodic_memory = EpisodicMemory(
                filepath=_resolve_path(args.episodic_memory_file),
                embedder=agent.embedder,
                reflector=LLMEpisodeReflector(
                    agent.llm_client,
                    agent.llm_model,
                ),
                top_k=args.episodic_top_k,
                min_similarity=args.episodic_min_score,
                max_episodes=args.episodic_max_episodes,
            )
            query_agent = EpisodicAgent(agent, episodic_memory)
        print_banner(memory, episodic_memory)
        if not ensure_index(agent):
            return
    except Exception as error:
        print(f"❌ 初始化失败: {error}")
        return

    print_help(memory, episodic_memory)

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
            print_help(memory, episodic_memory)
            continue
        if question == "/memory":
            print_memory(memory)
            continue
        if question == "/state":
            if isinstance(memory, StructuredWorkingMemory):
                print_state(memory)
            else:
                print("未启用结构化工作记忆，请使用 --structured-state。")
            continue
        if question == "/episodes":
            if episodic_memory is None:
                print("未启用情景记忆，请使用 --episodic-memory-file。")
            else:
                print_episodes(episodic_memory)
            continue
        if question.startswith("/recall"):
            if episodic_memory is None:
                print("未启用情景记忆，请使用 --episodic-memory-file。")
                continue
            parts = question.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("用法: /recall <query>")
                continue
            print("\n🔎 相似历史经验:")
            print_recalled(episodic_memory.recall(parts[1]))
            if episodic_memory.last_recall_error:
                print(f"⚠️ 召回失败: {episodic_memory.last_recall_error}")
            continue
        if question.startswith("/forget-episode"):
            if episodic_memory is None:
                print("未启用情景记忆，请使用 --episodic-memory-file。")
                continue
            parts = question.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                print("用法: /forget-episode <id>")
                continue
            removed = episodic_memory.delete(parts[1].strip())
            print("✅ 已删除。" if removed else "未找到匹配经验。")
            continue
        if question == "/clear-episodes":
            if episodic_memory is None:
                print("未启用情景记忆，请使用 --episodic-memory-file。")
            elif episodic_memory.clear():
                print("✅ 长期情景记忆已清空。")
            else:
                print(f"⚠️ 清空失败: {episodic_memory.last_recording_error}")
            continue
        if question == "/extract":
            if not isinstance(memory, StructuredWorkingMemory):
                print("未启用结构化工作记忆，请使用 --structured-state。")
                continue
            previous_version = memory.state_version
            if not memory.flush_pending():
                print("没有待抽取的 pending 回合。")
            elif memory.last_extraction_error:
                print(f"⚠️ 抽取失败: {memory.last_extraction_error}")
            elif memory.state_version == previous_version:
                print("✅ 抽取完成，没有需要更新的结构化信息。")
            else:
                print("✅ 结构化状态已更新。")
                print_state(memory)
            continue
        if question.startswith("/forget"):
            if not isinstance(memory, StructuredWorkingMemory):
                print("未启用结构化工作记忆，请使用 --structured-state。")
                continue
            parts = question.split(maxsplit=2)
            if len(parts) != 3:
                print("用法: /forget <category> <key>")
                continue
            try:
                removed = memory.forget(parts[1], parts[2])
                print("✅ 已删除。" if removed else "未找到匹配条目。")
            except ValueError as error:
                print(f"删除失败: {error}")
            continue
        if question == "/clear":
            memory.clear()
            print("✅ 短期记忆已清空。")
            continue
        if question.startswith("/"):
            print(f"未知命令: {question}，输入 /help 查看帮助。")
            continue

        try:
            structured_memory = (
                memory if isinstance(memory, StructuredWorkingMemory) else None
            )
            base_memory = get_base_memory(memory)
            previous_summary = (
                base_memory.summary
                if isinstance(base_memory, SummaryBufferMemory)
                else None
            )
            previous_summary_error = (
                base_memory.last_summary_error
                if isinstance(base_memory, SummaryBufferMemory)
                else None
            )
            previous_state_version = (
                structured_memory.state_version if structured_memory else None
            )
            previous_extraction_error = (
                structured_memory.last_extraction_error
                if structured_memory
                else None
            )
            previous_episode_count = (
                len(episodic_memory) if episodic_memory is not None else None
            )
            query_agent.query(question, verbose=True, memory=memory)
            if episodic_memory is not None:
                if isinstance(query_agent, EpisodicAgent) and query_agent.last_recalled:
                    print("\n🧠 本轮已召回的历史经验:")
                    print_recalled(query_agent.last_recalled)
                if len(episodic_memory) != previous_episode_count:
                    newest = episodic_memory.episodes[-1]
                    print(
                        "\n💾 已记录任务经验: "
                        f"id={newest.id} outcome={newest.outcome}"
                    )
                if episodic_memory.last_reflection_error:
                    print(
                        "\n⚠️ 自动反思失败，已使用降级记录: "
                        f"{episodic_memory.last_reflection_error}"
                    )
                if episodic_memory.last_recording_error:
                    print(
                        "\n⚠️ 情景记忆写入失败，主回答不受影响: "
                        f"{episodic_memory.last_recording_error}"
                    )
            if isinstance(base_memory, SummaryBufferMemory):
                if base_memory.summary != previous_summary:
                    print(f"\n📝 历史摘要已更新:\n{base_memory.summary}")
                if (
                    base_memory.last_summary_error
                    and base_memory.last_summary_error != previous_summary_error
                ):
                    print(
                        "\n⚠️ 摘要失败，已保持上下文预算: "
                        f"{base_memory.last_summary_error}"
                    )
            if structured_memory:
                if structured_memory.state_version != previous_state_version:
                    print("\n📌 结构化状态已更新:")
                    print_state(structured_memory)
                if (
                    structured_memory.last_extraction_error
                    and structured_memory.last_extraction_error
                    != previous_extraction_error
                ):
                    print(
                        "\n⚠️ 结构化抽取失败，主回答不受影响: "
                        f"{structured_memory.last_extraction_error}"
                    )
        except Exception as error:
            print(f"\n❌ 查询出错: {error}")


if __name__ == "__main__":
    main()
