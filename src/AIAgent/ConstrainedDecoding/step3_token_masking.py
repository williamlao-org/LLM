"""
第三步：从"哪些字符合法"到"哪些 token 合法"

前两步我们搞定了：正则 → 状态机 → 每一步知道哪些字符合法。
但 LLM 不是逐字符生成的！它是逐 token 生成的。

什么是 token？
    LLM 有一个词表（vocabulary），比如 50000 个 token。
    每个 token 是一个字符串片段，可能是：
        - 单字符：  "a", "{", " "
        - 多字符：  "name", "hello", "\": \"", "arguments"
        - 子词：    "hel", "lo", "##ing"

    生成时，模型每一步从词表里挑一个 token，不是挑一个字符。

问题来了：
    状态机告诉我们"下一个字符只能是 {" — 好，那词表里哪些 token 以 { 开头？
    更复杂的：如果一个 token 包含多个字符（比如 "name"），
    那它是否合法取决于这几个字符是否能连续走通状态机。

这就是 outlines 最精妙的一步：
    预计算一张表 —— 对每个 DFA 状态，词表中哪些 token 是合法的。

运行方式：
    cd /Users/slyh/MyDir/Project/LLM
    uv run python -m src.AIAgent.ConstrainedDecoding.step3_token_masking
"""

from __future__ import annotations

import random

from .step1_fsm_basics import FSM
from .step2_regex_to_fsm import regex_to_dfa


# ============================================================
# Part 1：模拟词表（Vocabulary）
# ============================================================
#
# 真实的 LLM 词表有 50000~150000 个 token，每个有一个 ID。
# 这里我们造一个小词表来演示原理。
#
# 关键点：有些 token 是多字符的。比如 "name" 是一个 token，
# 模型可以一步就生成 "name" 四个字符。


MOCK_VOCABULARY: dict[int, str] = {
    # --- 单字符 token ---
    0: "{",
    1: "}",
    2: '"',
    3: ":",
    4: " ",
    5: ",",
    6: "a",
    7: "b",
    8: "c",
    9: "d",
    10: "e",
    11: "h",
    12: "l",
    13: "m",
    14: "n",
    15: "o",
    16: "1",
    17: "2",
    18: "\n",
    # --- 多字符 token（这才是重点！）---
    19: "name",       # 4 个字符，一步生成
    20: "age",        # 3 个字符
    21: '": "',       # 4 个字符，包含引号冒号空格引号
    22: "hello",      # 5 个字符
    23: "world",      # 5 个字符
    24: "dog",        # 3 个字符
    25: '"}',         # 2 个字符：关引号 + 关花括号
    26: '{"',         # 2 个字符：开花括号 + 开引号
    27: "lo",         # 2 个字符
    28: "hel",        # 3 个字符
}


# ============================================================
# Part 2：构建 token mask 表
# ============================================================
#
# 核心算法：
#
#   对词表中每一个 token，对 DFA 的每一个状态：
#       尝试从该状态开始，逐字符走 DFA
#       如果每一步都能走通 → 这个 token 在该状态下是合法的
#       记下走完后到达的状态（下一步从那里继续）
#
#   最终得到一张表：
#       { 状态 → { token_id: 到达状态 } }
#
#   生成时，对于当前 DFA 状态，查表就知道哪些 token 可以采样。
#
# 性能：这个表只需要构建一次（离线），生成时每步只是查字典 O(1)。
# outlines 的核心优化就在这里 —— 预计算，不在采样时做。


def walk_token(fsm: FSM, start_state: str, token: str) -> str | None:
    """从 start_state 出发，逐字符走 token 的每个字符。

    全部走通 → 返回最终状态
    中途走不通 → 返回 None
    """
    fsm.current_state = start_state
    for ch in token:
        if not fsm.step(ch):
            return None  # 这个字符走不通，整个 token 不合法
    return fsm.current_state


def build_token_mask_table(
    fsm: FSM,
    vocabulary: dict[int, str],
) -> dict[str, dict[int, str]]:
    """构建 token mask 表。

    返回：{ DFA状态 → { token_id → 走完后的DFA状态 } }

    对于每个 DFA 状态，字典里存在的 token_id 就是合法的。
    不存在的就是非法的 → 采样时概率置零。
    """
    # 先收集 DFA 的所有状态
    all_states: set[str] = set()
    all_states.add(fsm.start_state)
    all_states.update(fsm.accept_states)
    for (from_s, _), to_s in fsm.transitions.items():
        all_states.add(from_s)
        all_states.add(to_s)

    # 对每个状态 × 每个 token，尝试走一遍
    table: dict[str, dict[int, str]] = {}
    for state in sorted(all_states):
        valid_tokens: dict[int, str] = {}
        for token_id, token_str in vocabulary.items():
            end_state = walk_token(fsm, state, token_str)
            if end_state is not None:
                valid_tokens[token_id] = end_state
        table[state] = valid_tokens

    return table


# ============================================================
# Part 3：用 mask 表模拟约束生成
# ============================================================


def simulate_constrained_generation(
    fsm: FSM,
    vocabulary: dict[int, str],
    table: dict[str, dict[int, str]],
    seed: int = 42,
) -> str:
    """模拟约束生成过程。

    每一步：
        1. 查 mask 表，得到当前状态下合法的 token 集合
        2. 从中随机选一个（模拟 LLM 采样，但只从合法 token 里选）
        3. 拼接到输出，更新状态
        4. 到达接受状态 → 结束

    因为每一步只从合法 token 里采样，输出保证符合正则。
    """
    rng = random.Random(seed)
    current_state = fsm.start_state
    output_tokens: list[str] = []
    steps: list[dict] = []

    for step_num in range(50):  # 防止无限循环
        valid = table.get(current_state, {})

        if not valid:
            break  # 死胡同（不应该发生在正确的 DFA 上）

        # 在合法 token 里，区分"能继续的"和"能结束的"
        can_continue = {
            tid: ns for tid, ns in valid.items() if ns not in fsm.accept_states
        }
        can_finish = {
            tid: ns for tid, ns in valid.items() if ns in fsm.accept_states
        }

        # 如果能结束，有一定概率选择结束（模拟模型决定停下）
        if can_finish and (not can_continue or rng.random() < 0.3):
            chosen_id = rng.choice(list(can_finish.keys()))
        elif can_continue:
            chosen_id = rng.choice(list(can_continue.keys()))
        else:
            chosen_id = rng.choice(list(valid.keys()))

        chosen_token = vocabulary[chosen_id]
        next_state = valid[chosen_id]

        steps.append(
            {
                "step": step_num,
                "state": current_state,
                "valid_count": len(valid),
                "chosen_id": chosen_id,
                "chosen_token": chosen_token,
                "next_state": next_state,
            }
        )

        output_tokens.append(chosen_token)
        current_state = next_state

        if current_state in fsm.accept_states:
            break

    return "".join(output_tokens), steps


# ============================================================
# Part 4：演示
# ============================================================


def demo_mask_table():
    """展示 mask 表的构建过程。"""
    print("=" * 60)
    print('Demo 1：为 {"name": "[a-z]+"} 构建 token mask 表')
    print("=" * 60)

    pattern = r'\{"name": "[a-z]+"\}'
    fsm = regex_to_dfa(pattern)
    table = build_token_mask_table(fsm, MOCK_VOCABULARY)

    # 展示几个关键状态的 mask 表
    print(f"\n正则: {pattern}")
    print(f"词表大小: {len(MOCK_VOCABULARY)} 个 token")
    print()

    # 起始状态
    state = fsm.start_state
    valid = table[state]
    print(f"状态 {state}（起始）→ 合法 token:")
    for tid, ns in sorted(valid.items()):
        token_str = MOCK_VOCABULARY[tid]
        print(f'    token[{tid:>2}] = {token_str!r:<12} → 到达 {ns}')
    total = len(MOCK_VOCABULARY)
    print(f'  → {len(valid)}/{total} 个 token 合法，其余 {total - len(valid)} 个被 mask 掉')

    # 走到 {"（通过单步或多字符 token）
    # 先找到 {" 后的状态
    end_state = walk_token(fsm, fsm.start_state, '{"')
    if end_state:
        print(f'\n状态 {end_state}（已生成 \'{{"\'）→ 合法 token:')
        valid = table[end_state]
        for tid, ns in sorted(valid.items()):
            token_str = MOCK_VOCABULARY[tid]
            print(f'    token[{tid:>2}] = {token_str!r:<12} → 到达 {ns}')
        print(f"  → 只有 'name' 和 'n' 是合法的！'age' 被 mask 掉了")

    # 走到 {"name": " 后（value 内部）
    end_state = walk_token(fsm, fsm.start_state, '{"name": "')
    if end_state:
        print(f'\n状态 {end_state}（已生成 \'{{"name": "\'，value 内部）→ 合法 token:')
        valid = table[end_state]
        for tid, ns in sorted(valid.items()):
            token_str = MOCK_VOCABULARY[tid]
            print(f'    token[{tid:>2}] = {token_str!r:<12} → 到达 {ns}')
        print(f"  → 多字符 token 如 'hello'、'world' 都合法")
        print(f"  → 数字 token '1'、'2' 被 mask 掉")
        print(f"  → 换行 token '\\n' 被 mask 掉")
        print(f"  → 这就是约束解码在 token 层面的效果！")


def demo_constrained_generation():
    """模拟多次约束生成，展示输出总是合法的。"""
    print("\n")
    print("=" * 60)
    print("Demo 2：模拟约束生成（每次输出都保证合法）")
    print("=" * 60)

    pattern = r'\{"name": "[a-z]+"\}'
    fsm = regex_to_dfa(pattern)
    table = build_token_mask_table(fsm, MOCK_VOCABULARY)

    print(f"\n正则: {pattern}")
    print(f"模拟 5 次生成，每次用不同的随机种子:\n")

    for seed in range(5):
        output, steps = simulate_constrained_generation(fsm, MOCK_VOCABULARY, table, seed)
        print(f"  种子={seed}  输出: {output}")

    # 详细展示一次生成过程
    print(f"\n--- 详细展示第一次生成的每一步 ---\n")
    output, steps = simulate_constrained_generation(fsm, MOCK_VOCABULARY, table, seed=0)
    for s in steps:
        valid = table[s["state"]]
        valid_tokens = [f"{MOCK_VOCABULARY[tid]!r}" for tid in sorted(valid.keys())]
        # 只显示前 8 个
        if len(valid_tokens) > 8:
            display = ", ".join(valid_tokens[:8]) + f"... (共{len(valid_tokens)}个)"
        else:
            display = ", ".join(valid_tokens)
        print(
            f"  步骤{s['step']}  状态={s['state']:<4}  "
            f"合法token=[{display}]"
        )
        print(
            f"  {'':>8}  选择: token[{s['chosen_id']}] = {s['chosen_token']!r}  "
            f"→ 到达 {s['next_state']}"
        )
        print()

    print(f"  最终输出: {output}")
    print(f"  格式正确？这是由状态机保证的，不是碰运气！")


def demo_comparison():
    """对比有约束和没约束的区别。"""
    print("\n")
    print("=" * 60)
    print("Demo 3：对比——有约束 vs 无约束")
    print("=" * 60)

    pattern = r'\{"name": "[a-z]+"\}'
    fsm = regex_to_dfa(pattern)
    table = build_token_mask_table(fsm, MOCK_VOCABULARY)

    print(f"""
无约束（你现在的做法）：
  模型从全部 {len(MOCK_VOCABULARY)} 个 token 里自由采样
  可能输出: {{"name": "hello"}}       ← 运气好
  可能输出: {{"name": 123}}           ← 格式错误
  可能输出: {{"name":                 ← 没写完
  可能输出: 好的我来调用工具 {{...}}   ← 多了废话
  → 你的 llm_json_parser 用 index("{{") 去抠，能抢救一些

有约束（outlines 的做法）：
  每一步只从合法 token 里采样""")

    # 展示每个状态的合法/被屏蔽数量
    all_states_in_path: list[tuple[str, str]] = []
    state = fsm.start_state
    for ch in '{"name": "hello"}':
        all_states_in_path.append((state, ch))
        fsm.current_state = state
        fsm.step(ch)
        state = fsm.current_state

    # 去重，只展示几个有代表性的状态
    seen = set()
    for st, ch in all_states_in_path:
        if st in seen:
            continue
        seen.add(st)
        valid = table.get(st, {})
        blocked = len(MOCK_VOCABULARY) - len(valid)
        print(f"  状态 {st}: {len(valid):>2} 个合法 / {blocked:>2} 个屏蔽")

    print(f"""
  → 每一步都有大量 token 被屏蔽
  → 输出 100% 符合格式，不需要事后解析
  → 不需要 index("{{") 抠 JSON
  → 不需要 TurnAbort 错误重试""")


def demo_summary():
    print("\n")
    print("=" * 60)
    print("总结：你实现了 outlines 的完整核心")
    print("=" * 60)
    print("""
三步串起来，就是 outlines 的完整流水线：

  step1: FSM 基础
    → 理解状态机如何追踪"当前位置"和"合法下一步"

  step2: 正则 → NFA → DFA（自动生成状态机）
    → Thompson 构造 + 子集构造
    → JSON Schema → 正则 → 状态机，全自动

  step3: DFA + 词表 → token mask 表（本文件）
    → 预计算每个状态下哪些 token 合法
    → 生成时查表 + mask logits → 输出保证合法

实际的 outlines 还多做了几件事：
  1. 用真实 tokenizer（如 tiktoken、sentencepiece）的词表
  2. 支持完整的 JSON Schema（嵌套对象、数组、枚举...）
  3. 缓存和优化（mask 表可以很大，需要高效存储）
  4. 和推理框架集成（vLLM、transformers 的 LogitsProcessor）

但核心原理就是你刚才写的这些。
从"请输出 JSON"的口头约束，到 mask logits 的代码约束，
区别就在于：一个靠求，一个靠堵。
""")


if __name__ == "__main__":
    demo_mask_table()
    demo_constrained_generation()
    demo_comparison()
    demo_summary()
