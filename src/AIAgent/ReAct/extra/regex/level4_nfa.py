"""
Level 4: 手写 NFA —— 和 level4_dfa.py 配套, 识别"倒数第 3 个字符是 a"的串 (字母表 {a, b})
合法: "aab" "aba" "abb" "aaa" "baab" "bbabb"  (倒数第3个 = a)
非法: "" "a" "ab" "ba" "bb" "bab" "abab"       (太短 / 倒数第3个 = b)

======================================================================
这就是你脑子里那套思路: "盯住一个 a, 赌它是倒数第3个, 然后让它往末尾滑"。

  来个 a -> 分叉: 一个宇宙赌"就是你了"(去 Q1), 另一个宇宙继续往后读(留 Q0)
  Q1 再读 1 个字符 -> Q2 (这个 a 现在是倒数第2)
  Q2 再读 1 个字符 -> Q3 (这个 a 现在是倒数第1... 不对, 是倒数第3? 见下)

你当时卡在"每来一个 a 就要单独设一个 w__a, DFA 装不下" —— 没错, DFA 装不下,
所以才需要 NFA。NFA 的"活跃状态集合"天生就能同时盯住好几个 a:
读到第 n 个 a 就分叉一个新宇宙, 所有赌注并行保存在一个 set 里, 互不干扰。

对照 level4_dfa.py: 同一个语言, DFA 要 16 个状态(死记最近3个字符 = 2^3 满窗口),
NFA 只要 4 个状态。k 从 3 加到 10: DFA 要 2^10≈1024, NFA 还是 11 个 —— 线性 vs 指数。
这就是 NFA 的必要性: 能力和 DFA 相等, 但指数级地更简洁。
======================================================================

NFA 状态图 (无 ε, 纯非确定性 —— 注意 Q0 读 a 时同时去 {Q0, Q1}):

            a,b
          ┌─────┐
          ▼     │
   ▶ (Q0)──────┘
      │  └──a──▶ (Q1)──a/b──▶ (Q2)──a/b──▶ (Q3*)
      │       "赌这个a是倒数第3个"                  (Q3 接受)
      └──b──▶ (Q0)

状态含义 (赌中时, 该 a 距当前末尾的位置):
  Q0 - 还在读, 没赌 (读 a 时分叉: 留 Q0 继续 / 去 Q1 开赌)
  Q1 - 已赌"刚读的这个 a 是倒数第3个", 后面还应恰好有 2 个字符
  Q2 - 赌中的话, 那个 a 现在是倒数第2个, 后面还应恰好有 1 个字符
  Q3 - 接受态 (那个 a 正好落在倒数第3位)。若后面还有字符 = 赌错, 这个宇宙死掉
"""

from enum import Enum, auto


class State(Enum):
    Q0 = auto()  # 初始 & 一直在读前缀(自环), 读到 a 时分叉开赌
    Q1 = auto()  # 刚赌完: 这个 a 应是倒数第3
    Q2 = auto()  # 那个 a 滑到倒数第2
    Q3 = auto()  # 那个 a 滑到倒数第3的落点 = 接受态
    DEAD = auto()  # 显式死亡态: 无路可走的分支陷阱


class CharClass(Enum):
    A = auto()
    B = auto()
    OTHER = auto()

    @classmethod
    def classify(cls, ch: str) -> "CharClass":
        if ch == "a":
            return cls.A
        elif ch == "b":
            return cls.B
        else:
            return cls.OTHER


def validate_nfa(
    initial_state: State,
    accept_states: set[State],
    transition: dict[State, dict[CharClass, set[State]]],
):
    if initial_state not in State:
        raise ValueError("初始状态不合法")

    for accept_state in accept_states:
        if accept_state not in State:
            raise ValueError(f"接收状态 {accept_state} 不合法")

    for state in State:
        if state not in transition.keys():
            raise ValueError(f"transition 没有定义状态 {state}")

        for char in CharClass:
            to_states = transition[state]
            if char not in to_states.keys():
                raise ValueError(f"transition 没有定义状态 {state} 接收 {char} 的去向")
            # NFA 与 DFA 的唯一区别: 去向是一个"状态集合"(可空集=死, 可多元素=分叉)
            if not isinstance(to_states[char], set):
                raise ValueError(
                    f"transition 定义状态 {state} 接收 {char} 时去向不是集合: {to_states[char]}"
                )
            for target in to_states[char]:
                if not isinstance(target, State):
                    raise ValueError(
                        f"transition 定义状态 {state} 接收 {char} 时跳转到非法状态 {target}"
                    )


# 状态转移表 —— 值是 set, 这是 NFA 的命根子。
# Q0 读 A 去 {Q0, Q1}: 一个输入分叉成两个状态 = 非确定性。
# 无效转移都指向 DEAD —— 这个分支进陷阱, 后续所有字符都在 DEAD 自环。
S = State
C = CharClass
transition = {
    S.Q0: {C.A: {S.Q0, S.Q1}, C.B: {S.Q0}, C.OTHER: {S.DEAD}},  # 留着继续 / 同时开赌
    S.Q1: {C.A: {S.Q2}, C.B: {S.Q2}, C.OTHER: {S.DEAD}},  # 再吃 1 个字符 -> 倒数第2
    S.Q2: {C.A: {S.Q3}, C.B: {S.Q3}, C.OTHER: {S.DEAD}},  # 再吃 1 个字符 -> 倒数第3(接受)
    S.Q3: {C.A: {S.DEAD}, C.B: {S.DEAD}, C.OTHER: {S.DEAD}},  # 接受态无出边(再来字符=赌错=死)
    S.DEAD: {C.A: {S.DEAD}, C.B: {S.DEAD}, C.OTHER: {S.DEAD}},  # 自环陷阱: 进来就出不去
}
initial_state = State.Q0
accept_states = {State.Q3}
validate_nfa(initial_state, accept_states, transition)


def simulate(s: str) -> bool:
    """
    NFA 的灵魂: 维护"当前同时所在的状态集合", 逐字符推进。
    本例没有 ε, 所以不需要 ε-闭包, 逻辑更纯粹。

      1. current = {初始状态}
      2. 每读一个字符 ch:  next = current 里每个状态在 ch 下去向的并集
      3. 读完后, current 与 accept_states 有交集就接受
    """
    current = {initial_state}
    print(f"  初始活跃集: {{{', '.join(st.name for st in current)}}}")

    for ch in s:
        cat = CharClass.classify(ch)
        nxt = set()
        for st in current:
            targets = transition[st][cat]
            nxt |= targets
            # 标记进入 DEAD 的分支
            if targets == {S.DEAD}:
                print(f"    {st.name} --{cat.name}--> DEAD ✗")
            else:
                print(f"    {st.name} --{cat.name}--> {targets}")
        current = nxt
        shown = ", ".join(sorted(st.name for st in current)) or "(空=全部赌死)"
        print(f"  读入 '{ch}' (类别:{cat.name}) -> 活跃集 {{{shown}}}")

    return len(current & accept_states) > 0


if __name__ == "__main__":
    test_cases = [
        ("ababababababaab", True),
        ("aba", True),
        ("abb", True),
        ("aaa", True),
        ("baab", True),
        ("bbabb", True),
        ("", False),
        ("a", False),
        ("ab", False),
        ("ba", False),
        ("bb", False),
        ("bab", False),
        ("abab", False),
        ("ababxababab", False),  # OTHER -> 全部赌死
    ]

    print("=" * 40)
    for s, expected in test_cases:
        print(f"\n输入: '{s}'")
        result = simulate(s)
        status = "✓" if result == expected else "✗ BUG!"
        print(f"  结果: {result}, 期望: {expected}  {status}")

    print("\n" + "=" * 40)
    print(f"这个 NFA 用了 {len(State)} 个状态识别'倒数第3个是a'（含显式 DEAD 态）。")
    print("对照 level4_dfa.py: 同一个语言的 DFA 要 16 个状态。")
    print("k 从 3 加到 10: DFA 要 2^10≈1024 个状态, NFA 还是 11 个(+DEAD) —— 这就是必要性。")
