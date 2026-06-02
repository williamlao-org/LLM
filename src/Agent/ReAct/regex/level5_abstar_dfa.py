"""
照 DFA 样式手写 (ab)*  —— "ab 这个组合重复零次或多次", 匹配整串
合法: "" "ab" "abab" "ababab"
非法: "a" "aba" "abb" "ba" "b" "abx"

状态:
  S0 - 初始 & 接受态(空串/刚读完一组ab; 期待新一组的 'a', 或就此结束)
  S1 - 读了 'a', 正期待 'b' 来凑成一组
  DEAD - 读错了字符, 死

对照单独的 M_ab (识别 "ab"):  S0 ─a→ S1 ─b→ S2*   接受态={S2}
本文件的 (ab)*:               S0*─a→ S1 ─b→ S0    接受态={S0}
                                          ▲────────┘
看出 * 干了什么没有? 它没"在 M_ab 外面包一层", 而是把 M_ab 的接受态 S2
*合并回了起点 S0*(那条 b 边本该去 S2, 现在改指 S0), 并把 S0 改成接受态。
"""

from enum import Enum, auto


class State(Enum):
    S0 = auto()
    S1 = auto()
    DEAD = auto()


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


def validate_dfa(
    initial_state: State,
    accept_states: set[State],
    transition: dict[State, dict[CharClass, State]],
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
            if not isinstance(to_states[char], State):
                raise ValueError(
                    f"transition 定义状态 {state} 接收 {char} 时跳转到非法状态 {to_states[char]}"
                )


transition = {
    State.S0: {CharClass.A: State.S1, CharClass.B: State.DEAD, CharClass.OTHER: State.DEAD},
    State.S1: {CharClass.A: State.DEAD, CharClass.B: State.S0, CharClass.OTHER: State.DEAD},
    State.DEAD: {CharClass.A: State.DEAD, CharClass.B: State.DEAD, CharClass.OTHER: State.DEAD},
}
initial_state = State.S0
accept_states = {State.S0}
validate_dfa(initial_state, accept_states, transition)


def run_dfa(s):
    state = initial_state

    for ch in s:
        cat = CharClass.classify(ch)
        state = transition[state][cat]
        print(f"  读入 '{ch}' (类别:{cat.name}) -> 状态 {state.name}")

    return state in accept_states


test_cases = [
    ("", True),
    ("ab", True),
    ("abab", True),
    ("ababab", True),
    ("a", False),
    ("aba", False),
    ("abb", False),
    ("ba", False),
    ("b", False),
    ("abx", False),
]

print("=" * 40)
for s, expected in test_cases:
    print(f"\n输入: '{s}'")
    result = run_dfa(s)
    status = "✓" if result == expected else "✗ BUG!"
    print(f"  结果: {result}, 期望: {expected}  {status}")
