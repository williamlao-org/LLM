"""
照 DFA 样式手写 a*  (零个或多个 a, 匹配整串)
合法: "" "a" "aa" "aaa"
非法: "b" "ab" "ba" "aab" "x"

状态:
  S0 - 初始 & 接受态(空串就接受; 读到 a 自环留在 S0)
  DEAD - 读到非 a 字符, 死
"""

from enum import Enum, auto


class State(Enum):
    S0 = auto()
    DEAD = auto()


class CharClass(Enum):
    A = auto()
    OTHER = auto()

    @classmethod
    def classify(cls, ch: str) -> "CharClass":
        if ch == "a":
            return cls.A
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
    State.S0: {CharClass.A: State.S0, CharClass.OTHER: State.DEAD},
    State.DEAD: {CharClass.A: State.DEAD, CharClass.OTHER: State.DEAD},
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
    ("a", True),
    ("aa", True),
    ("aaa", True),
    ("b", False),
    ("ab", False),
    ("ba", False),
    ("aab", False),
    ("x", False),
]

print("=" * 40)
for s, expected in test_cases:
    print(f"\n输入: '{s}'")
    result = run_dfa(s)
    status = "✓" if result == expected else "✗ BUG!"
    print(f"  结果: {result}, 期望: {expected}  {status}")
