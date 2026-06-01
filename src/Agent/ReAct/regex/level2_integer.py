"""
Level 2: DFA - 识别整数字面量
合法: "123", "0", "-456", "+789"
非法: "", "-", "12.3", "abc"

状态:
  S0 - 初始状态
  S1 - 已读到符号(+/-)
  S2 - 已读到数字 (接受状态)
  DEAD - 死状态(非法输入, 不可恢复)
"""

from enum import Enum, auto


class State(Enum):
    S0 = auto()
    S1 = auto()
    S2 = auto()
    DEAD = auto()


class CharClass(Enum):
    SIGN = auto()
    DIGIT = auto()
    OTHER = auto()

    @classmethod
    def classify(cls, ch: str) -> "CharClass":
        """将字符分类:DFA 只关心字符的'类别',不关心具体字符"""
        if ch in ("+", "-"):
            return cls.SIGN
        elif ch.isdigit():
            return cls.DIGIT
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
            # if to_states[char] not in State:
            if not isinstance(to_states[char], State):
                raise ValueError(
                    f"transition 定义状态 {state} 接收 {char} 时跳转到非法状态 {to_states[char]}"
                )


# 状态转移表
transition = {
    State.S0: {
        CharClass.SIGN: State.S1,
        CharClass.DIGIT: State.S2,
        CharClass.OTHER: State.DEAD,
    },
    State.S1: {
        CharClass.SIGN: State.DEAD,
        CharClass.DIGIT: State.S2,
        CharClass.OTHER: State.DEAD,
    },
    State.S2: {
        CharClass.SIGN: State.DEAD,
        CharClass.DIGIT: State.S2,
        CharClass.OTHER: State.DEAD,
    },
    State.DEAD: {
        CharClass.SIGN: State.DEAD,
        CharClass.DIGIT: State.DEAD,
        CharClass.OTHER: State.DEAD,
    },
}
initial_state = State.S0
accept_states = {State.S2}
validate_dfa(initial_state, accept_states, transition)


def run_dfa(s):
    state = initial_state

    for ch in s:
        cat = CharClass.classify(ch)
        state = transition[state][cat]
        print(f"  读入 '{ch}' (类别:{cat}) -> 状态 {state}")

    return state in accept_states


# 测试
test_cases = [
    ("123", True),
    ("0", True),
    ("-456", True),
    ("+789", True),
    ("", False),
    ("-", False),
    ("12.3", False),
    ("abc", False),
    ("--5", False),
    ("+", False),
    ("42abc", False),
]

print("=" * 40)
for s, expected in test_cases:
    print(f"\n输入: '{s}'")
    result = run_dfa(s)
    status = "✓" if result == expected else "✗ BUG!"
    print(f"  结果: {result}, 期望: {expected}  {status}")
