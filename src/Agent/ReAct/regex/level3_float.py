"""
Level 3: DFA - 识别浮点数
合法: "123"  "-123"  "0"  "12.3"  "-12.3"  "0.5"
非法: ""  "-"  "12."  ".5"  "12.3.4"  "abc"  "1-2"

状态:
  S0 - 初始状态
  S1 - 已读到符号(+/-)
    S2 - 已读到整数位数字
    S3 - 已读到小数点
    S4 - 已读到小数位数字
  DEAD - 死状态(非法输入, 不可恢复)
"""

from enum import Enum, auto


class State(Enum):
    S0 = auto()  # 初始状态，什么都没读
    S1 = auto()  # 读到符号位
    S2 = auto()  # 读到整数位
    S3 = auto()  # 读到小数点
    S4 = auto()  # 读到小数位
    DEAD = auto()  # 死状态


# 元素
class Character(Enum):
    SIGN = auto()
    DIGIT = auto()
    DOT = auto()
    OTHER = auto()

    @classmethod
    def classify(cls, ch: str):
        if ch in ("+", "-"):
            return cls.SIGN
        if ch.isdigit():
            return cls.DIGIT
        if ch == ".":
            return cls.DOT
        return cls.OTHER


def validate_dfa(
    initial_state: State,
    accept_states: set[State],
    transition: dict[State, dict[Character, State]],
):
    if initial_state not in State:
        raise ValueError("初始状态不合法")

    for accept_state in accept_states:
        if accept_state not in State:
            raise ValueError(f"接收状态 {accept_state} 不合法")

    for state in State:
        if state not in transition.keys():
            raise ValueError(f"transition 没有定义状态 {state}")

        for char in Character:
            to_states = transition[state]
            if char not in to_states.keys():
                raise ValueError(f"transition 没有定义状态 {state} 接收 {char} 的去向")
            # if to_states[char] not in State:
            if not isinstance(to_states[char], State):
                raise ValueError(
                    f"transition 定义状态 {state} 接收 {char} 时跳转到非法状态 {to_states[char]}"
                )


# 状态转移表
# transition[当前状态][输入字符] = 下一个状态
initial_state = State.S0
accept_states = {State.S2, State.S4}
transition = {
    State.S0: {  # 什么都没读
        Character.SIGN: State.S1,
        Character.DIGIT: State.S2,
        Character.DOT: State.DEAD,
        Character.OTHER: State.DEAD,
    },
    State.S1: {  # 读到符号
        Character.SIGN: State.DEAD,
        Character.DIGIT: State.S2,
        Character.DOT: State.DEAD,
        Character.OTHER: State.DEAD,
    },
    State.S2: {  # 读到整数
        Character.SIGN: State.DEAD,
        Character.DIGIT: State.S2,
        Character.DOT: State.S3,
        Character.OTHER: State.DEAD,
    },
    State.S3: {  # 读到小数点
        Character.SIGN: State.DEAD,
        Character.DIGIT: State.S4,
        Character.DOT: State.DEAD,
        Character.OTHER: State.DEAD,
    },
    State.S4: {  # 读到小数位
        Character.SIGN: State.DEAD,
        Character.DIGIT: State.S4,
        Character.DOT: State.DEAD,
        Character.OTHER: State.DEAD,
    },
    State.DEAD: {
        Character.SIGN: State.DEAD,
        Character.DIGIT: State.DEAD,
        Character.DOT: State.DEAD,
        Character.OTHER: State.DEAD,
    },
}

validate_dfa(initial_state, accept_states, transition)


def run_dfa(binary_str):
    state = initial_state

    for ch in binary_str:
        try:
            ch = Character.classify(ch)
        except ValueError as e:
            return False, e

        state = transition[state][ch]
        print(f"  读入 '{ch}' -> 状态 {state}")

    # 状态0是接受状态(余数为0, 能被3整除)
    return state in accept_states


# 测试
test_cases = [
    ("123", True),
    ("-123", True),
    ("0", True),
    ("12.3", True),
    ("-12.3", True),
    ("0.5", True),
    ("", False),
    ("-", False),
    ("12.", False),
    (".5", False),
    ("12.3.4", False),
    ("abc", False),
    ("1-2", False),
]

print("=" * 40)
for s, expected in test_cases:
    print(f"\n输入: '{s}'")
    result = run_dfa(s)
    status = "✓" if result == expected else "✗ BUG!"
    print(f"  结果: {result}, 期望: {expected}  {status}")
