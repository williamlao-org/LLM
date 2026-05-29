"""
Level 1: DFA - 判断二进制字符串能否被3整除
状态: 0, 1, 2 (代表余数)
初始状态: 0
接受状态: {0}
"""

from enum import Enum


class State(Enum):
    S0 = 0
    S1 = 1
    S2 = 2


# 元素
class Character(Enum):
    ZERO = "0"
    ONE = "1"


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
accept_states = {State.S0}
transition = {
    State.S0: {Character.ZERO: State.S0, Character.ONE: State.S1},
    State.S1: {Character.ZERO: State.S2, Character.ONE: State.S0},
    State.S2: {Character.ZERO: State.S1, Character.ONE: State.S2},
}

validate_dfa(initial_state, accept_states, transition)


def run_dfa(binary_str):
    state = initial_state

    for ch in binary_str:
        try:
            ch = Character(ch)
        except ValueError as e:
            return False, e

        state = transition[state][ch]
        print(f"  读入 '{ch}' -> 状态 {state}")

    # 状态0是接受状态(余数为0, 能被3整除)
    return state in accept_states, state


# 测试
test_cases = [
    "0",  # 0, 能整除
    "11",  # 3, 能整除
    "110",  # 6, 能整除
    "1001",  # 9, 能整除
    "10",  # 2, 不能
    "101",  # 5, 不能
    "111",  # 7, 不能
    "11003",  # 12, 能整除
]

for binary in test_cases:
    # print(f"\n输入: {binary} (十进制: {int(binary, 2)})")
    accepted, final_state = run_dfa(binary)
    print(
        f"  最终状态: {final_state}, {'✓ 能被3整除' if accepted else '✗ 不能被3整除'}"
    )
