"""
Level 4 (对照组): DFA - 识别"倒数第 3 个字符是 a"的串 (字母表 {a, b})
合法: "aab" "aba" "abb" "aaa" "baab" "bbabb"  (倒数第3个 = a)
非法: "" "a" "ab" "ba" "bb" "bab" "abab"       (太短 / 倒数第3个 = b)

======================================================================
看这个文件的唯一目的: 体会 NFA 的必要性。
对照 level4_nfa.py —— 同一个语言,NFA 只用 4 个状态就画完了。

DFA 为什么这么臃肿? 因为 DFA 读到一个 a 时,无法判断它是不是"倒数第3个"
—— 得看后面还剩几个字符。而 DFA 不许"分身去赌",只能把判断所需的信息
全部塞进状态里 == "死记最近 3 个字符"。

  状态 = 最近 3 个字符的窗口 (不足 3 个用 '_' 占位)
  读到新字符 ch: 新窗口 = (旧窗口 + ch)[-3:]   (左移,丢掉最老的)
  接受: 窗口第 0 位 == 'a'  (即倒数第3个是 a)

记 3 个字符 -> 2^3 = 8 个"满窗口"状态,加上开头不足 3 个的过渡态,
本文件一共 16 个状态。把 "3" 换成 "5" 就是 2^5=32 个,换成 10 就是 1024 个
—— 你手画不出来了。这就是 NFA 存在的理由: 指数级地更简洁。
======================================================================
"""

from enum import Enum, auto


class State(Enum):
    # 窗口状态: W 后面跟 3 位,'_' 表示该位还没字符。如 W__a = 只读了1个字符且它是a
    W___ = auto()  # 初始: 一个字符都没读
    W__a = auto()
    W__b = auto()
    W_aa = auto()
    W_ab = auto()
    W_ba = auto()
    W_bb = auto()
    Waaa = auto()  # 以下 8 个是"满窗口",窗口首位是 a 的即接受态
    Waab = auto()
    Waba = auto()
    Wabb = auto()
    Wbaa = auto()
    Wbab = auto()
    Wbba = auto()
    Wbbb = auto()
    DEAD = auto()  # 读到 a/b 以外的字符


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


# 状态转移表 —— 每行就是"窗口左移一格"。看它有多长,这就是 NFA 想替你省掉的东西。
S = State
C = CharClass
transition = {
    S.W___: {C.A: S.W__a, C.B: S.W__b, C.OTHER: S.DEAD},
    S.W__a: {C.A: S.W_aa, C.B: S.W_ab, C.OTHER: S.DEAD},
    S.W__b: {C.A: S.W_ba, C.B: S.W_bb, C.OTHER: S.DEAD},
    S.W_aa: {C.A: S.Waaa, C.B: S.Waab, C.OTHER: S.DEAD},
    S.W_ab: {C.A: S.Waba, C.B: S.Wabb, C.OTHER: S.DEAD},
    S.W_ba: {C.A: S.Wbaa, C.B: S.Wbab, C.OTHER: S.DEAD},
    S.W_bb: {C.A: S.Wbba, C.B: S.Wbbb, C.OTHER: S.DEAD},
    S.Waaa: {C.A: S.Waaa, C.B: S.Waab, C.OTHER: S.DEAD},
    S.Waab: {C.A: S.Waba, C.B: S.Wabb, C.OTHER: S.DEAD},
    S.Waba: {C.A: S.Wbaa, C.B: S.Wbab, C.OTHER: S.DEAD},
    S.Wabb: {C.A: S.Wbba, C.B: S.Wbbb, C.OTHER: S.DEAD},
    S.Wbaa: {C.A: S.Waaa, C.B: S.Waab, C.OTHER: S.DEAD},
    S.Wbab: {C.A: S.Waba, C.B: S.Wabb, C.OTHER: S.DEAD},
    S.Wbba: {C.A: S.Wbaa, C.B: S.Wbab, C.OTHER: S.DEAD},
    S.Wbbb: {C.A: S.Wbba, C.B: S.Wbbb, C.OTHER: S.DEAD},
    S.DEAD: {C.A: S.DEAD, C.B: S.DEAD, C.OTHER: S.DEAD},
}
initial_state = State.W___
# 接受态 = 窗口首位是 a (倒数第3个是 a)
accept_states = {State.Waaa, State.Waab, State.Waba, State.Wabb}
validate_dfa(initial_state, accept_states, transition)


def run_dfa(s):
    state = initial_state

    for ch in s:
        cat = CharClass.classify(ch)
        state = transition[state][cat]
        print(f"  读入 '{ch}' (类别:{cat.name}) -> 状态 {state.name}")

    return state in accept_states


# 测试
test_cases = [
    ("aab", True),
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
    ("ababx", False),  # OTHER -> DEAD
]

print("=" * 40)
for s, expected in test_cases:
    print(f"\n输入: '{s}'")
    result = run_dfa(s)
    status = "✓" if result == expected else "✗ BUG!"
    print(f"  结果: {result}, 期望: {expected}  {status}")

print("\n" + "=" * 40)
print(f"这个 DFA 用了 {len(State)} 个状态识别'倒数第3个是a'。")
print("对照 level4_nfa.py: 同类问题的 NFA 只要 4 个状态。")
print("k 从 3 加到 10: DFA 要 2^10≈1024 个状态,NFA 还是十几个 —— 这就是必要性。")
