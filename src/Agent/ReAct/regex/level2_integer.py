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


def classify(ch):
    """将字符分类，DFA不关心具体字符，只关心字符的'类别'"""
    if ch in ('+', '-'):
        return 'sign'
    elif ch.isdigit():
        return 'digit'
    else:
        return 'other'


# 状态转移表
transition = {
    'S0':   {'sign': 'S1', 'digit': 'S2', 'other': 'DEAD'},
    'S1':   {'sign': 'DEAD', 'digit': 'S2', 'other': 'DEAD'},
    'S2':   {'sign': 'DEAD', 'digit': 'S2', 'other': 'DEAD'},
    'DEAD': {'sign': 'DEAD', 'digit': 'DEAD', 'other': 'DEAD'},
}

accept_states = {'S2'}


def run_dfa(s):
    state = 'S0'

    for ch in s:
        cat = classify(ch)
        state = transition[state][cat]
        print(f"  读入 '{ch}' (类别:{cat}) -> 状态 {state}")

    return state in accept_states


# 测试
test_cases = [
    ("123",   True),
    ("0",     True),
    ("-456",  True),
    ("+789",  True),
    ("",      False),
    ("-",     False),
    ("12.3",  False),
    ("abc",   False),
    ("--5",   False),
    ("+",     False),
    ("42abc", False),
]

print("="*40)
for s, expected in test_cases:
    print(f"\n输入: '{s}'")
    result = run_dfa(s)
    status = "✓" if result == expected else "✗ BUG!"
    print(f"  结果: {result}, 期望: {expected}  {status}")
