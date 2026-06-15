"""
第二步：正则表达式 → 自动生成状态机

上一步我们手写了 13 个状态才搞定 {"name": "xxx"}。
这一步我们让机器自动做这件事。

整个流水线：

    正则表达式（字符串）
        ↓  解析器（Parser）
    抽象语法树（AST）
        ↓  Thompson 构造法
    NFA（非确定性有限自动机）
        ↓  子集构造法
    DFA（确定性有限自动机）= 上一步的 FSM

每一步做什么、为什么需要，代码里都有详细注释。

运行方式：
    cd /Users/slyh/MyDir/Project/LLM
    uv run python -m src.AIAgent.ConstrainedDecoding.step2_regex_to_fsm
"""

from __future__ import annotations

from dataclasses import dataclass
from .step1_fsm_basics import FSM


# ============================================================
# Part 1：正则表达式的 AST（抽象语法树）
# ============================================================
#
# 正则表达式本质上是一棵树。比如 ab|c 的意思是：
#
#       Alt(|)
#      /      \
#   Concat    Literal('c')
#   /    \
# Lit('a') Lit('b')
#
# 我们先定义这棵树的节点类型，后面 Parser 负责把字符串变成树，
# Thompson 构造法负责把树变成 NFA。


@dataclass
class Literal:
    """单个字符。比如正则里的 a、b、1"""

    char: str


@dataclass
class CharClass:
    """字符类。比如 [a-z] = {'a','b',...,'z'}"""

    chars: set[str]


@dataclass
class Concat:
    """连接。ab = 先匹配 a 再匹配 b"""

    left: RegexNode
    right: RegexNode


@dataclass
class Alt:
    """选择。a|b = 匹配 a 或 b"""

    left: RegexNode
    right: RegexNode


@dataclass
class Star:
    """零或多次。a* = 匹配 a 零次或多次"""

    child: RegexNode


@dataclass
class Plus:
    """一或多次。a+ = 匹配 a 至少一次"""

    child: RegexNode


@dataclass
class Optional:
    """零或一次。a? = 匹配 a 或什么都不匹配"""

    child: RegexNode


# 所有节点类型的联合
RegexNode = Literal | CharClass | Concat | Alt | Star | Plus | Optional


# ============================================================
# Part 2：正则表达式解析器（字符串 → AST）
# ============================================================
#
# 这是一个递归下降解析器。它按运算符优先级从低到高递归：
#
#   优先级（低→高）：
#     | （选择）      →  parse_alternation
#     连接（ab）      →  parse_concatenation
#     * + ?（量词）   →  parse_quantified
#     原子（字符/组） →  parse_atom
#
# 比如 a|bc+ 会解析成：
#   Alt( Literal('a'), Concat( Literal('b'), Plus(Literal('c')) ) )
#
# 优先级保证了 + 先绑定 c，然后 bc+ 连接，最后 | 把两边分开。


class RegexParser:
    """把正则表达式字符串解析成 AST。"""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0

    def peek(self) -> str | None:
        """看当前字符但不移动指针。"""
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def advance(self) -> str:
        """消费当前字符并返回。"""
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def parse(self) -> RegexNode:
        """入口：解析整个正则表达式。"""
        node = self._parse_alternation()
        if self.pos != len(self.pattern):
            raise ValueError(f"解析未完成，第 {self.pos} 位意外字符: '{self.peek()}'")
        return node

    def _parse_alternation(self) -> RegexNode:
        """解析 A|B|C...（最低优先级）"""
        left = self._parse_concatenation()
        while self.peek() == "|":
            self.advance()  # 吃掉 '|'
            right = self._parse_concatenation()
            left = Alt(left, right)
        return left

    def _parse_concatenation(self) -> RegexNode:
        """解析 AB（隐式连接，没有运算符）

        什么时候停？遇到 | 或 ) 或字符串结束。
        """
        nodes: list[RegexNode] = []
        while self.peek() is not None and self.peek() not in "|)":
            nodes.append(self._parse_quantified())

        if not nodes:
            raise ValueError(f"空表达式，位置 {self.pos}")

        # 把 [A, B, C] 折叠成 Concat(Concat(A, B), C)
        result = nodes[0]
        for node in nodes[1:]:
            result = Concat(result, node)
        return result

    def _parse_quantified(self) -> RegexNode:
        """解析 A* / A+ / A?（量词绑定到前面的原子上）"""
        node = self._parse_atom()
        ch = self.peek()
        if ch == "*":
            self.advance()
            return Star(node)
        elif ch == "+":
            self.advance()
            return Plus(node)
        elif ch == "?":
            self.advance()
            return Optional(node)
        return node

    def _parse_atom(self) -> RegexNode:
        """解析原子：字符、[字符类]、(分组)、转义"""
        ch = self.peek()

        if ch == "(":
            # 分组：吃掉 (，递归解析到 )
            self.advance()
            node = self._parse_alternation()
            if self.peek() != ")":
                raise ValueError(f"括号不匹配，位置 {self.pos}")
            self.advance()  # 吃掉 ')'
            return node

        elif ch == "[":
            return self._parse_char_class()

        elif ch == "\\":
            # 转义字符：\{ → Literal('{'), \n → Literal('n')
            self.advance()  # 吃掉 '\'
            escaped = self.advance()
            return Literal(escaped)

        else:
            # 普通字符
            self.advance()
            return Literal(ch)

    def _parse_char_class(self) -> RegexNode:
        """解析字符类 [a-z]、[0-9]、[abc] 等"""
        self.advance()  # 吃掉 '['
        chars: set[str] = set()

        while self.peek() != "]":
            start = self.advance()

            if self.peek() == "-" and self.pos + 1 < len(self.pattern):
                # 范围：a-z
                self.advance()  # 吃掉 '-'
                end = self.advance()
                for code in range(ord(start), ord(end) + 1):
                    chars.add(chr(code))
            else:
                # 单个字符
                chars.add(start)

        self.advance()  # 吃掉 ']'
        return CharClass(chars)


# ============================================================
# Part 3：NFA（非确定性有限自动机）
# ============================================================
#
# NFA 和上一步的 DFA/FSM 有两个关键区别：
#
#   1. 同一个 (状态, 字符) 可以转移到【多个】状态
#      DFA：(q0, 'a') → q1           （只有一个目标）
#      NFA：(q0, 'a') → {q1, q3}     （可以同时去多个地方）
#
#   2. 有 ε（epsilon）转移 —— 不消费任何字符就能跳转
#      (q0, ε) → q1  意思是"不读任何输入，直接从 q0 跳到 q1"
#
# 为什么需要 NFA？因为 Thompson 构造法产出的就是 NFA。
# 正则的 | 运算天然需要"同时尝试两条路"，这就是非确定性。
# NFA 之后再用子集构造法转成 DFA，才能用于实际的约束解码。


class NFA:
    """非确定性有限自动机。ε-转移用 char=None 表示。"""

    def __init__(self):
        self._next_id = 0
        self.transitions: dict[tuple[int, str | None], set[int]] = {}
        self.start: int = -1
        self.accept: int = -1

    def new_state(self) -> int:
        """分配一个新状态编号。"""
        sid = self._next_id
        self._next_id += 1
        return sid

    def add_transition(self, from_s: int, char: str | None, to_s: int):
        """添加转移。char=None 表示 ε-转移。"""
        key = (from_s, char)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add(to_s)

    def epsilon_closure(self, states: set[int]) -> frozenset[int]:
        """计算 ε-闭包：从 states 出发，只走 ε-转移能到达的所有状态。

        为什么需要这个？因为 NFA 可以不读任何输入就跳状态。
        所以"当前在状态 q0"其实意味着"当前在 q0 以及 q0 通过 ε 能到的所有状态"。
        """
        stack = list(states)
        closure = set(states)
        while stack:
            s = stack.pop()
            for next_s in self.transitions.get((s, None), set()):
                if next_s not in closure:
                    closure.add(next_s)
                    stack.append(next_s)
        return frozenset(closure)

    @property
    def alphabet(self) -> set[str]:
        """收集这个 NFA 用到的所有字符（不含 ε）。"""
        return {ch for (_, ch) in self.transitions if ch is not None}


# ============================================================
# Part 4：Thompson 构造法（AST → NFA）
# ============================================================
#
# 这是 Ken Thompson 在 1968 年发明的算法。核心思想：
#
#   对 AST 的每种节点类型，构造一个小 NFA 片段：
#   - 每个片段有且仅有一个入口（start）和一个出口（accept）
#   - 用 ε-转移把片段们拼接起来
#
# 五种基本构造（画成图最直观）：
#
# ① Literal 'c':
#       (s) ──c──→ (a)
#
# ② CharClass [a-z]:
#       (s) ──a──→ (a)
#       (s) ──b──→ (a)     每个字符一条转移，共享起点和终点
#       (s) ──c──→ (a)
#       ...
#
# ③ Concat AB:
#       (A.s) ──...──→ (A.a) ──ε──→ (B.s) ──...──→ (B.a)
#       A 的出口通过 ε 连到 B 的入口
#
# ④ Alt A|B:
#             ε → (A.s) ──...──→ (A.a) ── ε
#       (s) ─┤                              ├──→ (a)
#             ε → (B.s) ──...──→ (B.a) ── ε
#       新起点 ε-分叉到两条路，两条路 ε-汇合到新终点
#
# ⑤ Star A*:
#                    ε（跳过，零次）
#       (s) ──────────────────────────────→ (a)
#        │                                   ↑
#        └─ε─→ (A.s) ──...──→ (A.a) ──ε──→──┤
#                               │            │
#                               └──ε──→ (A.s)  （循环，再来一次）
#
# Plus A+ 和 Optional A? 是 Star 的变体（去掉跳过或去掉循环）。


def thompson_build(node: RegexNode, nfa: NFA) -> tuple[int, int]:
    """递归地把 AST 节点编译成 NFA 片段，返回 (start, accept)。"""

    # ① 单字符
    if isinstance(node, Literal):
        s = nfa.new_state()
        a = nfa.new_state()
        nfa.add_transition(s, node.char, a)
        return s, a

    # ② 字符类
    if isinstance(node, CharClass):
        s = nfa.new_state()
        a = nfa.new_state()
        for ch in node.chars:
            nfa.add_transition(s, ch, a)
        return s, a

    # ③ 连接 AB
    if isinstance(node, Concat):
        s1, a1 = thompson_build(node.left, nfa)
        s2, a2 = thompson_build(node.right, nfa)
        nfa.add_transition(a1, None, s2)  # A 出口 ε→ B 入口
        return s1, a2

    # ④ 选择 A|B
    if isinstance(node, Alt):
        s = nfa.new_state()
        a = nfa.new_state()
        s1, a1 = thompson_build(node.left, nfa)
        s2, a2 = thompson_build(node.right, nfa)
        nfa.add_transition(s, None, s1)  # 分叉
        nfa.add_transition(s, None, s2)
        nfa.add_transition(a1, None, a)  # 汇合
        nfa.add_transition(a2, None, a)
        return s, a

    # ⑤ 零或多次 A*
    if isinstance(node, Star):
        s = nfa.new_state()
        a = nfa.new_state()
        cs, ca = thompson_build(node.child, nfa)
        nfa.add_transition(s, None, cs)  # 进入子片段
        nfa.add_transition(s, None, a)  # 跳过（零次）
        nfa.add_transition(ca, None, cs)  # 循环（再来一次）
        nfa.add_transition(ca, None, a)  # 退出
        return s, a

    # ⑥ 一或多次 A+（和 A* 的唯一区别：没有"跳过"那条 ε）
    if isinstance(node, Plus):
        s = nfa.new_state()
        a = nfa.new_state()
        cs, ca = thompson_build(node.child, nfa)
        nfa.add_transition(s, None, cs)  # 进入（必须至少走一次）
        # 没有 s → a 的 ε ← 这就是和 Star 的区别！
        nfa.add_transition(ca, None, cs)  # 循环
        nfa.add_transition(ca, None, a)  # 退出
        return s, a

    # ⑦ 零或一次 A?（和 A* 的区别：没有"循环"那条 ε）
    if isinstance(node, Optional):
        s = nfa.new_state()
        a = nfa.new_state()
        cs, ca = thompson_build(node.child, nfa)
        nfa.add_transition(s, None, cs)  # 进入
        nfa.add_transition(s, None, a)  # 跳过
        # 没有 ca → cs 的 ε ← 不循环
        nfa.add_transition(ca, None, a)  # 退出
        return s, a

    raise TypeError(f"未知节点类型: {type(node)}")


def regex_to_nfa(pattern: str) -> NFA:
    """正则表达式 → NFA（解析 + Thompson 构造的组合）"""
    ast = RegexParser(pattern).parse()
    nfa = NFA()
    start, accept = thompson_build(ast, nfa)
    nfa.start = start
    nfa.accept = accept
    return nfa


# ============================================================
# Part 5：子集构造法（NFA → DFA）
# ============================================================
#
# NFA 不能直接用来做约束解码，因为它是"非确定性"的：
# 一个输入字符可能同时去多个状态。我们需要 DFA（确定性的）。
#
# 子集构造法的核心思想：
#
#   DFA 的一个状态 = NFA 状态的一个集合
#
#   直觉：NFA 可以"同时在多个状态"（因为 ε 和非确定性），
#   那我们就用一个集合来表示"NFA 当前可能在的所有状态"，
#   这个集合本身就是 DFA 的一个状态。
#
# 步骤：
#   1. DFA 起始状态 = NFA 起始状态的 ε-闭包（一个集合）
#   2. 对每个 DFA 状态（一个集合），对每个字符 c：
#      - 看集合里每个 NFA 状态走 c 能到哪些 NFA 状态
#      - 对到达的状态求 ε-闭包
#      - 这个新集合 = 一个新的（或已有的）DFA 状态
#   3. 如果 DFA 状态（集合）包含 NFA 的接受状态 → 它也是 DFA 的接受状态
#   4. 重复直到没有新状态产生


def nfa_to_dfa(nfa: NFA) -> FSM:
    """子集构造法：NFA → DFA（复用 step1 的 FSM 类）。"""

    # DFA 起始状态 = NFA 起始状态的 ε-闭包
    start_closure = nfa.epsilon_closure({nfa.start})

    # 用整数给 DFA 状态编号，映射表记录 frozenset → 编号
    state_map: dict[frozenset[int], int] = {start_closure: 0}
    next_id = 1

    # 收集 NFA 用到的所有字符
    alphabet = nfa.alphabet

    # 用 BFS 探索所有可达的 DFA 状态
    worklist = [start_closure]
    dfa_transitions: list[tuple[int, str, int]] = []
    dfa_accept: set[int] = set()

    # 起始状态就可能是接受状态（比如 A* 可以匹配空串）
    if nfa.accept in start_closure:
        dfa_accept.add(0)

    while worklist:
        current = worklist.pop()
        current_id = state_map[current]

        for ch in sorted(alphabet):  # 排序只是为了确定性输出
            # 对集合里每个 NFA 状态，走字符 ch 能到哪？
            next_nfa_states: set[int] = set()
            for s in current:
                next_nfa_states |= nfa.transitions.get((s, ch), set())

            if not next_nfa_states:
                continue  # 这个字符在当前 DFA 状态下没有转移

            # 对到达的 NFA 状态求 ε-闭包 → 得到新的 DFA 状态
            next_closure = nfa.epsilon_closure(next_nfa_states)

            if next_closure not in state_map:
                state_map[next_closure] = next_id
                worklist.append(next_closure)
                if nfa.accept in next_closure:
                    dfa_accept.add(next_id)
                next_id += 1

            dfa_transitions.append((current_id, ch, state_map[next_closure]))

    # 构建 step1 的 FSM 对象
    fsm = FSM(start_state="q0", accept_states={f"q{i}" for i in dfa_accept})
    for from_id, ch, to_id in dfa_transitions:
        fsm.add_transition(f"q{from_id}", ch, f"q{to_id}")

    return fsm


# ============================================================
# Part 6：组装 —— 一行搞定
# ============================================================


def regex_to_dfa(pattern: str) -> FSM:
    """正则表达式 → DFA，一行搞定。

    这就是 outlines 的核心流水线：
        regex string → parse → AST → Thompson → NFA → subset → DFA
    """
    nfa = regex_to_nfa(pattern)
    dfa = nfa_to_dfa(nfa)
    return dfa


# ============================================================
# Part 7：演示
# ============================================================


def demo_integer():
    """用正则 [0-9]+ 自动生成 FSM，对比 step1 手写的。"""
    print("=" * 60)
    print("Demo 1：自动生成 [0-9]+ 的状态机")
    print("=" * 60)

    fsm = regex_to_dfa("[0-9]+")

    print(f"\n自动生成的 DFA:")
    print(f"  起始状态: {fsm.start_state}")
    print(f"  接受状态: {fsm.accept_states}")
    print(f"  转移数量: {len(fsm.transitions)}")

    # 测试
    for test in ["42", "0", "123", "", "4a"]:
        fsm.reset()
        ok = True
        for ch in test:
            if not fsm.step(ch):
                ok = False
                break
        accepted = ok and fsm.is_accepted()
        mark = "✅" if accepted else "❌"
        print(f"  '{test}' → {mark}")

    # 展示约束解码能力
    fsm.reset()
    print(f"\n约束解码视角：")
    print(f"  初始状态 {fsm.current_state} → 合法字符: {sorted(fsm.get_valid_chars())}")
    fsm.step("7")
    print(
        f"  输入 '7' → 状态 {fsm.current_state} → 合法字符: {sorted(fsm.get_valid_chars())}"
    )
    print(f"  和 step1 手写的完全一样！但这次是自动生成的。")


def demo_json():
    """用正则自动生成 {"name": "xxx"} 的状态机。"""
    print("\n")
    print("=" * 60)
    print('Demo 2：自动生成 {"name": "<value>"} 的状态机')
    print("=" * 60)

    # 注意转义：正则里 \{ 表示字面量 {（因为 { 在某些正则方言里有特殊含义）
    pattern = r'\{"name": "[a-z]+"\}'
    print(f"\n正则表达式: {pattern}")

    fsm = regex_to_dfa(pattern)
    print(f"自动生成的 DFA:")
    print(f"  起始状态: {fsm.start_state}")
    print(f"  接受状态: {fsm.accept_states}")
    print(f"  转移数量: {len(fsm.transitions)}")

    # 测试
    tests = [
        '{"name": "hello"}',  # ✅ 合法
        '{"name": "a"}',  # ✅ 合法
        '{"name": ""}',  # ❌ + 要求至少一个字符
        '{"name": "Hello"}',  # ❌ 大写不行
        '{"age": "hello"}',  # ❌ key 不对
    ]

    print(f"\n测试:")
    for test in tests:
        fsm.reset()
        ok = True
        for ch in test:
            if not fsm.step(ch):
                ok = False
                break
        accepted = ok and fsm.is_accepted()
        mark = "✅" if accepted else "❌"
        print(f"  {test:<30} → {mark}")

    # 逐步展示约束解码
    print(f"\n逐步约束解码（和 step1 对比）:")
    fsm.reset()
    generated = ""
    for ch in '{"name": "he':
        valid = fsm.get_valid_chars()
        count = len(valid)
        if count <= 5:
            valid_str = str(sorted(valid))
        else:
            valid_str = f"({count}个字符)"
        print(f"  已生成: {generated!r:<20} 合法下一字符: {valid_str:<30} → 选 '{ch}'")
        fsm.step(ch)
        generated += ch

    # 展示当前位置（value 内部）的合法字符
    valid = fsm.get_valid_chars()
    print(f"\n  现在在 value 内部，合法字符: {sorted(valid)}")
    print(f'  → a-z 和 "（关闭引号）都合法')
    print(f"  → 数字、空格、换行...全被 mask 掉")
    print(f"  → 和 step1 手写的 q10 状态完全一样！")


def demo_alternation():
    """展示 | 运算的处理：(dog|cat)"""
    print("\n")
    print("=" * 60)
    print("Demo 3：展示选择运算 (dog|cat)")
    print("=" * 60)

    pattern = "(dog|cat)"
    print(f"\n正则: {pattern}")

    fsm = regex_to_dfa(pattern)

    for test in ["dog", "cat", "dot", "ca", "dogs"]:
        fsm.reset()
        ok = True
        for ch in test:
            if not fsm.step(ch):
                ok = False
                break
        accepted = ok and fsm.is_accepted()
        mark = "✅" if accepted else "❌"
        print(f"  '{test}' → {mark}")

    # 约束解码视角
    print(f"\n约束解码视角:")
    fsm.reset()
    print(f"  起始 → 合法字符: {sorted(fsm.get_valid_chars())}")
    print(f"  → 只有 'd' 和 'c' 是合法的（dog 的 d 和 cat 的 c）")

    fsm.step("d")
    print(f"  输入 'd' → 合法字符: {sorted(fsm.get_valid_chars())}")
    print(f"  → 选了 'd' 就只能走 dog 那条路，下一个只能是 'o'")


def demo_pipeline_summary():
    print("\n")
    print("=" * 60)
    print("总结：outlines 的核心流水线")
    print("=" * 60)
    print("""
你刚才跑的代码实现了完整的流水线：

    regex_to_dfa(pattern)
        │
        ├─ RegexParser(pattern).parse()     # 字符串 → AST
        │       解析正则文法，识别 |、*、+、[]、() 等
        │
        ├─ thompson_build(ast, nfa)         # AST → NFA
        │       每种节点造一个小 NFA 片段，用 ε 拼起来
        │
        └─ nfa_to_dfa(nfa)                  # NFA → DFA
                子集构造：DFA 的一个状态 = NFA 状态的集合

最终得到的 DFA 就是 step1 里的 FSM：
    - 确定性的（每个状态+字符只有一个去处）
    - 能回答 get_valid_chars()（约束解码的核心）
    - 而且是自动生成的，不用手写！

下一步（step3）：
    我们把这个 DFA 和一个 tokenizer 的词表结合起来。
    不再是"哪些字符合法"，而是"哪些 token 合法"。
    因为 LLM 不是逐字符生成的，它是逐 token 生成的。
    一个 token 可能包含多个字符（比如 "hello" 可能是一个 token）。
    这就是 outlines 最精妙的一步。
""")


if __name__ == "__main__":
    demo_integer()
    demo_json()
    demo_alternation()
    demo_pipeline_summary()
