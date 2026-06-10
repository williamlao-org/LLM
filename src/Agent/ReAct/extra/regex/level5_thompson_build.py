"""
Level 5: 把"造自动机的那套理论"写成代码 —— Thompson 构造 + 子集构造
配套 level4_nfa.py / level4_dfa.py 一起看。

level4 给你的是"手写好的成品机器"(状态固定、能起名 S0/S1)。
本文件给你的是"造机器的工厂":你给它一个正则的积木拼法,它替你把机器焊出来。

核心就一句话,贯穿全文:
  每个"图章函数"(lit/concat/union/star)只做两件事——
    1) new 出几个全新状态
    2) 加几条 ε 边
  它【绝不】去读传进来的子机器(frag)内部长什么样。
  ——这就是"不动脑子":配方是死的,对任意正则都成立。

状态为什么用整数不用 enum?
  手写时状态是固定的、你能起名;机器生成时状态是临时 new 出来的,
  数量事先不知道,用自增整数当 id 最自然。状态本来就是匿名的。

ε 用 None 表示(不吃任何字符就能走的空边)。
"""

EPS = None  # ε: 空边

StateId = int  # 状态就是一个自增整数 id，没有别的含义
Symbol = str | None  # 边上的符号：字符 或 EPS(None)


class Frag:
    """一台正在拼装中的 NFA 片段。Thompson 约定:每个片段恰好一个入口、一个出口。"""
    def __init__(self, start: StateId, accept: StateId):
        self.start = start    # 入口状态 id
        self.accept = accept  # 出口(接受)状态 id


class Builder:
    """积木厂:持有一张【共享】转移表,所有片段的边都加在这里。
    拼装 = 往这张表里加 ε 边,已有的边一个都不改。
    """
    def __init__(self):
        # 按"符号"分桶:trans[st][sym] = 吃 sym 能直接到的状态集合。
        # 查的时候 trans[st].get(ch) 一步拿到,不用扫一遍再逐个比 sym。
        self.trans: dict[StateId, dict[Symbol, set[StateId]]] = {}
        self._next: StateId = 0     # 下一个可用状态 id

    def new_state(self) -> StateId:
        state_id = self._next
        self._next += 1
        self.trans[state_id] = {}
        return state_id

    def add(self, frm: StateId, symbol: Symbol, to: StateId) -> None:
        self.trans[frm].setdefault(symbol, set()).add(to)

    # ───────────────── 图章①:单字符 c   (对应第1张图里的 ◯─c─→◎) ─────────────────
    def lit(self, ch: str) -> Frag:
        start = self.new_state()
        accept = self.new_state()
        self.add(start, ch, accept)  # 唯一一条吃字符的边
        return Frag(start, accept)

    # ───────────────── 图章②:接 (R 然后 S)   (第1张图:把 a 的出口 ε 焊到 b 的入口) ─────────────────
    def concat(self, left: Frag, right: Frag) -> Frag:
        self.add(left.accept, EPS, right.start)   # ★ 全部动作就这一条 ε 焊接
        return Frag(left.start, right.accept)
        # 注意:没有任何一句去看 left 或 right 里面有什么 —— 不动脑子

    # ───────────────── 图章③:或 (R | S)   (第2张图:新起点/新接受 + 4 条 ε) ─────────────────
    def union(self, left: Frag, right: Frag) -> Frag:
        start = self.new_state()
        accept = self.new_state()
        self.add(start, EPS, left.start)     # 新起点 ε 进 R
        self.add(start, EPS, right.start)    # 新起点 ε 进 S
        self.add(left.accept, EPS, accept)   # R 出口 ε 汇到新接受
        self.add(right.accept, EPS, accept)  # S 出口 ε 汇到新接受
        return Frag(start, accept)

    # ───────────────── 图章④:星 (R*)   (第3张图:新起点/新接受 + 4 条 ε) ─────────────────
    def star(self, frag: Frag) -> Frag:
        start = self.new_state()
        accept = self.new_state()
        self.add(start, EPS, frag.start)     # 进去读一遍
        self.add(start, EPS, accept)         # 一次都不读(空串也接受)
        self.add(frag.accept, EPS, frag.start)  # 读完跳回去,再来一遍(循环)
        self.add(frag.accept, EPS, accept)   # 读够了,结束
        return Frag(start, accept)


# ═══════════════ ε-闭包:level4_nfa 没有的那一步 ═══════════════
# level4_nfa 的"活跃集合"只沿吃字符的边走;现在多了 ε,
# 所以每到一处,要先把"不吃字符就能顺着 ε 溜达到的所有状态"也算进活跃集。
def eps_closure(builder: Builder, states: set[StateId]) -> set[StateId]:
    closure = set(states)
    stack = list(states)
    while stack:
        st = stack.pop()
        for to in builder.trans[st].get(EPS, ()):   # 直接拿这一状态的所有 ε 出边
            if to not in closure:
                closure.add(to)
                stack.append(to)
    return closure


def simulate_nfa(builder: Builder, frag: Frag, text: str) -> bool:
    """直接跑 ε-NFA,逻辑和 level4_nfa 一样,只是每步多包一层 eps_closure。"""
    current = eps_closure(builder, {frag.start})
    for ch in text:
        nxt = set()
        for st in current:
            nxt.update(builder.trans[st].get(ch, ()))   # 直接拿吃 ch 能到的状态
        current = eps_closure(builder, nxt)
    return frag.accept in current


# ═══════════════ 子集构造:把 ε-NFA 碾平成 DFA(就是流水线那一步) ═══════════════
# 产物正是 level4_dfa.py 里你手写的那种"一个状态对每个字符只有一个去向"的表,
# 只不过这里是自动生成的。DFA 的一个状态 = ε-NFA 的"一坨活跃集"。
def to_dfa(builder, frag, alphabet):
    start = frozenset(eps_closure(builder, {frag.start}))
    trans = {}
    accept = set()
    seen = {start}
    work = [start]
    while work:
        S = work.pop()
        if frag.accept in S:
            accept.add(S)
        trans[S] = {}
        for ch in alphabet:
            move = set()
            for st in S:
                move.update(builder.trans[st].get(ch, ()))   # 直接拿吃 ch 能到的状态
            T = frozenset(eps_closure(builder, move))   # 空集 = 死状态
            trans[S][ch] = T
            if T not in seen:
                seen.add(T)
                work.append(T)
    return start, accept, trans


def run_dfa(start, accept, trans, s):
    """成品 DFA 的跑法:每个字符就一次查表,O(n),不分身、不回头。"""
    state = start
    for ch in s:
        state = trans[state][ch]
    return state in accept


# ═══════════════════════════ 拼装 (a|ab)* ═══════════════════════════
# 这几行就是我画的那三张图的全部内容,翻译成代码:
builder = Builder()
frag_a_alt    = builder.lit("a")                      # 印一块单独的 a (给"或"的左边)
frag_a_concat = builder.lit("a")                      # 印一块 a (给 ab 用)
frag_b        = builder.lit("b")                      # 印一块 b
frag_ab       = builder.concat(frag_a_concat, frag_b) # 图章②:a 接 b  ->  ab
frag_a_or_ab  = builder.union(frag_a_alt, frag_ab)   # 图章③:a | ab
regex         = builder.star(frag_a_or_ab)            # 图章④:(a|ab)*

ALPHABET = ["a", "b"]

test_cases = [
    ("", True),
    ("ab", True),
    ("a", True),
    ("aab", True),
    ("abab", True),
    ("ba", False),     # 开头是 b
    ("abb", False),    # 出现 bb
    ("b", False),
]


if __name__ == "__main__":
    print("=" * 50)
    print(f"机械焊出来的 ε-NFA 一共 {builder._next} 个状态(又丑又冗余,因为没人优化)")
    print("-" * 50)
    print("【直接跑 ε-NFA】")
    for s, expected in test_cases:
        got = simulate_nfa(builder, regex, s)
        mark = "✓" if got == expected else "✗ BUG!"
        print(f"  '{s:5}' -> {got!s:5} (期望 {expected}) {mark}")

    # 子集构造:碾平成 DFA
    start, accept, dfa = to_dfa(builder, regex, ALPHABET)
    order = list(dfa.keys())
    label = {S: ("DEAD" if len(S) == 0 else f"D{i}") for i, S in enumerate(order)}

    print("=" * 50)
    print(f"子集构造把它碾成 DFA,只剩 {len(order)} 个状态。生成的转移表:")
    print("(这就是 flex/RE2 会自动吐出、而你在 level4_dfa 里手写过的那种表)")
    print("-" * 50)
    for S in order:
        star_mark = " *接受" if S in accept else ""
        start_mark = " <起点" if S == start else ""
        row = "   ".join(f"{ch}->{label[dfa[S][ch]]}" for ch in ALPHABET)
        print(f"  {label[S]:5}: {row}{start_mark}{star_mark}")

    print("-" * 50)
    print("【跑生成的 DFA,应与 ε-NFA 完全一致】")
    for s, expected in test_cases:
        got = run_dfa(start, accept, dfa, s)
        mark = "✓" if got == expected else "✗ BUG!"
        print(f"  '{s:5}' -> {got!s:5} (期望 {expected}) {mark}")

    print("=" * 50)
    print("流水线:正则(积木拼法) →lit/concat/union/star→ ε-NFA →子集构造→ DFA")
    print("每一段都是死板的机械操作,全程不需要你去想任何状态的含义。")
