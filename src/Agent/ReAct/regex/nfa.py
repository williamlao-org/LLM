# ============================================================
# Thompson NFA —— 第三步：解析器，从字符串自动建图
# ============================================================
#
# 前两步我们已经有了：
#   - State / Frag / patch          → 建图的零件
#   - literal / concat / alternate / star → 拼图的组合子
#   - simulate                      → 跑图的模拟器
#
# 但现在要手写 concat(star(literal('a')), literal('b')) 来表达 "a*b"，
# 太啰嗦了。这一步加一个解析器，让你直接写：
#
#   nfa = compile("a*b")
#   simulate(nfa, "aab")   → True
#
# 解析分三小步：
#   1. 插入显式连接符：  "a*b"  →  "a*.b"    （用 . 表示"串联"）
#   2. 中缀转后缀：      "a*.b" →  "a*b."    （后缀表达式，方便用栈求值）
#   3. 后缀求值：        "a*b." →  用栈调用组合子，建出完整 NFA
# ============================================================

from typing import List, Literal, Optional, Tuple, Union


# ============================================================
# 第一阶段：建图（只跟 pattern 有关，和 text 无关）
# ============================================================

# ---- 节点 ----

SPLIT = -1   # 分裂态：不吃字符，ε 分叉走两条路
MATCH = -2   # 接受态：终点

Symbol = Union[str, int]  # 普通字符，或 SPLIT/MATCH 这种哨兵值

# 一个还没接上的出边位置。
#
# 例如 (node, 'out') 的意思是：
#   node.out 现在还是 None，后面 patch 时会把它接到某个目标 State。
#
# 这里必须同时记录 node 和字段名，因为 Frag 的悬空线头可能在片段内部的多个节点上，
# 不一定在 Frag.start 上。比如 a|b 的两个线头分别是 a.out 和 b.out。
PatchSite = Tuple["State", Literal["out", "out1"]]


class State:
    """
    图里的一个节点。

    三种形态：
    - 字符态：c 是一个字符（如 'a'），表示"要消耗这个字符才能通过"
    - 分裂态：c == SPLIT，不消耗字符，同时走 out 和 out1 两条路
    - 接受态：c == MATCH，终点，没有出边

    类型关系：
        c:    str | int
              普通字符，或者 SPLIT/MATCH
        out:  State | None
              第一条出边。字符态吃完字符后走 out；SPLIT 也会走 out。
        out1: State | None
              第二条出边，只给 SPLIT 用。
    """
    def __init__(
        self,
        c: Symbol,
        out: Optional["State"] = None,
        out1: Optional["State"] = None,
    ):
        self.c: Symbol = c                  # 通行条件：字符，或 SPLIT/MATCH
        self.out: Optional["State"] = out   # 出边1：下一个 State；暂时未知时为 None
        self.out1: Optional["State"] = out1 # 出边2：只有 SPLIT 会用；也可能暂时为 None


# ---- 半成品电路 (Frag) ----

class Frag:
    """
    建图过程中的半成品——一小段图，有入口，但出口还没接上。

    可以把它想成“正在拼的一段 NFA”：
        - State 节点之间已经接好的边，真的存在于 node.out / node.out1 里
        - 还没接好的边，不直接丢掉，而是记在 outs 里，等后面的片段出现后 patch

    属性：
        start: 这段图的入口节点，类型是 State。
               simulate 最后就是从这个节点开始跑。
        outs:  悬空线头列表，每个元素是 (node, 字段名)
               类型是 List[Tuple[State, Literal["out", "out1"]]]
               意思是"node 的这个字段还是 None，等着被焊上"。

    例子：
        literal('a') 会产生：

            [a] -> ???

        对应：
            start = a_node
            outs  = [(a_node, 'out')]
    """
    def __init__(self, start: State, outs: List[PatchSite]):
        self.start: State = start
        self.outs: List[PatchSite] = outs


def patch(outs: List[PatchSite], target: State) -> None:
    """
    焊接：把一堆悬空线头全部接到 target 节点上。

    例如：
        outs = [(a_node, 'out'), (split_node, 'out1')]

    执行 patch(outs, target) 后：
        a_node.out     = target
        split_node.out1 = target
    """
    for node, attr in outs:
        setattr(node, attr, target)


# ---- 基本构件 ----

def literal(c: str) -> Frag:
    """
    单个字符 → Frag。

    刚创建字符节点时，还不知道它后面要接谁，所以 out 先留空：

        [c] -> ???

    这个空着的 out 会被放进 Frag.outs，后面 concat/build 时再补上。
    """
    node = State(c)                        # out 先是 None
    return Frag(node, [(node, 'out')])


# ---- 三个组合子 ----

def concat(f1: Frag, f2: Frag) -> Frag:
    """
    串联：f1 后面接 f2。

    做法：
        把 f1 的所有悬空出口，都接到 f2 的入口。

    例子：
        f1: [a] -> ???
        f2: [b] -> ???

    patch 后：
        [a] -> [b] -> ???
    """
    patch(f1.outs, f2.start)
    return Frag(f1.start, f2.outs)


def alternate(f1: Frag, f2: Frag) -> Frag:
    """
    选择：f1 | f2。

    新建一个 SPLIT 作为入口：

              -> f1.start ... ???
        SPLIT
              -> f2.start ... ???

    两边的出口都还悬着，所以新的 outs 是 f1.outs + f2.outs。
    """
    split = State(SPLIT, f1.start, f2.start)
    return Frag(split, f1.outs + f2.outs)


def star(f: Frag) -> Frag:
    """
    重复：f*（0 次或多次）。

    新建一个 SPLIT：
        - out  进入 f，表示匹配一次
        - out1 跳过 f，表示匹配零次

    f 自己跑完以后，再 patch 回 split，形成循环。
    最后留下 split.out1 作为悬空出口，等后面的片段来接。
    """
    split = State(SPLIT)
    split.out = f.start       # 路1：进入循环体
    patch(f.outs, split)      # 循环体跑完后回到 split
    split.out1 = None         # 路2：跳过循环体，暂时悬空
    return Frag(split, [(split, 'out1')])


def build(frag: Frag) -> State:
    """
    收尾：把所有剩余悬空线头接到 MATCH 终点。

    build 之后，Frag 变成一张完整 NFA，返回入口 State。
    """
    accept = State(MATCH)
    patch(frag.outs, accept)
    return frag.start


# ============================================================
# 第一阶段（续）：解析器 —— 字符串 → 自动调用组合子
# ============================================================

# ---- 第 1 小步：插入显式连接符 '.' ----

def add_concat_dots(pattern):
    """
    正则表达式里，串联是隐式的："ab" 其实是 "a.b"。
    这一步把隐式的串联变成显式的 '.'，方便后面处理。

    例子：
        "ab"   →  "a.b"
        "a*b"  →  "a*.b"
        "a(b"  →  "a.(b"
        "(a)b" →  "(a).b"
        "a|b"  →  "a|b"      （| 两边不加，| 自己就是分隔符）
    """
    result = []
    for i, ch in enumerate(pattern):
        result.append(ch)
        if i + 1 < len(pattern):
            next_ch = pattern[i + 1]
            # 左边是"能产出东西的"：普通字符、)、*
            left_produces  = (ch not in ('(', '|', '.'))
            # 右边是"能接受东西的"：普通字符、(
            right_consumes = (next_ch not in (')', '|', '*', '.'))
            if left_produces and right_consumes:
                result.append('.')
        # 不需要加 . 的情况：
        #   a|b  → | 已经是分隔符
        #   a*   → * 是后缀运算符，紧跟前面的字符
        #   (a   → ( 是开头，不需要和前面连
    return ''.join(result)


# ---- 第 2 小步：中缀 → 后缀（调度场算法） ----

def to_postfix(infix):
    """
    把带 . 和 | 的中缀表达式转成后缀表达式。
    * 是一元后缀运算符，遇到直接输出。

    运算符优先级（高的先算）：
        *   最高（紧贴前面的字符）
        .   中等（串联）
        |   最低（选择）

    例子：
        "a.b"    → "ab."
        "a|b"    → "ab|"
        "a*.b"   → "a*b."
        "a.b|c"  → "ab.c|"

    算法：经典的调度场（Shunting Yard），和数学表达式的中缀转后缀一模一样。
    """
    precedence = {'|': 1, '.': 2}
    output = []
    ops = []       # 运算符栈

    for ch in infix:
        if ch == '(':
            ops.append(ch)

        elif ch == ')':
            # 把栈里的运算符弹出，直到遇到 (
            while ops[-1] != '(':
                output.append(ops.pop())
            ops.pop()   # 弹掉 (

        elif ch == '*':
            # 一元后缀，直接输出（它已经紧跟在操作数后面了）
            output.append(ch)

        elif ch in precedence:
            # 二元运算符：把栈里优先级 >= 自己的先弹出
            while (ops and ops[-1] != '('
                   and ops[-1] in precedence
                   and precedence[ops[-1]] >= precedence[ch]):
                output.append(ops.pop())
            ops.append(ch)

        else:
            # 普通字符（操作数），直接输出
            output.append(ch)

    # 把栈里剩余的运算符全弹出
    while ops:
        output.append(ops.pop())

    return ''.join(output)


# ---- 第 3 小步：后缀求值 —— 用栈调用组合子 ----

def compile(pattern):
    """
    从正则表达式字符串直接建出 NFA，返回起点。

    完整流程：
        "a*b"  →(加点)→  "a*.b"  →(转后缀)→  "a*b."  →(栈求值)→  NFA 起点

    求值规则（和计算器算后缀表达式一模一样）：
        遇到普通字符 → literal(c)，压栈
        遇到 *       → 弹一个，star(它)，结果压回去
        遇到 .       → 弹两个，concat(前, 后)，结果压回去
        遇到 |       → 弹两个，alternate(前, 后)，结果压回去
        最后栈里剩一个 Frag → build 收尾，返回起点
    """
    dotted  = add_concat_dots(pattern)
    postfix = to_postfix(dotted)

    stack = []
    for ch in postfix:
        if ch == '.':
            f2 = stack.pop()
            f1 = stack.pop()
            stack.append(concat(f1, f2))
        elif ch == '|':
            f2 = stack.pop()
            f1 = stack.pop()
            stack.append(alternate(f1, f2))
        elif ch == '*':
            f = stack.pop()
            stack.append(star(f))
        else:
            stack.append(literal(ch))

    return build(stack.pop())


# ============================================================
# 第二阶段：跑图（只跟 text 有关，图已经建好了）
# ============================================================

def add_epsilon(node, state_set):
    """处理 ε 转移：SPLIT 节点不消耗字符，递归展开。"""
    if node is None:
        return
    if node.c == SPLIT:
        add_epsilon(node.out,  state_set)
        add_epsilon(node.out1, state_set)
    else:
        state_set.add(node)


def simulate(start, text):
    """在 NFA 图上跑一个 text，判断是否匹配。"""
    current = set()
    add_epsilon(start, current)

    for ch in text:
        next_states = set()
        for node in current:
            if node.c == ch:
                add_epsilon(node.out, next_states)
        current = next_states

    for node in current:
        if node.c == MATCH:
            return True
    return False


# ============================================================
# 测试：直接用字符串，一步到位
# ============================================================

def test(pattern, text, expected):
    nfa = compile(pattern)
    result = simulate(nfa, text)
    status = "✓" if result == expected else "✗ FAIL"
    print(f'  {status}  pattern="{pattern}"  text="{text}"  => {result}')


print("=== 单个字符 ===")
test("a", "a",  True)
test("a", "b",  False)
test("a", "",   False)

print("\n=== 串联 ===")
test("abc", "abc", True)
test("abc", "ab",  False)
test("abc", "abcd",False)

print("\n=== 选择 | ===")
test("a|b",   "a",  True)
test("a|b",   "b",  True)
test("a|b",   "c",  False)
test("ab|cd",  "ab", True)
test("ab|cd",  "cd", True)
test("ab|cd",  "ac", False)

print("\n=== 重复 * ===")
test("a*",  "",    True)
test("a*",  "aaa", True)
test("a*",  "b",   False)

print("\n=== 混合 ===")
test("a*b",    "b",     True)
test("a*b",    "ab",    True)
test("a*b",    "aaab",  True)
test("a*b",    "aa",    False)
test("a*b|c",  "aaab",  True)
test("a*b|c",  "c",     True)
test("a*b|c",  "d",     False)

print("\n=== 括号 ===")
test("(a|b)*", "",     True)
test("(a|b)*", "abab", True)
test("(a|b)*", "abc",  False)
test("(a|b)c", "ac",   True)
test("(a|b)c", "bc",   True)
test("(a|b)c", "cc",   False)
