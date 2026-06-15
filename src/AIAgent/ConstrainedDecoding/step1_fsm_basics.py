"""
第一步：理解有限状态机（FSM）—— 约束解码的核心数据结构

约束解码要回答一个问题：
    "已经生成了这些字符，下一个字符只能是哪些？"

FSM 就是回答这个问题的工具。它的工作方式：
    - 有若干个「状态」
    - 有「转移规则」：在状态 A 收到字符 x → 跳到状态 B
    - 有「接受状态」：到了这些状态说明输入合法
    - 关键能力：在任意状态，能列出"哪些字符能让我合法转移"

我们先用最简单的例子建立直觉，再逐步升级到 JSON。

运行方式：
    cd /Users/slyh/MyDir/Project/LLM
    python -m src.AIAgent.ConstrainedDecoding.step1_fsm_basics
"""


# ============================================================
# Part 1：FSM 类 —— 非常小，但就是约束解码的骨架
# ============================================================


class FSM:
    """
    确定性有限状态机（DFA）。

    核心就三样东西：
        transitions:   {(状态, 字符) → 下一个状态}  的映射表
        start_state:   起点
        accept_states: 终点集合（到了这些状态 = 输入合法）
    """

    def __init__(self, start_state, accept_states: set):
        self.transitions: dict[tuple[str, str], str] = {}
        self.start_state = start_state
        self.accept_states = accept_states
        self.current_state = start_state

    def add_transition(self, from_state: str, char: str, to_state: str):
        """添加一条转移规则：在 from_state 收到 char → 跳到 to_state"""
        self.transitions[(from_state, char)] = to_state

    def reset(self):
        """回到起点"""
        self.current_state = self.start_state

    def step(self, char: str) -> bool:
        """输入一个字符，走一步。返回是否成功转移。"""
        key = (self.current_state, char)
        if key in self.transitions:
            self.current_state = self.transitions[key]
            return True
        return False  # 没有这条转移 = 非法输入

    def is_accepted(self) -> bool:
        """当前状态是否是接受状态？"""
        return self.current_state in self.accept_states

    def get_valid_chars(self) -> set[str]:
        """
        ★ 约束解码的核心方法 ★

        返回当前状态下所有合法的下一个字符。
        在真正的约束解码中，这个集合会被用来 mask 模型的 logits：
            - 合法字符对应的 token → 保留概率
            - 不在集合里的 token → 概率置零
        """
        valid = set()
        for (state, char) in self.transitions:
            if state == self.current_state:
                valid.add(char)
        return valid


# ============================================================
# Part 2：最简单的例子 —— 识别整数 [0-9]+
# ============================================================
#
# 状态图：
#
#   ┌───────┐   0-9    ┌─────────┐
#   │  q0   │ ───────→ │   q1    │ ─┐
#   │ start │          │ accept  │  │ 0-9（自环）
#   └───────┘          └─────────┘ ←┘
#
# q0: 还没见到数字（起始状态，不是接受状态）
# q1: 已经见到至少一个数字（接受状态）
#
# 转移规则就两条：
#   q0 + 数字 → q1   （开始）
#   q1 + 数字 → q1   （继续）


def build_integer_fsm() -> FSM:
    """构建一个识别「一个或多个数字」的状态机。"""
    fsm = FSM(start_state="q0", accept_states={"q1"})
    for d in "0123456789":
        fsm.add_transition("q0", d, "q1")  # 第一个数字
        fsm.add_transition("q1", d, "q1")  # 后续数字
    return fsm


def demo_integer():
    print("=" * 60)
    print("Demo 1：识别整数 [0-9]+")
    print("=" * 60)

    fsm = build_integer_fsm()

    # --- 合法输入 ---
    test = "42"
    fsm.reset()
    print(f"\n输入: '{test}'")
    for ch in test:
        valid = fsm.get_valid_chars()
        print(f"  状态={fsm.current_state}  合法下一字符={sorted(valid)}  输入='{ch}'  → ", end="")
        fsm.step(ch)
        print(f"到达 {fsm.current_state}")
    print(f"  → 最终状态={fsm.current_state}  接受={fsm.is_accepted()} ✅")

    # --- 非法输入 ---
    test = "4a2"
    fsm.reset()
    print(f"\n输入: '{test}'")
    for ch in test:
        valid = fsm.get_valid_chars()
        print(f"  状态={fsm.current_state}  合法下一字符={sorted(valid)}  输入='{ch}'  → ", end="")
        ok = fsm.step(ch)
        if ok:
            print(f"到达 {fsm.current_state}")
        else:
            print(f"❌ 非法！'a' 不在合法集合里 → 约束解码会阻止它被采样")
            break


# ============================================================
# Part 3：升级 —— 识别 {"key": "value"} 格式
# ============================================================
#
# 我们来造一个能识别简化版 JSON 的状态机：
#     {"name": "任意小写字母"}
#
# 这比整数复杂，但原理完全一样 —— 多几个状态和转移而已。
#
# 状态图（手动构建）：
#
#   q0  ──{──→  q1  ──"──→  q2  ──n──→  q3  ──a──→  q4  ──m──→  q5
#                                                                 │
#   q5  ──e──→  q6  ──"──→  q7  ──:──→  q8  ── ──→  q9  ──"──→  q10
#                                                                 │
#   q10 ──a-z──→ q10（自环，接收任意小写字母）                       │
#   q10 ──"──→  q11  ──}──→  q12（接受状态！）
#
# 也就是说，这个状态机只接受类似这样的字符串：
#     {"name": "hello"}
#     {"name": "abc"}
# 不接受其他任何东西。


def build_json_kv_fsm() -> FSM:
    """构建识别 {"name": "<小写字母+>"} 的状态机。"""
    fsm = FSM(start_state="q0", accept_states={"q12"})

    # {"name": "value"}
    # 每个字符对应一次状态转移
    fsm.add_transition("q0", "{", "q1")
    fsm.add_transition("q1", '"', "q2")
    fsm.add_transition("q2", "n", "q3")
    fsm.add_transition("q3", "a", "q4")
    fsm.add_transition("q4", "m", "q5")
    fsm.add_transition("q5", "e", "q6")
    fsm.add_transition("q6", '"', "q7")
    fsm.add_transition("q7", ":", "q8")
    fsm.add_transition("q8", " ", "q9")
    fsm.add_transition("q9", '"', "q10")

    # q10：在 value 的引号内部，接受任意小写字母（自环）
    for ch in "abcdefghijklmnopqrstuvwxyz":
        fsm.add_transition("q10", ch, "q10")

    # 关闭引号和花括号
    fsm.add_transition("q10", '"', "q11")
    fsm.add_transition("q11", "}", "q12")

    return fsm


def demo_json_kv():
    print("\n")
    print("=" * 60)
    print('Demo 2：识别 {"name": "<value>"} 格式')
    print("=" * 60)

    fsm = build_json_kv_fsm()

    # --- 合法输入 ---
    test = '{"name": "hello"}'
    fsm.reset()
    print(f"\n输入: {test}")
    for ch in test:
        valid = fsm.get_valid_chars()
        # 合法字符太多时只显示数量
        valid_display = sorted(valid) if len(valid) <= 5 else f"({len(valid)}个字符)"
        print(f"  {fsm.current_state:>4}  合法={valid_display!s:<30}  输入='{ch}'  → ", end="")
        fsm.step(ch)
        print(fsm.current_state)
    print(f"  → 接受={fsm.is_accepted()} ✅")

    # --- 展示约束解码的关键时刻 ---
    print(f"\n--- 约束解码的关键时刻 ---")
    fsm.reset()
    # 走到 q10（value 内部）
    for ch in '{"name": "':
        fsm.step(ch)

    print(f'\n已生成: \'{{"name": "\' → 当前状态: {fsm.current_state}')
    valid = fsm.get_valid_chars()
    print(f"合法下一字符: {sorted(valid)}")
    print(f"→ 只有小写字母和 '\"' 是合法的")
    print(f"→ 如果模型想输出数字、大括号、换行...全部会被 mask 掉！")
    print(f"→ 这就是约束解码：不是检查输出对不对，而是让错误的输出不可能产生")


# ============================================================
# Part 4：思考题（为下一步做准备）
# ============================================================


def show_thinking_questions():
    print("\n")
    print("=" * 60)
    print("思考题（下一步要解决的问题）")
    print("=" * 60)
    print("""
上面的状态机有个巨大的局限：我们是手动构建的。
每个字符一条转移规则，写得又臭又长。

如果 JSON Schema 变了呢？如果 value 可以是数字呢？
如果有嵌套对象呢？不可能每次都手写状态机。

所以 outlines 做的事是：

    JSON Schema → 正则表达式 → 自动生成状态机

比如你的 tool_call schema：
    {"name": string, "arguments": object}

会先变成一个正则（虽然很长很丑）：
    \\{"name":\\s*"[a-z_]+"\\s*,\\s*"arguments":\\s*\\{.*\\}\\}

然后用算法把这个正则编译成状态机。
这样不管 schema 怎么变，状态机都是自动生成的。

下一步我们就来实现这个：正则 → 状态机 的自动转换。
""")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    demo_integer()
    demo_json_kv()
    show_thinking_questions()
