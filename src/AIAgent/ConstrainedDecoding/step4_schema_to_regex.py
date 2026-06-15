"""
第四步：JSON Schema → 正则表达式

前三步的流水线有一个手工环节：正则表达式是我们自己写的。
    \\{"name": "[a-z]+"\\}   ← 这个是人写的

真正的 outlines 是自动化的：给一个 JSON Schema，自动生成对应的正则。
这一步就来实现这个自动化。

核心思路：递归翻译
    - string  → "[字符类]*"
    - integer → -?[0-9]+
    - boolean → (true|false)
    - enum    → ("值1"|"值2"|...)
    - object  → {"key1": 值1正则, "key2": 值2正则}
    - array   → \\[元素正则(, 元素正则)*\\]

每种类型有自己的正则模板，object 和 array 递归调用子 schema。

运行方式：
    cd /Users/slyh/MyDir/Project/LLM
    uv run python -m src.AIAgent.ConstrainedDecoding.step4_schema_to_regex
"""

from __future__ import annotations

from .step1_fsm_basics import FSM
from .step2_regex_to_fsm import regex_to_dfa
from .step3_token_masking import (
    build_token_mask_table,
    simulate_constrained_generation,
    walk_token,
)


# ============================================================
# Part 1：JSON Schema → 正则表达式（递归翻译）
# ============================================================
#
# 这个函数是 outlines 流水线的"第 0 步"：
#
#   JSON Schema ──→ 正则 ──→ NFA ──→ DFA ──→ Token Mask
#      这一步         step2    step2    step3
#
# 之前我们手写正则，现在让代码从 Schema 自动生成。


def schema_to_regex(schema: dict, depth: int = 0, max_depth: int = 3) -> str:
    """把 JSON Schema 翻译成正则表达式。

    Args:
        schema:    JSON Schema 字典
        depth:     当前递归深度（防止无限嵌套）
        max_depth: 最大递归深度，超过后用宽松模式

    Returns:
        正则表达式字符串，可以直接喂给 step2 的 regex_to_dfa()
    """

    # --- 枚举类型：固定的几个值选一个 ---
    if "enum" in schema:
        return _enum_to_regex(schema["enum"])

    schema_type = schema.get("type")

    # --- 基本类型 ---
    if schema_type == "string":
        return _string_to_regex(schema)

    if schema_type == "integer":
        # -?[0-9]+  表示可选负号 + 一个或多个数字
        return "-?[0-9]+"

    if schema_type == "number":
        # 整数或小数：-?[0-9]+(\\.[0-9]+)?
        # 注意 \\. 在我们的解析器里表示字面量的点
        return "-?[0-9]+(\\.[0-9]+)?"

    if schema_type == "boolean":
        return "(true|false)"

    if schema_type == "null":
        return "null"

    # --- 复合类型 ---
    if schema_type == "object":
        return _object_to_regex(schema, depth, max_depth)

    if schema_type == "array":
        return _array_to_regex(schema, depth, max_depth)

    # 兜底：当作字符串处理
    return '"[a-zA-Z0-9_]*"'


def _enum_to_regex(values: list) -> str:
    """枚举值 → (值1|值2|...) 选择正则。

    比如 enum: ["execute_command", "read_file"]
    → ("execute_command"|"read_file")
    """
    parts: list[str] = []
    for v in values:
        if isinstance(v, str):
            # 字符串枚举带引号：execute_command → "execute_command"
            parts.append(f'"{v}"')
        elif isinstance(v, bool):
            parts.append("true" if v else "false")
        elif isinstance(v, int):
            parts.append(str(v))
        elif v is None:
            parts.append("null")

    if len(parts) == 1:
        return parts[0]
    return "(" + "|".join(parts) + ")"


def _string_to_regex(schema: dict) -> str:
    """字符串类型 → 正则。

    如果 schema 里有 pattern，用它；否则用默认字符集。
    """
    pattern = schema.get("pattern")
    if pattern:
        # 用户自定义 pattern，包上引号
        return f'"{pattern}"'

    # 默认：字母、数字、下划线、空格、点、斜杠、减号
    # 注意：在 [字符类] 里，- 放在最前面避免被当成范围符
    return '"[-a-zA-Z0-9_ ./]*"'


def _object_to_regex(schema: dict, depth: int, max_depth: int) -> str:
    """对象类型 → 正则。

    关键思路：
        properties 里的 key 名是固定的（schema 决定的），
        所以正则里 key 是字面量，value 是递归生成的子正则。

        {"key1": value1_regex, "key2": value2_regex}
    """
    if depth >= max_depth:
        # 超过最大深度，用宽松模式匹配任意简单对象
        return '"[a-zA-Z0-9_ ]*"'

    properties = schema.get("properties", {})

    if not properties:
        return "{}"

    # 每个属性生成 "key": value_regex
    prop_parts: list[str] = []
    for key, value_schema in properties.items():
        value_regex = schema_to_regex(value_schema, depth + 1, max_depth)
        prop_parts.append(f'"{key}": {value_regex}')

    # 用 , (逗号+空格) 连接所有属性
    inner = ", ".join(prop_parts)
    return "{" + inner + "}"


def _array_to_regex(schema: dict, depth: int, max_depth: int) -> str:
    """数组类型 → 正则。

    格式：\\[元素(, 元素)*\\]  或  \\[\\]（空数组）
    注意 [ ] 在正则里要转义成 \\[ \\]，因为我们的解析器把 [ 当字符类起始。
    """
    items_schema = schema.get("items", {"type": "string"})
    item_regex = schema_to_regex(items_schema, depth + 1, max_depth)

    # \\[  →  字面量 [
    # (item(, item)*)?  →  可选的非空内容
    # \\]  →  字面量 ]
    return f"\\[({item_regex}(, {item_regex})*)?\\]"


# ============================================================
# Part 2：给你的 Agent 工具定义 Schema
# ============================================================
#
# 这里模拟你的 ReActMulti agent 的工具定义。
# 实际中这些 schema 来自 Tool.parameters。


EXECUTE_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["execute_command"],
        },
        "arguments": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
        },
    },
}

READ_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["read_file"],
        },
        "arguments": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
        },
    },
}

# 合并版：tool_call 可以是 execute_command 或 read_file
# 简化处理：分别生成正则，再用 | 连接
TOOL_CALL_SCHEMAS = [EXECUTE_COMMAND_SCHEMA, READ_FILE_SCHEMA]


# ============================================================
# Part 3：面向 tool calling 的词表
# ============================================================

TOOL_VOCABULARY: dict[int, str] = {
    # 单字符标点
    0: "{",
    1: "}",
    2: '"',
    3: ":",
    4: " ",
    5: ",",
    # 多字符标点组合
    6: '{"',
    7: '"}',
    8: '": "',
    9: '", "',
    # 工具名
    10: "execute_command",
    11: "read_file",
    12: "web_search",
    # JSON key 名
    13: "name",
    14: "arguments",
    15: "command",
    16: "path",
    # 常见值内容
    17: "ls",
    18: "uname",
    19: "cat",
    20: "pwd",
    21: "a",
    22: "e",
    23: "l",
    24: "s",
    25: "-",
    26: "/",
    27: ".",
    28: "tmp",
    29: "etc",
    30: "home",
    31: "ls -la",
    32: "uname -a",
    33: "/tmp",
    34: "/etc/hosts",
    # 不应出现的 token
    35: "1",
    36: "true",
    37: "\n",
}


# ============================================================
# Part 4：演示
# ============================================================


def demo_schema_to_regex():
    """展示不同 schema 生成的正则。"""
    print("=" * 60)
    print("Demo 1：JSON Schema → 正则表达式")
    print("=" * 60)

    examples = [
        ("string", {"type": "string"}),
        ("integer", {"type": "integer"}),
        ("boolean", {"type": "boolean"}),
        (
            'enum (工具名)',
            {"type": "string", "enum": ["execute_command", "read_file"]},
        ),
        (
            "object (简单)",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        ),
        (
            "array (整数数组)",
            {"type": "array", "items": {"type": "integer"}},
        ),
    ]

    for desc, schema in examples:
        regex = schema_to_regex(schema)
        print(f"\n  {desc}")
        print(f"    Schema: {schema}")
        print(f"    正则:   {regex}")


def demo_tool_call_regex():
    """为你的 execute_command 工具生成正则。"""
    print("\n")
    print("=" * 60)
    print("Demo 2：为 execute_command 工具生成正则")
    print("=" * 60)

    regex = schema_to_regex(EXECUTE_COMMAND_SCHEMA)
    print(f"\n  Schema（简化展示）:")
    print(f'    {{"name": "execute_command", "arguments": {{"command": str}}}}')
    print(f"\n  自动生成的正则:")
    print(f"    {regex}")

    # 构建 DFA 并验证
    print(f"\n  构建 DFA 并验证:")
    fsm = regex_to_dfa(regex)
    tests = [
        '{"name": "execute_command", "arguments": {"command": "ls -la"}}',
        '{"name": "execute_command", "arguments": {"command": "uname -a"}}',
        '{"name": "execute_command", "arguments": {"command": ""}}',
        '{"name": "read_file", "arguments": {"command": "ls"}}',  # 工具名不对
    ]

    for test in tests:
        fsm.reset()
        ok = True
        for ch in test:
            if not fsm.step(ch):
                ok = False
                break
        accepted = ok and fsm.is_accepted()
        mark = "✅" if accepted else "❌"
        short = test if len(test) <= 60 else test[:57] + "..."
        print(f"    {mark} {short}")


def demo_full_pipeline():
    """完整流水线：Schema → Regex → DFA → Token Mask → 约束生成。"""
    print("\n")
    print("=" * 60)
    print("Demo 3：完整流水线 —— 从 Schema 到约束生成")
    print("=" * 60)

    # 第 0 步：Schema → Regex
    regex = schema_to_regex(EXECUTE_COMMAND_SCHEMA)
    print(f"\n  ① Schema → 正则: {regex}")

    # 第 1 步：Regex → DFA
    fsm = regex_to_dfa(regex)
    # 收集所有状态
    all_states = {fsm.start_state} | fsm.accept_states
    for (f, _), t in fsm.transitions.items():
        all_states.add(f)
        all_states.add(t)
    print(f"  ② 正则 → DFA: {len(all_states)} 个状态, {len(fsm.transitions)} 条转移")

    # 第 2 步：DFA + Vocabulary → Token Mask Table
    table = build_token_mask_table(fsm, TOOL_VOCABULARY)
    print(f"  ③ DFA + 词表 → Mask 表: {len(TOOL_VOCABULARY)} 个 token")

    # 展示几个关键状态的 mask
    print(f"\n  --- 关键状态的 token mask ---")

    # 起始状态
    valid = table[fsm.start_state]
    print(f"\n  状态 {fsm.start_state}（起始）:")
    for tid in sorted(valid.keys()):
        print(f"    ✅ token[{tid:>2}] = {TOOL_VOCABULARY[tid]!r}")
    blocked = len(TOOL_VOCABULARY) - len(valid)
    print(f"    ❌ 其余 {blocked} 个 token 被屏蔽")

    # 走到 {"name": " 后面
    end_state = walk_token(fsm, fsm.start_state, '{"name": "')
    if end_state:
        valid = table[end_state]
        print(f'\n  状态 {end_state}（已生成 \'{{"name": "\'）:')
        for tid in sorted(valid.keys()):
            print(f"    ✅ token[{tid:>2}] = {TOOL_VOCABULARY[tid]!r}")
        blocked = len(TOOL_VOCABULARY) - len(valid)
        print(f"    ❌ 其余 {blocked} 个 token 被屏蔽")
        print(f"    → 只有 'execute_command' 是合法的工具名！")

    # 走到 arguments 的 command 值内部
    prefix = '{"name": "execute_command", "arguments": {"command": "'
    end_state = walk_token(fsm, fsm.start_state, prefix)
    if end_state:
        valid = table[end_state]
        print(f"\n  状态 {end_state}（command 值内部）:")
        for tid in sorted(valid.keys())[:10]:
            print(f"    ✅ token[{tid:>2}] = {TOOL_VOCABULARY[tid]!r}")
        if len(valid) > 10:
            print(f"    ... 共 {len(valid)} 个合法 token")
        blocked = len(TOOL_VOCABULARY) - len(valid)
        print(f"    ❌ 其余 {blocked} 个 token 被屏蔽（数字、换行、true 等）")

    # 第 3 步：模拟约束生成
    print(f"\n  --- 模拟约束生成（3 次）---\n")
    for seed in range(3):
        output, steps = simulate_constrained_generation(
            fsm, TOOL_VOCABULARY, table, seed
        )
        print(f"    种子={seed}: {output}")

    print(f"\n  → 每次输出都是合法的 tool_call JSON！")
    print(f"  → 工具名一定是 'execute_command'（schema 约束）")
    print(f"  → 不会出现格式错误、多余废话、缺少字段")


def demo_multi_tool():
    """展示多工具场景：tool_call 可以是多种工具之一。"""
    print("\n")
    print("=" * 60)
    print("Demo 4：多工具场景 —— execute_command 或 read_file")
    print("=" * 60)

    # 分别生成正则，用 | 连接
    regexes = [schema_to_regex(s) for s in TOOL_CALL_SCHEMAS]
    combined = "(" + "|".join(regexes) + ")"

    print(f"\n  execute_command 正则:")
    print(f"    {regexes[0]}")
    print(f"\n  read_file 正则:")
    print(f"    {regexes[1]}")
    print(f"\n  合并正则（用 | 连接）:")
    print(f"    ({regexes[0]}|{regexes[1]})")

    # 构建 DFA
    fsm = regex_to_dfa(combined)
    table = build_token_mask_table(fsm, TOOL_VOCABULARY)

    # 验证
    print(f"\n  验证:")
    tests = [
        '{"name": "execute_command", "arguments": {"command": "ls"}}',
        '{"name": "read_file", "arguments": {"path": "/tmp"}}',
        '{"name": "web_search", "arguments": {"command": "ls"}}',
    ]
    for test in tests:
        fsm.reset()
        ok = True
        for ch in test:
            if not fsm.step(ch):
                ok = False
                break
        accepted = ok and fsm.is_accepted()
        mark = "✅" if accepted else "❌"
        print(f"    {mark} {test}")

    # 展示分叉点
    end_state = walk_token(fsm, fsm.start_state, '{"name": "')
    if end_state:
        valid = table[end_state]
        print(f'\n  分叉点 —— 已生成 \'{{"name": "\' 后:')
        for tid in sorted(valid.keys()):
            print(f"    ✅ token[{tid:>2}] = {TOOL_VOCABULARY[tid]!r}")
        print(f"    → 只有 execute_command 和 read_file 可选！")
        print(f"    → web_search 不在 schema 里，被 mask 掉了")


def demo_summary():
    print("\n")
    print("=" * 60)
    print("完整流水线总结")
    print("=" * 60)
    print("""
四步串起来，就是 outlines 的完整实现：

  step1: FSM 基础
    → 状态机的 get_valid_chars() 是约束解码的灵魂

  step2: 正则 → NFA → DFA
    → Thompson 构造 + 子集构造
    → 任意正则自动变成状态机

  step3: DFA + 词表 → Token Mask 表
    → 预计算每个状态下哪些 token 合法
    → 从字符级升级到 token 级

  step4: JSON Schema → 正则（本文件）
    → 递归翻译，每种类型有对应的正则模板
    → Schema 变了？重新生成正则就行，下游全自动

组装起来就是一行：

    schema → schema_to_regex → regex_to_dfa → build_token_mask_table
                ↑                                      ↓
          你定义工具时写的                      每一步采样时查表 mask

从"请输出 JSON"的 prompt 约束  →  代码级别的 logits mask
从"运气好才格式正确"          →  "格式不可能错"
从 llm_json_parser 的 index("{") 擦屁股  →  根本不需要擦
""")


if __name__ == "__main__":
    demo_schema_to_regex()
    demo_tool_call_regex()
    demo_full_pipeline()
    demo_multi_tool()
    demo_summary()
