# ============================================================================
# 【你要写的第 1 处 / 共 3 处】多工具版的 system prompt
# ----------------------------------------------------------------------------
# 参照隔壁 ReAct/prompt.py 的单工具版，把它改成"一个回合可以发起多个工具调用"。
#
# 思考题（写之前先想清楚，再动手）：
#   1. 字段名：单工具是 "tool_call"（单个对象）。多工具该叫什么？是对象还是数组？
#   2. 互斥规则：原版规定 tool_call 和 final_answer 恰好一个非 null。
#      多工具版这条怎么改？（提示：数组为空 vs final_answer 非 null）
#   3. 结果对账：一个回合发了 3 个调用，系统会回传 3 个结果。
#      LLM 怎么知道"哪个结果对应哪个调用"？要不要给每个调用一个 id？
#      —— 这一题会直接影响你第 3 处（main.py 的结果回传）怎么写。
#
# 写完后把下面这个占位字符串替换成你的真正 prompt。
# 注意：用 .format(tools=...) 注入工具列表，所以模板里的花括号要写成 {{ }}（除了 {tools}）。
# ============================================================================

SYSTEM_PROMPT = """
You are a assistant with follow available tools:

Available tools:
{tools}

You must think and act using the following structure and output exactly one JSON object each turn:

{{
  "reasoning": "<your thought, analysis, and reasoning for this step>",
  "tool_calls": [
    {{
        "name": "<tool_name>",
        "arguments": {{ ... }}
    }}
  ],
  "final_answer": "<If you have obtained the final answer, put it here; otherwise null>"
}}

Rules (explicit):
- Output must be strict JSON (parsable by `json.loads()`), with no surrounding commentary or extraneous characters.
- `tool_calls` is a list. An empty list `[]` means you are NOT calling any tool this turn.
- Exactly one of the following holds each turn:
    (a) `tool_calls` is non-empty AND `final_answer` is null  -> system executes the calls and returns results.
    (b) `tool_calls` is `[]` AND `final_answer` is non-null    -> session ends.
- When `tool_calls` is not `[]`, each value of those `name` must exactly match one of the tool names listed in the `Available tools` section above.
"""