import json


SYSTEM_PROMPT = """
You are an assistant.

Available tools:
{tools}

You must think and act using the following structure and output exactly one JSON object each turn:

{{
  "reasoning": "<your thought, analysis, and reasoning for this step>",
  "tool_call": {{
    "name": "<tool_name>",
    "arguments": {{ ... }}
  }} | null,
  "final_answer": "<If you have obtained the final answer, put it here; otherwise null>"
}}

Rules (explicit):
1. Output must be strict JSON (parsable by `json.loads()`), with no surrounding commentary or extraneous characters.
2. Exactly one of `tool_call` or `final_answer` must be non-null each turn; they cannot both be non-null.
3. If `tool_call` is non-null, `final_answer` must be null.
4. If `final_answer` is non-null, `tool_call` must be null and the session ends.
5. When `tool_call` is non-null, the `name` value must exactly match one of the tool names listed in the `Available tools` section above.

If `final_answer` is not null, terminate; otherwise the system will execute the specified `tool_call` and return `tool_result` to you.
"""

tools = [
    {
        "name": "calculate",
        "description": "Add two numbers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {"type": "number", "description": "The first number"},
                "num2": {"type": "number", "description": "The second number"},
            },
            "required": ["num1", "num2"],
        },
    }
]

print(json.dumps(tools,indent=2))