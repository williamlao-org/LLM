from dataclasses import dataclass
import json
import os
from typing import Any, Callable

import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def calculate(num1, num2):
    return num1 + num2


def web_search(query: str, max_results):
    resp = httpx.post(
        url="https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        results.append({"title": title, "snippet": content, "url": url})
    return results


tool_calculate = Tool(
    name="calculate",
    description="Add two numbers and return the sum.",
    parameters={
        "type": "object",
        "properties": {
            "num1": {"type": "number", "description": "The first number"},
            "num2": {"type": "number", "description": "The second number"},
        },
        "required": ["num1", "num2"],
    },
    func=calculate,
)

tool_websearch = Tool(
    name="web_search",
    description="Search the web for information about a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
            },
        },
        "required": ["query", "max_results"],
    },
    func=web_search,
)

tools: list[Tool] = []

tools.append(tool_calculate)
tools.append(tool_websearch)

tool_registry = {tool.name: tool.func for tool in tools}

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT.format(
            tools=json.dumps(
                [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
            )
        ),
    },
    {"role": "user", "content": "5+6等于多少？还有查一查今天比特币价格"},
]


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
)


def main(stream: bool = True):
    # reasoning_content = []
    # while True:
    #     resp = client.chat.completions.create(
    #         messages=messages, stream=False, model=os.getenv("OPENAI_MODEL")
    #     )

    #     content = resp.choices[0].message.content
    #     messages.append({"role": "assistant", "content": content})

    #     reasoning_content.append(
    #         {"reasoning_content": resp.choices[0].message.reasoning_content}
    #     )

    #     data = json.loads(content)

    #     if data.get("final_answer"):
    #         break

    #     tool_call = data.get("tool_call")
    #     if tool_call:
    #         tool_name = tool_call["name"]
    #         tool_arg = tool_call["arguments"]

    #         tool_fn = tool_registry.get(tool_name)
    #         if tool_fn is None:
    #             raise ValueError(f"Unknown tool: {tool_name}")

    #         tool_result = tool_fn(**tool_arg)

    #         messages.append(
    #             {"role": "user", "content": json.dumps({"tool_result": tool_result})}
    #         )

    # print(json.dumps(messages, ensure_ascii=False, indent=2))
    # print(json.dumps(reasoning_content, ensure_ascii=False, indent=2))

    reasoning_contents=[]

    while True:
        resp = client.chat.completions.create(
            messages=messages,
            model=os.getenv("OPENAI_MODEL"),
            stream=True,
            stream_options={"include_usage": True},
        )

        content = []
        reasoning_content = []

        for chunk in resp:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            reasoning_piece = getattr(delta, "reasoning_content", None)
            content_piece = delta.content or ""

            if reasoning_piece:
                reasoning_content.append(reasoning_piece)

            if content_piece:
                print(content_piece, end="", flush=True)
                content.append(content_piece)


        content = "".join(content)
        reasoning_content = "".join(reasoning_content)
        reasoning_contents.append(reasoning_content)


        content = json.loads(content)
        if content.get('final_answer'):
            break

        # 检查tool
        tool_call = content.get("tool_call")

        if tool_call:
            tool_name = tool_call["name"]
            tool_arguments = tool_call["arguments"]

            tool_fn = tool_registry.get(tool_name)
            if tool_fn is None:
                raise ValueError(f"Unknown tool:{tool_name}")

            tool_result = tool_fn(**tool_arguments)

            messages.append(
                {"role": "user", "content": json.dumps({"tool_result": tool_result})}
            )

    print(json.dumps(messages,ensure_ascii=False,indent=2))
    print(json.dumps(reasoning_contents,ensure_ascii=False,indent=2))

main()