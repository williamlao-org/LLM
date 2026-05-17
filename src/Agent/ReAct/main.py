import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from .tools import tools
from .prompt import SYSTEM_PROMPT

load_dotenv()


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
    reasoning_contents = []

    while True:
        resp = client.chat.completions.create(
            messages=messages,
            model=os.getenv("OPENAI_MODEL"),
            stream=stream,
            stream_options={"include_usage": True},
        )

        if stream:
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

        else:
            message = resp.choices[0].message
            content = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None)

        reasoning_contents.append(reasoning_content)

        messages.append({"role": "assistant", "content": content})

        content = json.loads(content)
        if content.get("final_answer"):
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

    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print(json.dumps(reasoning_contents, ensure_ascii=False, indent=2))


main()
