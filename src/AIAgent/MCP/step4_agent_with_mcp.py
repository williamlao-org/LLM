"""MCP 学习 Step 4: 把 MCP 工具接入 LLM Agent。

这一步展示 MCP 在真实场景中的用法：
  LLM（通过 function calling）决定调用哪个工具 → Client 通过 MCP 协议转发给 Server 执行

流程对比：

  【你之前的做法】
  LLM → function_call(name="add", args={a:3,b:5})
      → 你自己写的 add() 函数
      → 结果塞回 messages

  【用 MCP 之后】
  LLM → function_call(name="add", args={a:3,b:5})
      → MCP Client → (JSON-RPC) → MCP Server 的 add()
      → 结果塞回 messages

  区别就是中间多了一层 MCP 协议，但换来的是：
  - 工具可以在任何地方运行（本地子进程、远程服务器）
  - 工具是即插即用的（换一个 Server 就换一套工具，不用改 Agent 代码）

运行方式：
  需要设置 OPENAI_API_KEY 环境变量（或你使用的 LLM 的 key）
  uv run src/AIAgent/MCP/step4_agent_with_mcp.py
"""

from mcp.types import TextContent

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
)
import asyncio
import json
import os
from dotenv import load_dotenv

from openai import AsyncOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()


async def main():
    # ================================================================
    # 第一阶段：连接 MCP Server，获取工具列表
    # ================================================================
    server_script = os.path.join(os.path.dirname(__file__), "step2_sdk_server.py")
    server_params = StdioServerParameters(
        command="uv",
        args=["run", server_script],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 拿到 MCP Server 提供的工具列表
            tools_result = await session.list_tools()

            # ========================================================
            # 关键步骤：把 MCP 工具转换为 OpenAI function calling 格式
            # ========================================================
            # MCP 工具的 inputSchema 就是标准 JSON Schema，
            # 和 OpenAI function calling 要求的 parameters 格式完全一致！
            # 这不是巧合——MCP 就是设计成和 function calling 对齐的。
            openai_tools: list[ChatCompletionFunctionToolParam] = []
            for tool in tools_result.tools:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                            "strict": True,
                        },
                    }
                )

            print("从 MCP Server 获取到的工具，已转换为 OpenAI 格式：")
            for t in openai_tools:
                print(f"  - {t['function']['name']}: {t['function']['description']}")

            # ========================================================
            # 第二阶段：LLM Agent 循环（简化版 ReAct）
            # ========================================================
            client = AsyncOpenAI(
                base_url=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": "你是一个数学助手。使用工具来计算。"},
                {"role": "user", "content": "请计算 (3 + 5) * 7 的结果"},
            ]

            print(f"\n用户问题: {messages[-1]['content']}")
            print("-" * 60)

            # Agent 循环：持续和 LLM 交互，直到 LLM 不再调用工具
            max_iterations = 10
            for i in range(max_iterations):
                # 调用 LLM
                response = await client.chat.completions.create(
                    model=os.environ["OPENAI_MODEL"],
                    messages=messages,
                    tools=openai_tools,  # ← 这些工具来自 MCP Server！
                )

                assistant_msg = response.choices[0].message
                messages.append(assistant_msg.model_dump())

                # 如果 LLM 没有调用工具，说明它已经有了最终答案
                if not assistant_msg.tool_calls:
                    print(f"\nLLM 最终回答: {assistant_msg.content}")
                    break

                # LLM 决定调用工具 → 通过 MCP Client 转发给 Server
                for tool_call in assistant_msg.tool_calls:
                    if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                        continue  # 跳过非标准工具调用
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    print(f"[第 {i + 1} 轮] LLM 调用工具: {fn_name}({fn_args})")

                    # ============================================
                    # 核心：通过 MCP 协议调用工具
                    # ============================================
                    # 这一行就是 MCP 的价值所在：
                    # 不管工具的实现在哪里（本地/远程），调用方式完全一样。
                    mcp_result = await session.call_tool(fn_name, fn_args)

                    # 提取文本结果
                    result_text = ""
                    for content in mcp_result.content:
                        if hasattr(content, "text"):
                            assert isinstance(content, TextContent)
                            result_text += content.text

                    print(f"          工具返回: {result_text}")

                    # 把结果塞回 messages（OpenAI 格式）
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )

            print("-" * 60)
            print("✅ Agent 循环结束")


if __name__ == "__main__":
    asyncio.run(main())
