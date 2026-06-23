"""MCP 学习 Step 5b: 用 HTTP 方式连接 MCP Server 的 Client。

对比 step3（stdio client）：
  - stdio client: 用 stdio_client()，它会启动 Server 子进程
  - HTTP client:  用 streamablehttp_client()，它连接到已经运行的 HTTP 服务

使用方式：
  1. 先启动 step5_http_server.py:  uv run src/AIAgent/MCP/step5_http_server.py
  2. 再运行本文件:                  uv run src/AIAgent/MCP/step5_http_client.py
"""

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    # ----------------------------------------------------------------
    # 对比 stdio 的连接方式
    # ----------------------------------------------------------------
    #
    # 【stdio 方式 - step3】
    #   server_params = StdioServerParameters(command="uv", args=["run", "server.py"])
    #   async with stdio_client(server_params) as (read, write):
    #       ...
    #   → Client 自己启动 Server 子进程，管理生命周期
    #
    # 【HTTP 方式 - 这里】
    #   async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
    #       ...
    #   → Client 连接到已经运行的 Server，不管 Server 怎么启动的
    #
    # 注意：streamablehttp_client 返回的是三元组 (read, write, get_session_id)
    # 多了一个 get_session_id，因为 HTTP 是无状态的，需要 session ID 维持会话

    server_url = "http://localhost:8000/mcp"

    async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # 从这里往下，和 stdio client 完全一样！
            # 协议层的 API（initialize / list_tools / call_tool）不因传输方式而变

            await session.initialize()
            print("✅ 握手完成（通过 HTTP）")

            # 列出工具
            tools_result = await session.list_tools()
            print(f"\n工具列表（共 {len(tools_result.tools)} 个）：")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")

            # 调用工具
            result = await session.call_tool("add", {"a": 10, "b": 20})
            for content in result.content:
                if hasattr(content, "text"):
                    print(f"\nadd(10, 20) = {content.text}")

            result = await session.call_tool("multiply", {"a": 6, "b": 9})
            for content in result.content:
                if hasattr(content, "text"):
                    print(f"multiply(6, 9) = {content.text}")

    print("\n✅ HTTP 连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
