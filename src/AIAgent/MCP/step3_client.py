"""MCP 学习 Step 3: 写一个 MCP Client，连接 step2 的 Server。

这一步展示 Client 端到底做了什么：
1. 启动 Server 进程（作为子进程，通过 stdio 通信）
2. 完成 initialize 握手
3. 调用 tools/list 获取工具列表
4. 调用 tools/call 执行工具
5. 关闭连接

运行方式：
  uv run src/AIAgent/MCP/step3_client.py

它会自动启动 step2_sdk_server.py 作为子进程。
"""

from mcp.types import TextContent

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    # ----------------------------------------------------------------
    # 1) 定义如何启动 Server
    # ----------------------------------------------------------------
    # StdioServerParameters 告诉 Client：
    #   - 用什么命令启动 Server 进程
    #   - Server 进程的 stdin/stdout 就是通信通道
    #
    # 这就是为什么 MCP Server "就是一个普通程序"：
    # Client 把它当子进程启动，通过管道读写 JSON-RPC 消息。
    server_script = os.path.join(os.path.dirname(__file__), "step2_sdk_server.py")
    server_params = StdioServerParameters(
        command="uv",
        args=["run", server_script],
    )

    # ----------------------------------------------------------------
    # 2) 建立连接 + 握手
    # ----------------------------------------------------------------
    # stdio_client() 是一个 async context manager：
    #   - 进入时：启动 Server 子进程，建立 stdin/stdout 管道
    #   - 退出时：关闭管道，终止子进程
    #
    # ClientSession 是在传输通道之上的会话层：
    #   - 进入时：自动发送 initialize 请求，完成握手
    #   - 之后：提供 list_tools()、call_tool() 等高层方法
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # --------------------------------------------------------
            # 握手：必须显式调用 initialize()
            # --------------------------------------------------------
            # SDK 的 ClientSession 进入 context manager 时只建立了消息循环，
            # 并不会自动握手。你需要手动调用 initialize()。
            #
            # initialize() 内部做了两件事（看 session.py 第 160-209 行）：
            #   1. 发送 InitializeRequest → 收到 Server 的能力声明
            #   2. 发送 InitializedNotification → 告诉 Server "握手完成"
            init_result = await session.initialize()
            print("=" * 60)
            print("握手完成！Server 信息：")
            print(f"  协议版本: {init_result.protocolVersion}")
            print(f"  服务名:   {init_result.serverInfo.name}")
            print(f"  能力:     {init_result.capabilities}")
            print("=" * 60)

            # --------------------------------------------------------
            # 3) 获取工具列表 —— 对应 JSON-RPC method: tools/list
            # --------------------------------------------------------
            tools_result = await session.list_tools()

            print("=" * 60)
            print("Server 提供的工具列表：")
            print("=" * 60)
            for tool in tools_result.tools:
                print(f"\n  工具名: {tool.name}")
                print(f"  描述:   {tool.description}")
                print(f"  参数 Schema: {tool.inputSchema}")

            # --------------------------------------------------------
            # 4) 调用工具 —— 对应 JSON-RPC method: tools/call
            # --------------------------------------------------------
            # call_tool 的底层就是发送:
            # {"jsonrpc":"2.0","id":N,"method":"tools/call",
            #  "params":{"name":"add","arguments":{"a":3,"b":5}}}
            print("\n" + "=" * 60)
            print("调用 add(3, 5)：")
            print("=" * 60)

            result = await session.call_tool("add", {"a": 3, "b": 5})
            print(f"  isError: {result.isError}")
            for content in result.content:
                assert isinstance(content, TextContent)
                print(f"  type={content.type}, text={content.text}")

            # 再调用 multiply
            print("\n" + "=" * 60)
            print("调用 multiply(4, 7)：")
            print("=" * 60)

            result = await session.call_tool("multiply", {"a": 4, "b": 7})
            print(f"  isError: {result.isError}")
            for content in result.content:
                assert isinstance(content, TextContent)
                print(f"  type={content.type}, text={content.text}")

            print("\n✅ 完成！Client 即将断开连接，Server 子进程会被自动终止。")


if __name__ == "__main__":
    asyncio.run(main())
