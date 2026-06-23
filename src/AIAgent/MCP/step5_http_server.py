"""MCP 学习 Step 5: Streamable HTTP 传输 —— 让 MCP Server 变成 HTTP 服务。

对比 step2（stdio）：
  - stdio:  Server 是子进程，Client 通过管道通信，只能本地用
  - HTTP:   Server 是 HTTP 服务，Client 通过网络通信，可以远程用

SDK 切换传输方式只需改一行配置，工具定义完全不变！
这就是"协议层和传输层分离"的好处。

启动方式：
  uv run src/AIAgent/MCP/step5_http_server.py
  → 会在 http://localhost:8000/mcp 启动 HTTP 服务
"""

from mcp.server.fastmcp import FastMCP

# ----------------------------------------------------------------
# 创建 Server —— 工具定义和 step2 完全一样！
# ----------------------------------------------------------------
server = FastMCP(
    name="my-http-mcp-server",
)


@server.tool(description="两个数相加")
def add(a: float, b: float) -> str:
    """计算 a + b 的结果。"""
    return str(a + b)


@server.tool(description="两个数相乘")
def multiply(a: float, b: float) -> str:
    """计算 a * b 的结果。"""
    return str(a * b)


# ----------------------------------------------------------------
# 启动方式的区别 —— 这是唯一要改的地方
# ----------------------------------------------------------------
# step2 用的是: server.run()          → 默认 stdio 传输
# 这里用的是:  server.run(transport="streamable-http")  → HTTP 传输
#
# 底层发生了什么：
# 1. SDK 启动一个 uvicorn HTTP 服务器
# 2. 暴露一个端点（默认 /mcp）
# 3. Client 发 POST 请求到这个端点，Body 是 JSON-RPC 消息
# 4. Server 返回 JSON-RPC 响应（普通 JSON 或 SSE 流）
#
# 可选参数：
#   host="0.0.0.0"    → 监听所有网卡（默认 127.0.0.1）
#   port=8000          → 端口号（默认 8000）
if __name__ == "__main__":
    server.run(transport="streamable-http")
