"""MCP 学习 Step 2: 用官方 SDK 重写同一个 Server。

对比 step1 手写版，你会发现 SDK 帮你做了：
1. JSON-RPC 协议解析（你不用再手动 json.loads / json.dumps）
2. 消息路由（你不用再写 if method == "tools/list" 这样的分发逻辑）
3. initialize 握手（SDK 自动处理，你不用写）
4. 传输层管理（stdin/stdout 的读写、刷新）

你只需要关注：定义工具 + 写工具的执行逻辑。

安装：pip install mcp
"""

from mcp.server.fastmcp import FastMCP

# ----------------------------------------------------------------
# 创建 Server 实例
# ----------------------------------------------------------------
# FastMCP 是高层封装，类似 Flask/FastAPI 的风格。
# 底层还有一个 mcp.server.Server 低层 API，后面再说。
server = FastMCP(
    name="my-first-mcp-server",  # 对应 step1 里 serverInfo.name
    # version="0.1.0",
)


# ----------------------------------------------------------------
# 定义工具 —— 用装饰器，一个函数就是一个工具
# ----------------------------------------------------------------
# SDK 会自动做这些事情：
#   1. 从函数签名 + type hints 生成 inputSchema（JSON Schema）
#   2. 把函数注册到 tools/list 的返回列表里
#   3. 收到 tools/call 时，自动匹配 name → 调用对应函数
#   4. 把返回值包装成 {"content": [{"type": "text", "text": "..."}]}
@server.tool(description="两个数相加")
def add(a: float, b: float) -> str:
    """计算 a + b 的结果。

    Args:
        a: 第一个数
        b: 第二个数
    """
    return str(a + b)


# 再加一个工具，展示多工具的情况
@server.tool(description="两个数相乘")
def multiply(a: float, b: float) -> str:
    """计算 a * b 的结果。

    Args:
        a: 第一个数
        b: 第二个数
    """
    return str(a * b)


# ----------------------------------------------------------------
# 启动 Server
# ----------------------------------------------------------------
# server.run() 会：
#   1. 监听 stdin，等待 Client 发来 JSON-RPC 消息
#   2. 自动处理 initialize 握手
#   3. 自动响应 tools/list（返回上面注册的所有工具）
#   4. 自动响应 tools/call（调用对应的函数，包装返回值）
#   5. 默认使用 stdio 传输（还支持 sse、streamable-http）
if __name__ == "__main__":
    server.run()
