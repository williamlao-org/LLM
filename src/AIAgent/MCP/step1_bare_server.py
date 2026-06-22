"""MCP 学习 Step 1: 手写一个最简陋的 MCP Server（不用任何 SDK）。

目的：看清 MCP Server 到底在做什么。
答案：就是从 stdin 读 JSON-RPC 请求，往 stdout 写 JSON-RPC 响应。

传输方式：stdio（标准输入/输出），这是 MCP 最常用的本地传输方式。
消息格式遵循 JSON-RPC 2.0。

运行方式（不需要真的运行，看懂逻辑即可）：
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}' | python step1_bare_server.py
"""

import json
import sys


def handle_request(request: dict) -> dict | None:
    """根据 method 分发处理，返回 JSON-RPC Response（或 None 表示 Notification 不需要回复）。"""

    method = request.get("method")
    req_id = request.get("id")  # Notification 没有 id

    # ----------------------------------------------------------------
    # 1) initialize —— 握手
    #    Client 连上来第一件事就是发 initialize，告诉 Server：
    #    "我是谁、我支持哪些能力、我用的协议版本是什么"
    #    Server 回复自己的信息和能力。
    # ----------------------------------------------------------------
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",     # MCP 协议版本
                "serverInfo": {
                    "name": "my-first-mcp-server",   # 服务名
                    "version": "0.1.0",
                },
                "capabilities": {
                    "tools": {},  # 告诉 Client："我提供工具能力"
                    # 还可以有 "resources": {}, "prompts": {} 等
                },
            },
        }

    # ----------------------------------------------------------------
    # 2) notifications/initialized —— Client 说"握手完成"
    #    这是一个 Notification（没有 id），Server 不需要回复。
    # ----------------------------------------------------------------
    if method == "notifications/initialized":
        # 不回复，返回 None
        return None

    # ----------------------------------------------------------------
    # 3) tools/list —— Client 问"你有哪些工具？"
    #    Server 返回工具列表。每个工具有：
    #    - name: 工具名
    #    - description: 给 LLM 看的描述
    #    - inputSchema: JSON Schema，描述工具的参数
    # ----------------------------------------------------------------
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "add",
                        "description": "两个数相加",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number", "description": "第一个数"},
                                "b": {"type": "number", "description": "第二个数"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                ]
            },
        }

    # ----------------------------------------------------------------
    # 4) tools/call —— Client 说"帮我调某个工具"
    #    params 里有 name（工具名）和 arguments（参数）。
    #    Server 执行工具逻辑，返回结果。
    #
    #    返回格式：content 是一个数组，每个元素有 type 和对应内容。
    #    type 可以是 "text"（文本）、"image"（图片）、"resource"（资源引用）等。
    # ----------------------------------------------------------------
    if method == "tools/call":
        tool_name = request["params"]["name"]
        arguments = request["params"]["arguments"]

        if tool_name == "add":
            result = arguments["a"] + arguments["b"]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": str(result)}
                    ],
                    "isError": False,
                },
            }

        # 未知工具
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {"type": "text", "text": f"未知工具: {tool_name}"}
                ],
                "isError": True,
            },
        }

    # ----------------------------------------------------------------
    # 未知 method
    # ----------------------------------------------------------------
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }


def main():
    """主循环：逐行从 stdin 读 JSON-RPC 消息，处理后写到 stdout。

    真实的 MCP stdio 传输其实用的是"行分隔 JSON"(newline-delimited JSON)：
    每条消息占一行，以换行符分隔。非常简单。
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)

        if response is not None:
            # 写到 stdout，一条消息一行
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
