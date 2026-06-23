"""MCP 学习 Step 6: Resources 和 Prompts —— MCP 的另外两大能力。

Resources（资源）：Server 向 Client 暴露可读取的数据。
  - 每个 Resource 有一个 URI（如 "config://app/settings"）
  - Client 通过 resources/list 发现有哪些资源
  - Client 通过 resources/read 读取资源内容
  - 典型用途：把数据注入到 LLM 的上下文中（作为背景知识）

Prompts（提示模板）：Server 提供预制的 prompt 模板。
  - 每个 Prompt 有名字和可选参数
  - Client 通过 prompts/list 发现有哪些模板
  - Client 通过 prompts/get 获取渲染后的 prompt
  - 典型用途：标准化常用的 prompt 模式（代码审查、摘要、翻译等）

运行方式：
  uv run src/AIAgent/MCP/step6_resources_prompts.py
"""
import asyncio
import json
import os
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP

# ================================================================
# Server 部分
# ================================================================
server = FastMCP(name="demo-resources-prompts")


# ----------------------------------------------------------------
# 1) 定义 Resources —— 用 @server.resource 装饰器
# ----------------------------------------------------------------
# 每个 Resource 需要一个 URI。URI 格式是自定义的，MCP 不限制。
# 常见的 URI scheme：
#   file://   → 文件
#   db://     → 数据库
#   config:// → 配置
#   https://  → 网页

# 静态资源：URI 固定，内容在调用时动态生成
@server.resource(
    uri="config://app/settings",
    name="应用配置",
    description="当前应用的配置信息",
)
def get_app_settings() -> str:
    """返回应用配置（JSON 字符串）。

    Resource 函数返回 str，SDK 会自动包装成：
    {"contents": [{"uri": "config://app/settings", "text": "..."}]}
    """
    return json.dumps(
        {
            "app_name": "MCP 学习项目",
            "version": "0.1.0",
            "debug": True,
            "max_retries": 3,
        },
        ensure_ascii=False,
        indent=2,
    )


@server.resource(
    uri="data://system/status",
    name="系统状态",
    description="系统运行时状态信息",
)
def get_system_status() -> str:
    """Resource 可以返回动态数据。"""
    return json.dumps(
        {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": "23%",
            "memory_usage": "4.2GB / 16GB",
            "active_connections": 42,
        },
        ensure_ascii=False,
        indent=2,
    )


# Resource 模板：URI 里有参数（类似 REST 的路径参数）
@server.resource(
    uri="users://{user_id}/profile",
    name="用户资料",
    description="根据用户 ID 获取用户资料",
)
def get_user_profile(user_id: str) -> str:
    """URI 模板中的 {user_id} 会自动映射为函数参数。

    Client 请求 "users://alice/profile" 时，user_id="alice"。
    """
    fake_users = {
        "alice": {"name": "Alice", "role": "admin", "email": "alice@example.com"},
        "bob": {"name": "Bob", "role": "developer", "email": "bob@example.com"},
    }
    user = fake_users.get(user_id, {"error": f"用户 {user_id} 不存在"})
    return json.dumps(user, ensure_ascii=False, indent=2)

# 通过http查询实时股价
@server.resource(
    uri="stock://{symbol}",
    name="查询实时股价",
    description="查询指定股票的实时股价",
)
def get_stock_price(symbol: str) -> str:
    """查询指定股票的实时股价。"""
    import httpx
    
    url = f"https://hq.sinajs.cn/list={symbol}"
    resp = httpx.request('GET', url)
    if resp.status_code == 200:
        data = resp.text
        return json.dumps(data, ensure_ascii=False, indent=2)
    return json.dumps({"error": "查询失败"}, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------
# 2) 定义 Prompts —— 用 @server.prompt 装饰器
# ----------------------------------------------------------------
# Prompt 返回的是一组 messages（和 OpenAI 的 messages 格式一致），
# 可以直接拼接到 LLM 的对话历史中。

@server.prompt(
    name="code-review",
    description="对一段代码进行审查，提供改进建议",
)
def code_review_prompt(code: str, language: str = "python") -> str:
    """Prompt 函数的参数就是模板参数。

    Client 调用 prompts/get 时传入参数，Server 渲染模板。
    返回 str 时，SDK 会自动包装成:
    {"messages": [{"role": "user", "content": {"type": "text", "text": "..."}}]}
    """
    return f"""请审查以下 {language} 代码，关注：
1. 潜在的 bug 和边界情况
2. 性能问题
3. 代码风格和可读性
4. 安全隐患

```{language}
{code}
```

请用中文给出具体的改进建议。"""


@server.prompt(
    name="explain-error",
    description="解释一个错误信息并给出解决方案",
)
def explain_error_prompt(error_message: str, context: str = "") -> str:
    """另一个 Prompt 模板的例子。"""
    base = f"我遇到了以下错误:\n\n```\n{error_message}\n```\n"
    if context:
        base += f"\n上下文信息:\n{context}\n"
    base += "\n请解释这个错误的原因，并给出具体的解决步骤。"
    return base


# 也加一个工具，展示三种能力共存
@server.tool(description="两个数相加")
def add(a: float, b: float) -> str:
    return str(a + b)


# ================================================================
# Client 部分 —— 演示如何使用 Resources 和 Prompts
# ================================================================
async def run_client():
    server_script = os.path.abspath(__file__)
    server_params = StdioServerParameters(
        command="uv",
        args=["run", server_script, "--server"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # ========================================================
            # 演示 1: Resources
            # ========================================================
            print("=" * 60)
            print("📦 Resources（资源）")
            print("=" * 60)

            # 列出所有资源
            resources = await session.list_resources()
            print(f"\n可用资源（共 {len(resources.resources)} 个）：")
            for r in resources.resources:
                print(f"  📄 {r.uri}")
                print(f"     名称: {r.name}")
                print(f"     描述: {r.description}")

            # 读取静态资源
            print("\n--- 读取 config://app/settings ---")
            result = await session.read_resource("config://app/settings")
            for content in result.contents:
                print(f"  {content.text}")

            # 读取动态资源
            print("\n--- 读取 data://system/status ---")
            result = await session.read_resource("data://system/status")
            for content in result.contents:
                print(f"  {content.text}")

            # 列出资源模板
            templates = await session.list_resource_templates()
            print(f"\n资源模板（共 {len(templates.resourceTemplates)} 个）：")
            for t in templates.resourceTemplates:
                print(f"  📋 {t.uriTemplate}  →  {t.description}")

            # 用模板读取资源
            print("\n--- 读取 users://alice/profile ---")
            result = await session.read_resource("users://alice/profile")
            for content in result.contents:
                print(f"  {content.text}")

            # ========================================================
            # 演示 2: Prompts
            # ========================================================
            print("\n" + "=" * 60)
            print("💬 Prompts（提示模板）")
            print("=" * 60)

            # 列出所有 prompt
            prompts = await session.list_prompts()
            print(f"\n可用 Prompt（共 {len(prompts.prompts)} 个）：")
            for p in prompts.prompts:
                print(f"  🏷️  {p.name}: {p.description}")
                if p.arguments:
                    for arg in p.arguments:
                        required = "必填" if arg.required else "可选"
                        print(f"      参数: {arg.name} ({required})")

            # 获取渲染后的 prompt
            print("\n--- 获取 code-review prompt ---")
            prompt_result = await session.get_prompt(
                "code-review",
                arguments={
                    "code": "def add(a, b):\n    return a + b",
                    "language": "python",
                },
            )
            for msg in prompt_result.messages:
                print(f"  role: {msg.role}")
                print(f"  content: {msg.content.text[:200]}...")

            # ========================================================
            # 演示 3: 三种能力共存
            # ========================================================
            print("\n" + "=" * 60)
            print("🔧 Tools + Resources + Prompts 共存")
            print("=" * 60)

            tools = await session.list_tools()
            print(f"\n  Tools:     {[t.name for t in tools.tools]}")
            print(f"  Resources: {[r.uri for r in resources.resources]}")
            print(f"  Prompts:   {[p.name for p in prompts.prompts]}")

            print("\n✅ 完成！")


# ================================================================
# 入口：根据参数决定启动 Server 还是 Client
# ================================================================
import sys

if __name__ == "__main__":
    if "--server" in sys.argv:
        # 作为 Server 运行（被 Client 通过 stdio 启动）
        server.run()
    else:
        # 作为 Client 运行（启动 Server 子进程）
        asyncio.run(run_client())
