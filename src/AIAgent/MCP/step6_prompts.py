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
            # 演示 2: Prompts
            # ========================================================
            print("\n" + "=" * 60)
            print("💬 Prompts（提示模板）")
            print("=" * 60)

            # 列出所有 prompt
            prompts = await session.list_prompts()
            print(prompts)
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
            print(prompt_result)
            for msg in prompt_result.messages:
                print(f"  role: {msg.role}")
                print(f"  content: {msg.content.text[:200]}...")

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
