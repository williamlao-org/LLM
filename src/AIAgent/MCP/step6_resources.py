import datetime
import httpx
import os
import sys
import asyncio
import json

# mcp 模块
from mcp import StdioServerParameters, stdio_client, ClientSession
from mcp.server import FastMCP

server = FastMCP(name="resources_server")


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


# 通过 http 查询实时股价（使用 async def + AsyncClient 避免阻塞超时）
@server.resource(
    uri="stock://{symbol}",
    name="查询实时股价",
    description="查询指定股票的实时股价",
)
async def get_stock_price(symbol: str) -> str:
    """查询指定股票的实时股价。"""

    url = f"https://hq.sinajs.cn/list={symbol}"
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.text
            return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
    return json.dumps({"error": "查询失败"}, ensure_ascii=False, indent=2)


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

            # 列出静态资源（模板资源不会出现在此列表中）
            resources = await session.list_resources()
            print(f"\n可用静态资源（共 {len(resources.resources)} 个）：")
            for r in resources.resources:
                print(f"  📄 {r.uri}")
                print(f"     名称: {r.name}")
                print(f"     描述: {r.description}")

            # 列出资源模板
            templates = await session.list_resource_templates()
            print(f"\n资源模板（共 {len(templates.resourceTemplates)} 个）：")
            for t in templates.resourceTemplates:
                print(f"  📋 {t.uriTemplate}  →  {t.description}")

            # 通过模板读取动态资源（填入具体股票代码）
            print("\n--- 读取 stock://sz000001 ---")
            result = await session.read_resource("stock://sz000001")
            for content in result.contents:
                print(f"  {content.text}")

            # ========================================================
            # 演示 2: Tools + Resources 共存
            # ========================================================
            print("\n" + "=" * 60)
            print("🔧 Tools + Resources 共存")
            print("=" * 60)

            tools = await session.list_tools()
            print(f"\n  Tools:     {[t.name for t in tools.tools]}")
            print(f"  Resources: {[r.uri for r in resources.resources]}")
            print(
                f"  Templates: {[t.uriTemplate for t in templates.resourceTemplates]}"
            )

            print("\n✅ 完成！")


# ================================================================
# 入口：根据参数决定启动 Server 还是 Client
# ================================================================

if __name__ == "__main__":
    if "--server" in sys.argv:
        # 作为 Server 运行（被 Client 通过 stdio 启动）
        server.run()
    else:
        # 作为 Client 运行（启动 Server 子进程）
        asyncio.run(run_client())
