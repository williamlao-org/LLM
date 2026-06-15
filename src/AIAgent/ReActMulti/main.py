"""
ReActMulti Agent 主入口模块（多工具版）

和隔壁 ReAct 的唯一区别：一个回合可以发起【多个】工具调用。
单工具版是严格串行 think→act(1个)→observe；这一版是 think→act(N个)→observe(N个)。

"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .logger import get_logger
from .tools import tools
from .llm import LLMClient
from .renderer import ConsoleRenderer
from .session import SessionState


logger = get_logger(__name__)


if __name__ == "__main__":
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    context_limit_raw = os.getenv("OPENAI_CONTEXT_LIMIT")
    context_limit = int(context_limit_raw) if context_limit_raw else None

    llm_client = LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        context_limit=context_limit or 2000,
    )
    renderer = ConsoleRenderer()

    prompt = "执行命令查看当前主机信息，查看装了那些应用"
    session_state = SessionState.create(
        user_goal=prompt,
        workspace_dir=Path(__file__).resolve().parent / "workspace",
    )
    agent = Agent(llm_client, tools, session_state, renderer, keep_recent_tool_results=1)

    agent.run(
        # "执行 python 代码 print(1/0)，并且用 web_search 搜索 2024 年奥运会在哪举办，再对 ifconfig.me/ip 发起 http 请求拿到公网 IP。执行命令查看当前主机信息"
        prompt
    )
