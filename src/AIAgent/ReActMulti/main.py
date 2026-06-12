"""
ReActMulti Agent 主入口模块（多工具版）

和隔壁 ReAct 的唯一区别：一个回合可以发起【多个】工具调用。
单工具版是严格串行 think→act(1个)→observe；这一版是 think→act(N个)→observe(N个)。

"""

import os

from dotenv import load_dotenv

from .agent import Agent
from .logger import get_logger
from .tools import tools
from .llm import LLMClient
from .renderer import ConsoleRenderer


logger = get_logger(__name__)


if __name__ == "__main__":
    load_dotenv()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    llm_client = LLMClient(base_url=base_url, api_key=api_key, model=model)
    renderer = ConsoleRenderer()

    agent = Agent(llm_client, tools, renderer)

    agent.run(
        "执行 python 代码 print(1/0)，并且用 web_search 搜索 2024 年奥运会在哪举办，再对 ifconfig.me/ip 发起 http 请求拿到公网 IP。"
    )
