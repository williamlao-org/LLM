"""
ReActMulti Agent 主入口模块（多工具版）

和隔壁 ReAct 的唯一区别：一个回合可以发起【多个】工具调用。
单工具版是严格串行 think→act(1个)→observe；这一版是 think→act(N个)→observe(N个)。

"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import Agent
from .logger import get_logger
from .tools import tools as base_tools
from .llm import LLMClient
from .permission import (
    FallbackApprovalHandler,
    InteractiveApprovalHandler,
    PermissionResolver,
    RuleBasedApprovalHandler,
    append_allow_rule,
    load_permission_settings,
)
from .renderer import ConsoleRenderer
from .session import SessionState
from .subagent import build_agent_tools
from .tools.mcp_client import McpManager, load_mcp_config


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
        context_limit=context_limit or 128000,
    )
    renderer = ConsoleRenderer()

    prompt = (
        "你是主管 Agent。下面有两个相互【独立】的子任务，请用 spawn_agent 把它们"
        "分别委派给两个子 Agent 完成（不要自己动手做），最后用 final_answer 汇总两个"
        "子 Agent 的结论：\n"
        "子任务 A：计算 1 到 100 的整数之和，并把结果写入 sum_a.txt。\n"
        "子任务 B：用 web_search 查『2024 巴黎奥运会在哪个城市举办』，把答案写入 city_b.txt。"
    )
    workspace_dir = Path(__file__).resolve().parent / "workspace"
    session_state = SessionState.create(
        user_goal=prompt,
        workspace_dir=workspace_dir,
    )

    # MCP 接入:从 workspace 下的 .mcp.json 发现外部 stdio server,连接并把它们的工具
    # 翻译成本系统的 Tool。session 由 mcp_manager 持有,整段运行期保持存活,finally 关闭。
    # 没配 .mcp.json 时 configs 为空,start() 直接返回 [],对其余流程完全无感。
    mcp_manager = McpManager(load_mcp_config(workspace_dir / ".mcp.json"))
    mcp_tools = mcp_manager.start()

    # 权限裁决:加载持久化配置(模式 + allow/deny 规则),按"要不要人"两种装配。
    #
    # 要不要人,默认看有没有真终端,不用记环境变量(env 仍可强制覆盖):
    #   - 有 TTY(你坐在终端前) → 规则 + 人:规则 on_no_match=ask 对灰色地带"弃权",
    #     落到交互式 handler 弹窗问你;rm/sudo 等 deny 仍直接拒、不打扰你。
    #   - 无 TTY(管道/CI/后台) → 纯规则,on_no_match=deny 直接 fail-closed,绝不阻塞。
    # 关键:能不能被问到,取决于规则有没有提前 allow 它——allow 列得越全,落到人手里越少。
    # 默认配置只 allow 只读命令,所以写文件/网络/python 都会落到你这来确认。
    # 主 Agent 与所有子 Agent 共用这同一份 resolver,规则/记忆全树一致。
    settings = load_permission_settings()
    env_interactive = os.getenv("REACT_PERMISSION_INTERACTIVE")
    interactive = (
        env_interactive == "1"
        if env_interactive is not None
        else sys.stdin.isatty()
    )
    if interactive:
        approval_handler = FallbackApprovalHandler(
            RuleBasedApprovalHandler(settings, on_no_match="ask"),
            # on_remember:用户选"别再问"时把规则写回 settings.json,下次同工具在规则层
            # 就自动放行(连这个交互 handler 都到不了)——对标 Claude Code 的"Yes, don't ask again"。
            InteractiveApprovalHandler(on_remember=append_allow_rule),
        )
    else:
        approval_handler = RuleBasedApprovalHandler(settings)
    permission_resolver = PermissionResolver(approval_handler=approval_handler)

    # 给主 Agent 装上"基础工具 + spawn_agent"的分层工具集:depth=0 是主 Agent,
    # 它能委派出 depth=1 的子 Agent;到 max_depth 那层不再带 spawn,递归到底。
    tools = build_agent_tools(
        llm_client,
        base_tools + mcp_tools,
        depth=0,
        max_depth=2,
        permission_resolver=permission_resolver,
    )
    agent = Agent(
        llm_client,
        tools,
        session_state,
        renderer,
        keep_recent_tool_results=3,
        permission_resolver=permission_resolver,
    )

    try:
        agent.run(
            # "执行 python 代码 print(1/0)，并且用 web_search 搜索 2024 年奥运会在哪举办，再对 ifconfig.me/ip 发起 http 请求拿到公网 IP。执行命令查看当前主机信息"
            prompt
            # '写一个坦克大战游戏项目，有完整的开发流程'
        )
    finally:
        # 关闭 MCP session / stdio 子进程,避免残留进程。
        mcp_manager.shutdown()
