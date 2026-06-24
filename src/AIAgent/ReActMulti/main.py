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
from .memory import MemoryManager
from .tools import tools as base_tools
from .tools.memory_tools import save_memory_tool, search_memory_tool
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

    # 记忆的召回/提取 side-query 可选用更便宜的模型省钱(对标 memdir 用 Sonnet 选记忆)。
    # 配了 OPENAI_MEMORY_MODEL 就单独建个非流式 client,否则复用主 client。
    memory_model = os.getenv("OPENAI_MEMORY_MODEL")
    selector_llm = (
        LLMClient(
            base_url=base_url,
            api_key=api_key,
            model=memory_model,
            stream=False,
        )
        if memory_model
        else llm_client
    )

    renderer = ConsoleRenderer()

    workspace_dir = Path(__file__).resolve().parent / "workspace"
    # 多轮对话:session 整段存活,每轮把用户输入 append 进同一条历史。user_goal 只是
    # 记录字段(主 agent 不依赖),交互模式下用占位串,首条输入由 REPL 循环喂入。
    session_state = SessionState.create(
        user_goal="(interactive session)",
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

    # 记忆只给主 Agent:save/search 工具在 build_agent_tools 之后【单独追加】到主工具集,
    # 不进 base_tools——所以 spawn_agent 为子 Agent 重建工具集时拿不到它们,子 Agent 保持
    # 纯净隔离上下文(也不带记忆指令段,因为不注入 MemoryManager)。
    memory_manager = MemoryManager(llm_client, selector_llm=selector_llm)
    tools = [*tools, save_memory_tool, search_memory_tool]

    agent = Agent(
        llm_client,
        tools,
        session_state,
        renderer,
        keep_recent_tool_results=3,
        permission_resolver=permission_resolver,
        memory=memory_manager,
    )

    # ---- REPL:外层多轮循环 ----
    # 对标 Claude Code 的 REPL：内层 agent.run() 把【一个 user turn】跑到 final_answer
    # 就交还控制权;外层在这里读下一句输入,复用【同一个 agent / session】再 run。
    # 历史天然续上——session_state.messages 累积全部上文,run() 每次只 append 新 user 消息。
    # 退出:Ctrl-D / Ctrl-C / 输入 /exit | /quit。
    try:
        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()  # 让提示符换行,终端干净收尾
                break
            if not user_input:
                continue
            if user_input in ("/exit", "/quit"):
                break
            agent.run(user_input)
    finally:
        # 关闭 MCP session / stdio 子进程,避免残留进程。
        mcp_manager.shutdown()
