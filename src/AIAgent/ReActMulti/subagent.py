"""子 Agent 编排:把一个自包含子任务委派给独立上下文的子 Agent 跑完再收口。

为什么要有这一层(和"一回合多工具"的区别)
---------------------------------------------------
ReActMulti 的 "Multi" 原本指【一个回合并发多个工具调用】——它们仍共享同一条
对话历史、同一个上下文窗口。但真实复杂任务里,很多子任务彼此独立、各自要烧掉
一大片中间过程(读十个文件、试错命令、翻网页)。如果全塞进主对话:
    1. 主 Agent 的上下文被中间噪音撑爆,水位飙高、频繁触发压缩;
    2. 几个子任务的中间状态互相串味,模型容易拿错变量、引错结论。

`spawn_agent` 把一个子任务丢给【独立 SessionState 的子 Agent】:子 Agent 在自己
干净的上下文里 think→act→observe 到底,只把【最终结论】作为一条 tool_result 交回
主 Agent。主 Agent 看到的是"委派出去 → 拿回一句结论",中间几十步全部被隔离在
子上下文里——这就是子 Agent 编排的核心价值:**上下文隔离 + 结论聚合**。

设计要点
--------
- 递归而非写死层级:父 Agent 手里这把 spawn 工具,会在被调用时给子 Agent 再拼一套
  工具(含 depth+1 的 spawn,只要没到 max_depth)。靠 `build_agent_tools` 一个函数
  收口"某个深度该拿哪些工具",父子用同一规则,天然支持多层委派、又有 max_depth 兜底
  防止无限递归。
- 共享 workspace、隔离上下文:子 Agent 的文件操作落在父 Agent 同一个 workspace,
  所以"子 Agent 写了文件、父 Agent 接着读"是通的;但对话历史各自独立。
- 复用整条主循环:子 Agent 就是一个普通 `Agent`,压缩/记账/权限/超时全部照旧,
  没有给子任务开任何后门。

为什么放在独立模块(而不是 tools/ 里)
-----------------------------------------
这把工具需要 `import Agent`,而 `agent.py` 不 import 本模块,也不 import
`tools/__init__`——所以把 spawn 留在 `subagent.py`、只在装配层(main/测试)接进去,
就不会和 `tools` 包形成循环依赖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .llm import LLMClient
from .permission import PermissionResolver
from .renderer import Renderer, SilentRenderer
from .session import SessionState
from .tools.base import Tool, ToolResult, ToolRuntime


# 子 Agent 默认比主 Agent 更"短促":委派出去的是聚焦的小任务,给它太大的步数预算
# 只会在卡壳时烧更多 token。真需要更长可在装配层调。
DEFAULT_CHILD_MAX_STEPS = 20
# 递归深度上限:depth=0 是主 Agent,它能 spawn 出 depth=1 的子 Agent;到达 max_depth
# 的那一层不再发 spawn 工具,从根上杜绝"子 Agent 无限自我繁殖"。
DEFAULT_MAX_DEPTH = 2


class SubAgentRenderer(Renderer):
    """子 Agent 的精简渲染器:每个事件压成带缩进前缀的一行,呈现委派层级。

    主 Agent 用 ConsoleRenderer 全量流式(看得清细节);子 Agent 内部的逐 token
    流式对人是噪音,这里只保留"调了什么工具 / 成不成 / 最终结论"的骨架,并按 depth
    缩进,让"谁委派给谁"在终端里一目了然。
    """

    def __init__(self, depth: int, task: str) -> None:
        self.depth = depth
        self.task = task
        # depth 从 1 起(主 Agent 是 0),缩进与竖线让嵌套关系可读。
        self._prefix = "    " * (depth - 1) + "│ "

    def _line(self, text: str) -> None:
        print(f"{self._prefix}{text}", flush=True)

    def on_reasoning_delta(self, piece: str) -> None: ...  # 子 Agent 思考过程不外显
    def on_content_delta(self, piece: str) -> None: ...  # 子 Agent 中间内容不外显
    def on_command_output(self, line: str) -> None: ...  # 嵌套命令输出过于嘈杂,丢弃

    def on_tool_call(self, tool_call) -> None:
        args = getattr(tool_call, "arguments", {}) or {}
        brief = json.dumps(args, ensure_ascii=False)
        if len(brief) > 80:
            brief = brief[:77] + "..."
        self._line(f"🔧 子Agent(d{self.depth}) › {tool_call.name} {brief}")

    def on_tool_result(self, tool_result) -> None:
        if hasattr(tool_result, "to_dict"):
            tool_result = tool_result.to_dict()
        if tool_result.get("ok"):
            self._line("✅ 子工具完成")
        else:
            self._line(f"❌ 子工具失败: {tool_result.get('err')}")

    def on_final(self, answer) -> None:
        text = answer if isinstance(answer, str) else json.dumps(
            answer, ensure_ascii=False
        )
        if len(text) > 200:
            text = text[:197] + "..."
        self._line(f"🎯 子Agent(d{self.depth}) 收口: {text}")


SPAWN_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": (
                "交给子 Agent 独立完成的、自包含的子任务描述。子 Agent 看不到当前主"
                "对话的任何历史,所以这里必须写全背景、目标、约束和期望产出;子 Agent "
                "会在自己干净的上下文里完成它,并把最终结论作为本次工具调用的结果返回。"
            ),
        },
    },
    "required": ["task"],
}

SPAWN_AGENT_DESCRIPTION = (
    "把一个【自包含、可独立完成】的子任务委派给一个子 Agent。适合:子任务会产生大量"
    "中间过程(多次读文件/试命令/翻网页)、与主线其余部分相互独立、你只关心它的最终"
    "结论而非中间步骤。子 Agent 拥有独立的上下文窗口(中间过程不会污染你的对话),但"
    "与你共享同一个 workspace(它写的文件你能读到)。一次只委派一件事;若有多件独立"
    "子任务,可在同一回合发起多个 spawn_agent 调用,它们会各自隔离地完成。"
)


def make_spawn_agent_tool(
    llm: LLMClient,
    base_tools: Sequence[Tool],
    *,
    depth: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
    child_max_steps: int = DEFAULT_CHILD_MAX_STEPS,
    render_subagents: bool = True,
    permission_resolver: PermissionResolver | None = None,
) -> Tool:
    """造一把"在 depth 层持有"的 spawn_agent 工具。

    被调用时,它给子 Agent 拼 depth+1 的工具集(见 build_agent_tools),建一个独立
    SessionState 的子 Agent,跑到收口,把最终答案当 tool_result 交回。子 Agent 失败
    (步数耗尽/连续解析失败)则返回 fail,主 Agent 能据此改派或换路子。
    """

    def _call(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            return ToolResult.fail("spawn_agent 需要非空字符串参数 'task'")

        child_depth = depth + 1

        # 子 Agent 拿 depth+1 的工具集:到 max_depth 时 build 返回的集合里不含 spawn,
        # 递归自然终止。
        child_tools = build_agent_tools(
            llm,
            base_tools,
            depth=child_depth,
            max_depth=max_depth,
            child_max_steps=child_max_steps,
            render_subagents=render_subagents,
            permission_resolver=permission_resolver,
        )

        # 共享父 workspace:子 Agent 的产出(文件)要让父 Agent 后续读得到。
        workspace_dir = runtime.workspace_dir or Path.cwd()
        child_session = SessionState.create(
            user_goal=task,
            workspace_dir=workspace_dir,
            max_steps=child_max_steps,
        )

        child_renderer: Renderer = (
            SubAgentRenderer(child_depth, task)
            if render_subagents
            else SilentRenderer()
        )

        # 延迟导入打破"装配期循环":agent.py 不 import 本模块,本模块只在【运行时】
        # 才真正需要 Agent,放到函数体里 import 最干净。
        from .agent import Agent

        child_agent = Agent(
            llm,
            child_tools,
            child_session,
            child_renderer,
            max_consecutive_invalid=3,
            # 子 Agent 复用主 Agent 同一份权限裁决器:规则/模式全树一致,
            # 不给委派出去的子任务开任何权限后门。
            permission_resolver=permission_resolver,
        )

        final_answer = child_agent.run(task, max_steps=child_max_steps)

        usage = {
            "prompt_tokens": child_session.total_usage.prompt_tokens,
            "completion_tokens": child_session.total_usage.completion_tokens,
            "total_tokens": child_session.total_usage.total_tokens,
        }

        if final_answer is None:
            # 子 Agent 没能收口:如实把 status 回传,主 Agent 自己决定改派还是放弃。
            return ToolResult.fail(
                f"子 Agent 未完成任务 (status={child_session.status}, "
                f"steps={child_session.step_count}/{child_max_steps})",
                data={"status": child_session.status, "usage": usage},
            )

        return ToolResult.success(
            data={
                "result": final_answer,
                "status": child_session.status,
                "steps": child_session.step_count,
                "usage": usage,
            }
        )

    return Tool(
        name="spawn_agent",
        description=SPAWN_AGENT_DESCRIPTION,
        parameters=SPAWN_AGENT_PARAMETERS,
        call=_call,
        # serial:子 Agent 会写 workspace / 跑命令,有副作用,必须串行,不能和别的
        # 写工具在同一回合并发争抢同一份文件。
        concurrency="serial",
    )


def build_agent_tools(
    llm: LLMClient,
    base_tools: Sequence[Tool],
    *,
    depth: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
    child_max_steps: int = DEFAULT_CHILD_MAX_STEPS,
    render_subagents: bool = True,
    permission_resolver: PermissionResolver | None = None,
) -> list[Tool]:
    """给"处在 depth 层的 Agent"组装工具集:基础工具 +(未到上限时)一把 spawn。

    父子用同一个函数算各自该拿什么工具,层级语义只此一处,不会两边漂。depth>=max_depth
    时不再追加 spawn,递归到底。
    """
    if depth >= max_depth:
        return list(base_tools)

    spawn_tool = make_spawn_agent_tool(
        llm,
        base_tools,
        depth=depth,
        max_depth=max_depth,
        child_max_steps=child_max_steps,
        render_subagents=render_subagents,
        permission_resolver=permission_resolver,
    )
    return [*base_tools, spawn_tool]
