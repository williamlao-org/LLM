from __future__ import annotations

from uuid import uuid4
from .tools.base import ToolResult, ToolCall
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam
from typing import Literal, TypeAlias

from dataclasses import dataclass, field

CallId: TypeAlias = str
SessionStatus = Literal["running", "completed", "failed", "max_steps", "waiting_user"]
TurnRoute = Literal["tool_calls", "final", "invalid"]
ToolExecutionStatus = Literal["pending", "running", "succeeded", "failed", "timeout"]


@dataclass
class SessionState:
    session_id: str
    status: SessionStatus
    user_goal: str

    workspace_dir: Path
    cwd: Path

    turns: list[TurnRecord]
    messages: list[ChatCompletionMessageParam]  # 可选：模型上下文缓存，不是唯一真实状态

    tool_executions: dict[CallId, ToolExecutionRecord]
    background_tasks: dict[str, BackgroundTask]

    last_usage: UsageRecord | None = None
    total_usage: UsageRecord = field(default_factory=lambda: UsageRecord())

    step_count: int = 0
    max_steps: int = 25

    @classmethod
    def create(
        cls, user_goal: str, workspace_dir: Path, max_steps: int = 50
    ) -> SessionState:
        return cls(
            session_id=uuid4().hex[:6],
            status="running",
            user_goal=user_goal,
            workspace_dir=workspace_dir,
            cwd=workspace_dir,
            turns=[],
            messages=[],
            tool_executions={},
            background_tasks={},
            max_steps=max_steps,
        )

    def _next_step(self) -> int:
        self.step_count += 1
        return self.step_count

    def record_assistant_turn(
        self,
        assistant_raw: str,
        parsed: dict,
        route: Literal["tool_calls", "final"],
        tool_calls: list[ToolCall] | None = None,
    ) -> TurnRecord:
        """记录一轮合法 assistant 输出。

        session 只记录已经解析好的事件,不负责解析 JSON。这样协议解析可以留在
        util/protocol 层,以后换输出协议时不会污染状态层。
        """
        tool_calls = tool_calls or []

        if route == "tool_calls" and not tool_calls:
            raise ValueError("route='tool_calls' 时必须提供 tool_calls")
        if route == "final" and tool_calls:
            raise ValueError("route='final' 时不能提供 tool_calls")

        tool_execution_ids: list[CallId] = []
        for tool_call in tool_calls:
            if not tool_call.id:
                raise ValueError("ToolCall 缺少 id,无法建立工具执行记录")
            if tool_call.id in self.tool_executions or tool_call.id in tool_execution_ids:
                raise ValueError(f"重复的 tool_call id: {tool_call.id}")

            tool_execution_ids.append(tool_call.id)

        step = self._next_step()

        for tool_call in tool_calls:
            self.tool_executions[tool_call.id] = ToolExecutionRecord(
                call=tool_call,
                result=None,
                step=step,
                status="pending",
            )

        turn = TurnRecord(
            step=step,
            assistant_raw=assistant_raw,
            parsed=parsed,
            route=route,
            tool_execution_ids=tool_execution_ids,
            error=None,
        )
        self.turns.append(turn)
        return turn

    def record_invalid_turn(
        self,
        assistant_raw: str,
        error: str,
        parsed: dict | None = None,
    ) -> "TurnRecord":
        """记录一轮无效 assistant 输出,比如 JSON 解析失败或 route 失败。"""
        turn = TurnRecord(
            step=self._next_step(),
            assistant_raw=assistant_raw,
            parsed=parsed or {},
            route="invalid",
            tool_execution_ids=[],
            error=error,
        )
        self.turns.append(turn)
        return turn

    def record_tool_execution(
        self,
        call_id: CallId,
        result: ToolResult,
        status: Literal["succeeded", "failed", "timeout"] | None = None,
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> ToolExecutionRecord:
        """把某个 tool_call 的执行结果写回全局索引。"""
        execution = self.tool_executions.get(call_id)
        if execution is None:
            raise KeyError(f"Unknown tool_call id: {call_id}")

        execution.result = result
        execution.status = status or ("succeeded" if result.ok else "failed")
        if started_at is not None:
            execution.started_at = started_at
        if ended_at is not None:
            execution.ended_at = ended_at

        return execution

    def record_usage_for_turn(self, turn: "TurnRecord", usage: "UsageRecord") -> None:
        turn.usage = usage
        self.last_usage = usage
        self.total_usage.prompt_tokens += usage.prompt_tokens
        self.total_usage.completion_tokens += usage.completion_tokens
        self.total_usage.total_tokens += usage.total_tokens

    def mark_completed(self) -> None:
        self.status = "completed"

    def mark_max_steps(self) -> None:
        self.status = "max_steps"


@dataclass
class ToolExecutionRecord:
    call: ToolCall
    result: ToolResult | None
    step: int
    status: ToolExecutionStatus
    started_at: float | None = None
    ended_at: float | None = None


@dataclass
class UsageRecord:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class TurnRecord:
    step: int
    assistant_raw: str
    parsed: dict
    route: TurnRoute

    tool_execution_ids: list[CallId]
    error: str | None = None

    usage: UsageRecord | None = None


@dataclass
class BackgroundTask:
    pass
