"""工具调度执行器:把一轮里的若干 tool_calls 跑出结果。

从 Agent 里独立出来,职责单一——查表、钳超时、按原始顺序切并发安全批次、
把异常/超时吞成 ToolResult 占位。它不认识 ReAct 主循环,也不认识
整个 Renderer,只在构造时收一个 on_command_output 回调(execute_command 的流式输出
要从工具内部的 reader 线程往外喷,这是唯一需要的渲染钩子)。

调用 execute 时再从参数注入 on_call / on_result 两个插槽:循环的所有权在执行器
手里,渲染只是从插槽插话——不接也照跑(便于单测)。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import threading
from typing import Callable

from .permission import (
    PermissionApprovalHandler,
    PermissionPolicy,
    PermissionResolver,
)
from .session import ToolExecutionTerminal
from .tools.base import (
    Tool,
    ToolCall,
    ToolCancelledError,
    ToolResult,
    ToolRuntime,
)


@dataclass(frozen=True)
class ToolExecutionOutcome:
    call: ToolCall
    result: ToolResult
    status: ToolExecutionTerminal


class ToolExecutor:
    def __init__(
        self,
        tool_registry: dict[str, Tool],
        tool_timeout: float = 30,
        on_command_output: Callable[[str], None] | None = None,
        permission_policy: PermissionPolicy | None = None,
        permission_approval_handler: PermissionApprovalHandler | None = None,
        permission_resolver: PermissionResolver | None = None,
        workspace_dir: Path | None = None,
        cwd_provider: Callable[[], Path] | None = None,
        session_state=None,
        cancellation_check: Callable[[], bool] | None = None,
    ):
        if tool_timeout <= 0:
            raise ValueError("tool_timeout 必须 > 0")
        self.tool_registry = tool_registry
        self.tool_timeout = tool_timeout
        self.permission_resolver = permission_resolver or PermissionResolver(
            permission_policy or PermissionPolicy(),
            permission_approval_handler,
        )
        self.session_state = session_state
        self.workspace_dir = (
            workspace_dir
            or getattr(session_state, "workspace_dir", None)
            or Path.cwd()
        ).resolve()
        self.cwd_provider = (
            cwd_provider
            or (
                session_state.get_cwd
                if session_state is not None
                else lambda: self.workspace_dir
            )
        )
        self.cancellation_check = cancellation_check
        self.runtime = ToolRuntime(
            workspace_dir=self.workspace_dir,
            cwd_provider=self.cwd_provider,
            session_state=session_state,
            emit_output=on_command_output,
        )

    def _invoke_tool(
        self,
        tool_call: ToolCall,
        local_cancel: threading.Event,
        on_call_start: Callable[[], None] | None = None,
    ) -> ToolResult:
        """查找并执行【单个】工具，返回标准化 tool_result。"""
        tool = self.tool_registry.get(tool_call.name)
        if tool is None:
            return ToolResult.fail(err=f"Unknown tool: {tool_call.name}")

        runtime = replace(
            self.runtime,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            cancellation_check=lambda: local_cancel.is_set()
            or bool(self.cancellation_check and self.cancellation_check()),
        )

        try:
            runtime.raise_if_cancelled()
        except ToolCancelledError as e:
            return ToolResult.fail(str(e))

        permission = self.permission_resolver.resolve(
            tool_call,
            tool,
            runtime=runtime,
            cwd=self._current_cwd(),
            workspace_dir=self.workspace_dir,
        )
        if permission.decision != "allow":
            return ToolResult.fail(
                err=f"Permission denied: {permission.reason}",
                data={
                    "permission": {
                        "decision": permission.decision,
                        "reason": permission.reason,
                        "risk_flags": list(permission.risk_flags),
                        "source": permission.source,
                    }
                },
            )

        # 浅拷贝再改:钳超时是执行期的局部需要,不能回写 tool_call.arguments
        # ——那个 dict 同一对象被 session 记账引用着,原地改会篡改"已记录的历史输入"。
        arguments = dict(
            tool_call.arguments
            if permission.updated_arguments is None
            else permission.updated_arguments
        )

        # 内层超时必须 ≤ 外层线程预算:模型可以给工具传很大的 timeout,
        # 不钳制的话外层先掐,工具内部的超时机制(如 execute_command 转后台)永远轮不到登场
        if isinstance(arguments.get("timeout"), (int, float)):
            arguments["timeout"] = min(arguments["timeout"], self.tool_timeout)

        try:
            runtime.raise_if_cancelled()
            if on_call_start is not None:
                on_call_start()
            tool_result = tool.call(arguments, runtime)
        except ToolCancelledError as e:
            tool_result = ToolResult.fail(str(e))
        except Exception as e:
            tool_result = ToolResult.fail(f"{type(e).__name__}: {e}")

        return tool_result

    def _current_cwd(self) -> Path:
        try:
            return self.cwd_provider().resolve()
        except Exception:
            return self.workspace_dir

    def _is_concurrency_safe(self, tool_call: ToolCall) -> bool:
        """按本次参数判断能否并发；未知/判断异常一律按排他执行。"""
        tool = self.tool_registry.get(tool_call.name)
        if tool is None:
            return False
        try:
            return bool(tool.is_concurrency_safe(dict(tool_call.arguments)))
        except Exception:
            return False

    def _run_one(self, idx: int, tool_call: ToolCall) -> tuple[int, ToolExecutionOutcome]:
        tool = self.tool_registry.get(tool_call.name)
        local_cancel = threading.Event()
        timer: threading.Timer | None = None

        def start_deadline() -> None:
            nonlocal timer
            timer = threading.Timer(self.tool_timeout, local_cancel.set)
            timer.daemon = True
            timer.start()

        try:
            result = self._invoke_tool(
                tool_call,
                local_cancel,
                on_call_start=(
                    start_deadline
                    if tool is not None and tool.timeout_owner == "executor"
                    else None
                ),
            )
        finally:
            if timer is not None:
                timer.cancel()

        if local_cancel.is_set():
            result = ToolResult.fail(
                f"timeout: 超过 {self.tool_timeout}s，工具已响应取消并退出"
            )
            status: ToolExecutionTerminal = "timeout"
        else:
            status = "succeeded" if result.ok else "failed"

        return idx, ToolExecutionOutcome(
            call=tool_call,
            result=result,
            status=status,
        )

    def _run_concurrent_batch(
        self,
        indexed_calls: list[tuple[int, ToolCall]],
        on_result: Callable[[ToolResult], None] | None,
        max_workers: int,
    ) -> dict[int, ToolExecutionOutcome]:
        """并发跑一批 (原始下标, ToolCall),返回 {下标: outcome}。

        线程池而非进程池:工具都是 I/O 密集,等待时释放 GIL,线程足够;
        进程池还要求参数能 pickle,得不偿失。

        保序靠下标:每个 future 记住自己的原始下标,调用方按下标回填,
        无论谁先跑完都不乱——结果要按 tool_call.id 喂回 LLM,顺序错就对不上号。

        on_result 在主线程按完成顺序触发。_invoke_tool 已把异常吞成
        ToolResult.fail,单个工具失败被隔离;超时的调用以 fail 占位留在结果里,
        绝不"蒸发"(模型靠 id 对账,少一条都不行)。
        """
        out: dict[int, ToolExecutionOutcome] = {}
        if not indexed_calls:
            return out

        # context manager 会等待已经启动的调用真正退出。Python 线程不能安全强杀，
        # 所以 deadline 通过 ToolRuntime 的取消信号协作完成，绝不遗弃后台线程。
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._run_one, idx, tc) for idx, tc in indexed_calls
            ]
            for fut in as_completed(futures):
                idx, outcome = fut.result()
                if on_result:
                    on_result(outcome.result)
                out[idx] = outcome

        return out

    def _partition_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[list[tuple[int, ToolCall]]]:
        """保持原始顺序：连续安全调用合并成并发批，不安全调用各自单独成批。"""
        batches: list[list[tuple[int, ToolCall]]] = []
        previous_batch_is_safe = False
        for indexed in enumerate(tool_calls):
            _, tool_call = indexed
            safe = self._is_concurrency_safe(tool_call)
            if safe and batches and previous_batch_is_safe:
                batches[-1].append(indexed)
            else:
                batches.append([indexed])
            previous_batch_is_safe = safe
        return batches

    def execute(
        self,
        tool_calls: list[ToolCall],
        on_call: Callable[[ToolCall], None] | None = None,
        on_result: Callable[[ToolResult], None] | None = None,
        max_workers: int = 8,
    ) -> list[ToolExecutionOutcome]:
        """保持调用顺序切批执行,返回顺序恒等于输入。

        连续 concurrency-safe 调用并发；每个不安全调用独占一个批次。
        因此 `[read, read, write, read]` 是 `[read+read] → [write] → [read]`，
        不会把后面的 read 提前到 write 前面。

        on_call 先按输入顺序全报一遍("这一轮要调这些工具"),再开跑。
        """
        if on_call:
            for tool_call in tool_calls:
                on_call(tool_call)

        slots: list[ToolExecutionOutcome | None] = [None] * len(tool_calls)

        if max_workers < 1:
            raise ValueError("max_workers 必须 >= 1")

        for batch in self._partition_calls(tool_calls):
            for idx, slot in self._run_concurrent_batch(
                batch, on_result, min(max_workers, len(batch))
            ).items():
                slots[idx] = slot

        return [slot for slot in slots if slot is not None]
