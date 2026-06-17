"""工具调度执行器:把一轮里的若干 tool_calls 跑出结果。

从 Agent 里独立出来,职责单一——查表、钳超时、按 concurrency 分流(只读批并发、
写/命令批串行)、把异常/超时吞成 ToolResult 占位。它不认识 ReAct 主循环,也不认识
整个 Renderer,只在构造时收一个 on_command_output 回调(execute_command 的流式输出
要从工具内部的 reader 线程往外喷,这是唯一需要的渲染钩子)。

调用 execute 时再从参数注入 on_call / on_result 两个插槽:循环的所有权在执行器
手里,渲染只是从插槽插话——不接也照跑(便于单测)。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from typing import Callable

from .permission import (
    PermissionApprovalHandler,
    PermissionPolicy,
    PermissionResolver,
)
from .session import ToolExecutionTerminal
from .tools.base import Concurrency, Tool, ToolCall, ToolResult, ToolRuntime


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
    ):
        self.tool_registry = tool_registry
        self.tool_timeout = tool_timeout
        self.permission_resolver = permission_resolver or PermissionResolver(
            permission_policy or PermissionPolicy(),
            permission_approval_handler,
        )
        self.workspace_dir = (workspace_dir or Path.cwd()).resolve()
        self.cwd_provider = cwd_provider or (lambda: self.workspace_dir)
        self.runtime = ToolRuntime(
            workspace_dir=self.workspace_dir,
            cwd_provider=self.cwd_provider,
            emit_output=on_command_output,
        )

    def _invoke_tool(self, tool_call: ToolCall) -> ToolResult:
        """查找并执行【单个】工具，返回标准化 tool_result。"""
        tool = self.tool_registry.get(tool_call.name)
        if tool is None:
            return ToolResult.fail(err=f"Unknown tool: {tool_call.name}")

        runtime = replace(
            self.runtime,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
        )

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
        arguments = dict(permission.updated_arguments or tool_call.arguments)

        # 内层超时必须 ≤ 外层线程预算:模型可以给工具传很大的 timeout,
        # 不钳制的话外层先掐,工具内部的超时机制(如 execute_command 转后台)永远轮不到登场
        if isinstance(arguments.get("timeout"), (int, float)):
            arguments["timeout"] = min(arguments["timeout"], self.tool_timeout)

        try:
            tool_result = tool.call(arguments, runtime)
        except Exception as e:
            tool_result = ToolResult.fail(f"{type(e).__name__}: {e}")

        return tool_result

    def _current_cwd(self) -> Path:
        try:
            return self.cwd_provider().resolve()
        except Exception:
            return self.workspace_dir

    def _concurrency_for(self, tool_call: ToolCall) -> Concurrency:
        """查工具的并发策略;未知工具按最保守的 serial 处理(执行时本就会 fail)。"""
        tool = self.tool_registry.get(tool_call.name)
        return tool.concurrency if tool is not None else "serial"

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

        call_by_idx = dict(indexed_calls)
        budget = self.tool_timeout * ceil(len(indexed_calls) / max_workers)

        pool = ThreadPoolExecutor(max_workers=max_workers)
        fut_to_idx = {
            pool.submit(self._invoke_tool, tc): idx for idx, tc in indexed_calls
        }

        try:
            # 谁先跑完谁先渲染(实时),写回按下标(保序)
            for fut in as_completed(fut_to_idx, timeout=budget):
                idx = fut_to_idx[fut]
                result = fut.result()  # _invoke_tool 不抛异常,恒拿到 ToolResult
                if on_result:
                    on_result(result)
                out[idx] = ToolExecutionOutcome(
                    call=call_by_idx[idx],
                    result=result,
                    status="succeeded" if result.ok else "failed",
                )

        except (TimeoutError, FuturesTimeoutError):
            for idx, tc in indexed_calls:
                if idx not in out:
                    result = ToolResult.fail(
                        f"timeout: 超过 {budget}s 未完成(工具可能仍在后台运行)"
                    )
                    if on_result:
                        on_result(result)
                    out[idx] = ToolExecutionOutcome(
                        call=tc,
                        result=result,
                        status="timeout",
                    )

        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        return out

    def execute(
        self,
        tool_calls: list[ToolCall],
        on_call: Callable[[ToolCall], None] | None = None,
        on_result: Callable[[ToolResult], None] | None = None,
        max_workers: int = 8,
    ) -> list[ToolExecutionOutcome]:
        """按工具 concurrency 分流执行,返回 outcome 列表,顺序恒等于输入。

        只读批(parallel)整批丢进线程池并发;写/命令批(serial)逐个执行。
        两阶段不重叠——读批全跑完才跑写批——避免读写争抢同一资源(如同一文件)。
        串行批也逐个走 _run_concurrent_batch(单元素),因此超时/状态/保序逻辑
        与并发批完全一致。

        on_call 先按输入顺序全报一遍("这一轮要调这些工具"),再开跑。
        """
        if on_call:
            for tool_call in tool_calls:
                on_call(tool_call)

        parallel = [
            (i, tc)
            for i, tc in enumerate(tool_calls)
            if self._concurrency_for(tc) == "parallel"
        ]
        serial = [
            (i, tc)
            for i, tc in enumerate(tool_calls)
            if self._concurrency_for(tc) == "serial"
        ]

        slots: list[ToolExecutionOutcome | None] = [None] * len(tool_calls)

        # 阶段一:只读批并发
        for idx, slot in self._run_concurrent_batch(
            parallel, on_result, max_workers
        ).items():
            slots[idx] = slot

        # 阶段二:写/命令批串行——逐个丢单元素 pool,既保证串行又复用超时/状态逻辑
        for indexed in serial:
            for idx, slot in self._run_concurrent_batch(
                [indexed], on_result, max_workers
            ).items():
                slots[idx] = slot

        return [slot for slot in slots if slot is not None]
