from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from ..permission import PermissionCheckResult

# 并发策略:parallel = 只读/无本地副作用,可丢线程池并发;
# serial = 会改 workspace 或有副作用,必须串行执行。
# 默认 serial:新工具不显式声明时按最保守处理,宁可慢不要竞态。
Concurrency = Literal["parallel", "serial"]


@dataclass
class ToolRuntime:
    """执行器传给工具的运行期上下文,不属于模型可见参数。

    只放"程序运行时能力":渲染/进度回调、调用标识、workspace/cwd、取消信号等。
    不放模型生成的业务参数(command/file/timeout...),那些只走 ToolCall.arguments。
    """

    # 当前工具调用标识:用于日志、进度事件、后台任务关联。
    tool_name: str = ""
    tool_call_id: str = ""

    # 当前会话的本地执行边界。工具需要定位 workspace/cwd 时优先用这里,
    # 避免各工具自己猜 Path.cwd() 或维护重复状态。
    workspace_dir: Path | None = None
    cwd_provider: Callable[[], Path] | None = None

    # 文本流式输出:例如 shell stdout。命名保持通用,不绑定 command 工具。
    emit_output: Callable[[str], None] | None = None

    # 结构化进度事件:未来可用于下载进度、批处理进度、后台任务状态等。
    emit_progress: Callable[[dict[str, Any]], None] | None = None

    # 取消信号:未来用户中断/上层 abort 时,长任务工具可主动停止。
    is_cancelled: Callable[[], bool] | None = None


def _default_check_permission(
    args: dict[str, Any], runtime: ToolRuntime
) -> PermissionCheckResult:
    return PermissionCheckResult(
        "allow",
        f"{runtime.tool_name or 'tool'}: allowed by default tool permission",
        source="tool_default",
    )


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    call: Callable[[dict[str, Any], ToolRuntime], "ToolResult"]
    check_permission: Callable[[dict[str, Any], ToolRuntime], PermissionCheckResult] = (
        _default_check_permission
    )
    concurrency: Concurrency = "serial"

    def to_dict(self):
        # concurrency 是系统调度用的内部元数据,不喂给模型(模型只管想调什么)。
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class ToolResult:
    ok: bool
    err: str = ""
    data: Any = None

    @classmethod
    def success(cls, data=None) -> "ToolResult":
        return cls(ok=True, err="", data=data)

    @classmethod
    def fail(cls, err: str, data=None) -> "ToolResult":
        return cls(ok=False, err=err, data=data)

    def to_dict(self):
        return {
            "ok": self.ok,
            "err": self.err,
            "data": self.data,
        }
