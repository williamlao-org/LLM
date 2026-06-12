import os
import queue
import shlex
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable

from .base import Tool, ToolResult

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE_DIR.mkdir(exist_ok=True)

# ── session 级别 cwd ──────────────────────────────────────────────────────────

_cwd_lock = threading.Lock()
_cwd: Path = WORKSPACE_DIR


def get_cwd() -> Path:
    with _cwd_lock:
        return _cwd


def _set_cwd(new_cwd: Path) -> None:
    with _cwd_lock:
        global _cwd
        _cwd = new_cwd


# ── 后台任务注册表 ────────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _register_task(task_id: str, proc: subprocess.Popen, lines: list[str], done: threading.Event) -> None:
    with _tasks_lock:
        _tasks[task_id] = {"proc": proc, "lines": lines, "done": done}


def get_task_output(task_id: str) -> ToolResult:
    """查询后台任务的当前输出和状态。"""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return ToolResult.fail(f"Unknown task_id: {task_id}")
    done = task["done"].is_set()
    output = "".join(task["lines"])
    return ToolResult.success({
        "task_id": task_id,
        "done": done,
        "returncode": task["proc"].returncode if done else None,
        "output": output[-8000:],
    })


# ── execute_command ───────────────────────────────────────────────────────────

MAX_OUTPUT_CHARS = 8000


def execute_command(
    command: str,
    timeout: int = 20,
    run_in_background: bool = False,
    on_output: Callable[[str], None] | None = None,
) -> ToolResult:
    """
    执行 shell 命令，对齐 Claude Code BashTool 的核心机制：

    - cwd 注入法：在命令末尾追加 `&& pwd -P > $tmpfile`，命令执行完后
      读回临时文件来更新 session cwd，能正确捕获命令内部 cd 的效果。
    - 流式输出：后台读线程实时触发 on_output 回调。
    - 超时转后台：前台命令超时后不 kill，转为后台任务返回 task_id。
    - run_in_background：立即后台运行，返回 task_id。
    """
    try:
        cwd = get_cwd()

        # 注入 cwd 追踪：用临时文件，和 Claude Code 的 claude-{id}-cwd 一致
        cwd_file = Path(tempfile.mktemp(prefix="react-cwd-"))
        # 末尾追加 `&& pwd -P > tmpfile`，无论命令成败都不影响返回码
        # （pwd -P 只在主命令成功时才写，和 Claude Code 的 &&  行为一致）
        injected = f"eval {shlex.quote(command)} && pwd -P > {shlex.quote(str(cwd_file))}"

        proc = subprocess.Popen(
            ["/bin/bash", "-c", injected],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr 合并进 stdout，和 Claude Code 一致
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        return ToolResult.fail(f"命令不存在: {command.split()[0] if command.split() else command}")
    except Exception as e:
        return ToolResult.fail(f"{type(e).__name__}: {e}")

    output_lines: list[str] = []
    done_event = threading.Event()

    def _reader():
        for line in proc.stdout:
            output_lines.append(line)
            if on_output:
                on_output(line)
        proc.wait()
        # 先更新 cwd，再 set done_event，保证调用方 wait() 返回时 cwd 已就绪
        _read_cwd_file(cwd_file)
        done_event.set()

    threading.Thread(target=_reader, daemon=True).start()

    if run_in_background:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        _register_task(task_id, proc, output_lines, done_event)
        return ToolResult.success({
            "task_id": task_id,
            "message": f"命令已在后台运行，用 get_task_output('{task_id}') 查结果。",
        })

    finished = done_event.wait(timeout=timeout)

    if not finished:
        # 超时：不 kill，转后台
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        _register_task(task_id, proc, output_lines, done_event)
        return ToolResult.success({
            "task_id": task_id,
            "timed_out": True,
            "message": f"命令超过 {timeout}s 未完成，已转为后台任务 {task_id}。",
            "output_so_far": "".join(output_lines)[-MAX_OUTPUT_CHARS:],
        })

    output = "".join(output_lines)
    if len(output) > MAX_OUTPUT_CHARS:
        output = f"[...截断，仅显示末尾]\n{output[-MAX_OUTPUT_CHARS:]}"

    returncode = proc.returncode
    data = {"returncode": returncode, "output": output}

    if returncode == 0:
        return ToolResult.success(data)
    return ToolResult.fail(err=f"命令以退出码 {returncode} 结束", data=data)


def _read_cwd_file(cwd_file: Path) -> None:
    """读取 pwd -P 写入的临时文件，更新 session cwd，然后删除文件。

    和 Claude Code Shell.ts 里的 readFileSync + unlinkSync 逻辑对应。
    只在文件存在时（即主命令成功退出）才更新，失败命令不改变 cwd。
    """
    try:
        new_cwd = Path(cwd_file.read_text().strip())
        if new_cwd.is_dir():
            _set_cwd(new_cwd)
    except Exception:
        pass  # 文件不存在（命令失败）或路径非法，静默忽略
    finally:
        try:
            cwd_file.unlink()
        except Exception:
            pass


# ── 工具定义 ──────────────────────────────────────────────────────────────────

execute_command_tool = Tool(
    name="execute_command",
    description=(
        "Execute a shell command in the workspace. "
        "The working directory persists across calls (cd works). "
        "Long-running commands auto-background after timeout and return a task_id. "
        "Set run_in_background=true to background immediately."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds before auto-backgrounding (default: 20)",
                "default": 20,
            },
            "run_in_background": {
                "type": "boolean",
                "description": "If true, run immediately in background and return task_id",
                "default": False,
            },
        },
        "required": ["command"],
    },
    func=execute_command,
)

get_task_output_tool = Tool(
    name="get_task_output",
    description="Get the current output and status of a background task.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task_id returned by execute_command",
            },
        },
        "required": ["task_id"],
    },
    func=get_task_output,
)
