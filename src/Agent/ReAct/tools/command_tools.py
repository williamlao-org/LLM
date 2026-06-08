import traceback
import contextlib
import io

import subprocess
import shlex

from pathlib import Path

from .base import Tool, ToolResult


def execute_python(code: str):
    buffer = io.StringIO()
    err_info = ""
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code)
    except Exception as e:
        err_info = traceback.format_exc()

    if err_info:
        return ToolResult.fail(err_info)
    else:
        return ToolResult.success(buffer.getvalue())


WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"


def _safe_path(path: str) -> Path:
    safe_path = (WORKSPACE_DIR / path).resolve()
    if not safe_path.is_relative_to(WORKSPACE_DIR):
        raise ValueError("Unsafe path")
    return safe_path


def execute_command(command: str, timeout: int = 20):
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        data = {
            "returncode": result.returncode,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
        }

        if result.returncode == 0:
            return ToolResult.success(data)

        return ToolResult.fail(
            err=f"Command exited with code {result.returncode}",
            data=data,
        )

    except subprocess.TimeoutExpired:
        return ToolResult.fail(
            err=f"Command timed out after {timeout}s",
        )

    except Exception as e:
        return ToolResult.fail(
            err=f"{type(e).__name__}: {e}",
        )


execute_python_tool = Tool(
    name="execute_python",
    description="Execute python code and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The python code to execute"},
        },
        "required": ["code"],
    },
    func=execute_python,
)

execute_command_tool = Tool(
    name="execute_command",
    description="Execute a command and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute",
            },
        },
        "required": ["command"],
    },
    func=execute_command,
)
