import traceback
import contextlib
import httpx
import os
import io
import traceback
import contextlib

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def calculate(num1, num2):
    return num1 + num2


def web_search(query: str, max_results):
    resp = httpx.post(
        url="https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        results.append({"title": title, "snippet": content, "url": url})
    return results


def execute_python(code: str):
    buffer = io.StringIO()
    err_info = ""
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code)
    except Exception as e:
        err_info = traceback.format_exc()

    if err_info:
        return {"ok": False, "err": err_info, "content": ""}
    else:
        return {"ok": True, "err": "", "content": buffer.getvalue()}


import subprocess
import shlex


def execute_command(command: str, timeout: int = 20):
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "err": f"Command timed out after {timeout}s",
            "stdout": "",
            "stderr": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "err": str(e),
            "stdout": "",
            "stderr": "",
        }


def http_request(
    url: str,
    method: str = "GET",
    params: dict | None = None,
    body: dict | str | None = None,
    headers: dict | None = None,
):
    # params 是 GET 的
    kwargs: dict[str, Any] = {"params": params, "headers": headers, "timeout": 20}

    # body 是 POST 的，根据 body 格式选择是Content-Type: application/json还是原始字符串
    if isinstance(body, dict):
        kwargs["json"] = body
    elif isinstance(body, str):
        kwargs["content"] = body

    resp = httpx.request(method, url, **kwargs)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    try:
        body = resp.json() if "application/json" in content_type else resp.text
    except Exception:
        body = resp.text

    if isinstance(body, str) and len(body) > 4000:
        body = body[:4000]
        truncated = True
    else:
        truncated = False

    return {
        "ok": True,
        "err": "",
        "response": body,
    }


# 文件操作工具链
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"


def _safe_path(path: str) -> Path:
    safe_path = (WORKSPACE_DIR / path).resolve()
    if not safe_path.is_relative_to(WORKSPACE_DIR):
        raise ValueError("Unsafe path")
    return safe_path


def list_files(directory: str = "."):
    try:
        safe_directory = _safe_path(directory)
        files = [entry.name for entry in safe_directory.iterdir() if entry.is_file()]
        dirs = [entry.name for entry in safe_directory.iterdir() if entry.is_dir()]
        return {"ok": True, "files": files, "dirs": dirs}
    except Exception as e:
        return {"ok": False, "err": str(e), "files": [], "dirs": []}


def read_file(file: str, max_chars: int = 8000):
    try:
        safe_path = _safe_path(file)

        if not safe_path.is_file():
            return {"ok": False, "err": "Not a file", "content": ""}

        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)

        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        return {"ok": True, "err": "", "content": content, "truncated": truncated}
    except Exception as e:
        return {"ok": False, "err": str(e), "content": ""}


def write_file(file: str, content: str, overwrite: bool = True):
    try:
        safe_path = _safe_path(file)

        if safe_path.exists() and safe_path.is_dir():
            return {"ok": False, "err": "Path is a directory"}

        if safe_path.exists() and not overwrite:
            return {"ok": False, "err": "File already exists"}

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")

        return {
            "ok": True,
            "err": "",
            "message": "File written",
            "file": str(safe_path.relative_to(WORKSPACE_DIR)),
            "chars": len(content),
        }
    except Exception as e:
        return {"ok": False, "err": str(e)}


write_file_tool = Tool(
    name="write_file",
    description="Write content to a file inside the workspace. Creates parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "The relative path of the file to write",
            },
            "content": {
                "type": "string",
                "description": "The full content to write to the file",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite the file if it already exists",
                "default": True,
            },
        },
        "required": ["file", "content"],
    },
    func=write_file,
)


def edit_file(file: str, old_text: str, new_text: str):
    try:
        safe_path = _safe_path(file)

        if not safe_path.is_file():
            return {"ok": False, "err": "Not a file"}

        content = safe_path.read_text(encoding="utf-8")

        count = content.count(old_text)
        if count == 0:
            return {"ok": False, "err": "old_text not found"}
        if count > 1:
            return {
                "ok": False,
                "err": f"old_text found {count} times, replacement is ambiguous",
            }

        updated = content.replace(old_text, new_text, 1)
        safe_path.write_text(updated, encoding="utf-8")

        return {"ok": True, "err": "", "message": "File updated"}
    except Exception as e:
        return {"ok": False, "err": str(e)}


list_files_tool = Tool(
    name="list_files",
    description="List all files in a specified directory.",
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "The directory to list files in",
            },
        },
        "required": ["directory"],
    },
    func=list_files,
)

read_file_tool = Tool(
    name="read_file",
    description="Read the content of a file in the workspace. Returns the text content of the file.",
    parameters={
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "The relative path of the file to read",
            },
        },
        "required": ["file"],
    },
    func=read_file,
)

edit_file_tool = Tool(
    name="edit_file",
    description="Replace exact text in a file inside the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "file": {
                "type": "string",
                "description": "The file path inside the workspace",
            },
            "old_text": {
                "type": "string",
                "description": "The exact text to replace",
            },
            "new_text": {
                "type": "string",
                "description": "The replacement text",
            },
        },
        "required": ["file", "old_text", "new_text"],
    },
    func=edit_file,
)


calculate_tool = Tool(
    name="calculate",
    description="Add two numbers and return the sum.",
    parameters={
        "type": "object",
        "properties": {
            "num1": {"type": "number", "description": "The first number"},
            "num2": {"type": "number", "description": "The second number"},
        },
        "required": ["num1", "num2"],
    },
    func=calculate,
)

web_search_tool = Tool(
    name="web_search",
    description="Search the web for information about a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
            },
        },
        "required": ["query", "max_results"],
    },
    func=web_search,
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

http_request_tool = Tool(
    name="http_request",
    description="Make an HTTP request to a specified URL with given parameters, body, and headers.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to send the request to"},
            "method": {
                "type": "string",
                "description": "The HTTP method to use",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            },
            "params": {
                "type": "object",
                "description": "The query parameters to include in the request",
            },
            "body": {
                "type": ["object", "string"],
                "description": "The body of the request, if applicable",
            },
            "headers": {
                "type": "object",
                "description": "The headers to include in the request",
            },
        },
        "required": ["url", "method"],
    },
    func=http_request,
)


tools: list[Tool] = []

tools.append(calculate_tool)
tools.append(web_search_tool)
tools.append(execute_python_tool)
tools.append(execute_command_tool)

tools.append(http_request_tool)
tools.append(list_files_tool)
tools.append(read_file_tool)
tools.append(write_file_tool)
tools.append(edit_file_tool)
