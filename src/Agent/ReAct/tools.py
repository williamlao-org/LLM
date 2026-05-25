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


def http_request(
    url: str,
    method: str = "GET",
    params: dict | None = None,
    body: dict | str | None = None,
    headers: dict | None = None,
):
    try:
        kwargs = {"params": params, "headers": headers, "timeout": 20}
        if isinstance(body, dict):
            kwargs["json"] = body
        elif isinstance(body, str):
            kwargs["content"] = body

        resp = httpx.request(method, url, **kwargs)

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            resp_body = resp.json()
        else:
            resp_body = resp.text

        return {
            "status_code": resp.status_code,
            "body": resp_body,
        }
    except Exception as e:
        return {"status_code": None, "body": str(e)}


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

# 文件操作工具链
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"


def _safe_path(path: str):
    safe_path = (WORKSPACE_DIR / path).resolve()
    if not safe_path.is_relative_to(WORKSPACE_DIR):
        raise ValueError("Unsafe path")
    return safe_path


def list_files(directory: str="."):
    try:
        safe_directory = _safe_path(directory)
        files = [entry.name for entry in safe_directory.iterdir() if entry.is_file()]
        dirs = [entry.name for entry in safe_directory.iterdir() if entry.is_dir()]
        return {"ok": True, "files": files, "dirs": dirs}
    except Exception as e:
        return {"ok": False, "err": str(e), "files": [], "dirs": []}



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

tools: list[Tool] = []

tools.append(calculate_tool)
tools.append(web_search_tool)
tools.append(execute_python_tool)
tools.append(http_request_tool)
tools.append(list_files_tool)
