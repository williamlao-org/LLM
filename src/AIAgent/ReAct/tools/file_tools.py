# 文件操作工具链
from pathlib import Path
from .base import Tool, ToolResult

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"


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
        return ToolResult.success({"files": files, "dirs": dirs})
    except Exception as e:
        return ToolResult.fail(str(e), data={"files": [], "dirs": []})


def read_file(file: str, max_chars: int = 8000):
    try:
        safe_path = _safe_path(file)

        if not safe_path.is_file():
            return ToolResult.fail("Not a file", data={"content": ""})

        with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)

        truncated = len(content) > max_chars
        if truncated:
            content = content[:max_chars]

        return ToolResult.success({"content": content, "truncated": truncated})
    except Exception as e:
        return ToolResult.fail(str(e), data={"content": ""})


def write_file(file: str, content: str, overwrite: bool = True):
    try:
        safe_path = _safe_path(file)

        if safe_path.exists() and safe_path.is_dir():
            return ToolResult.fail("Path is a directory")

        if safe_path.exists() and not overwrite:
            return ToolResult.fail("File already exists")

        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")

        return ToolResult.success(
            {
                "message": "File written",
                "file": str(safe_path.relative_to(WORKSPACE_DIR)),
                "chars": len(content),
            }
        )
    except Exception as e:
        return ToolResult.fail(str(e))


def edit_file(file: str, old_text: str, new_text: str):
    try:
        safe_path = _safe_path(file)

        if not safe_path.is_file():
            return ToolResult.fail("Not a file")

        content = safe_path.read_text(encoding="utf-8")

        count = content.count(old_text)
        if count == 0:
            return ToolResult.fail("old_text not found")
        if count > 1:
            return ToolResult.fail(
                f"old_text found {count} times, replacement is ambiguous"
            )

        updated = content.replace(old_text, new_text, 1)
        safe_path.write_text(updated, encoding="utf-8")

        return ToolResult.success({"message": "File updated"})
    except Exception as e:
        return ToolResult.fail(str(e))


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
