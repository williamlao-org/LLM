"""记忆工具:让模型显式保存 / 搜索长期记忆。

仅挂给主 Agent(见 main.py 的装配)。记忆目录由 memory.paths 解析,不走
ToolRuntime——记忆本就在 workspace 沙箱之外,不受 file_tools 的 _safe_path 约束。
"""

from __future__ import annotations

from ..memory.store import (
    format_manifest,
    rebuild_index,
    scan_memory_files,
    write_memory_file,
)
from ..memory.types import MEMORY_TYPES
from .base import Tool, ToolResult, ToolRuntime


def save_memory(
    name: str,
    description: str,
    type: str,
    content: str,
    runtime: ToolRuntime | None = None,
) -> ToolResult:
    """写一条记忆并重建 MEMORY.md 索引。同名覆盖,兼作更新。"""
    if type not in MEMORY_TYPES:
        return ToolResult.fail(
            f"非法 type '{type}',必须是: {', '.join(MEMORY_TYPES)}"
        )
    if not str(name).strip() or not str(content).strip():
        return ToolResult.fail("name 和 content 不能为空")
    try:
        path = write_memory_file(name, description or "", type, content)
        rebuild_index()
        return ToolResult.success(
            {"message": "记忆已保存", "file": path.name, "type": type}
        )
    except OSError as e:
        return ToolResult.fail(str(e))


def search_memory(query: str = "", runtime: ToolRuntime | None = None) -> ToolResult:
    """返回记忆清单(文件名 + 类型 + 描述),供模型判断有无可复用/可更新的记忆。

    query 目前仅作语义提示保留,返回的是全量清单(条数有上限),由模型自行筛选。
    """
    headers = scan_memory_files()
    manifest = format_manifest(headers)
    return ToolResult.success(
        {"count": len(headers), "memories": manifest or "(暂无记忆)"}
    )


save_memory_tool = Tool(
    name="save_memory",
    description=(
        "把一条值得【跨会话长期保留】的记忆写入记忆系统(会自动维护 MEMORY.md 索引)。"
        "只存无法从代码/git 推导的内容:用户画像、工作偏好与纠正、项目背景、外部资源指针。"
        "同名记忆会被覆盖,可用于更新。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "记忆名(短横线 kebab-case),也用作文件名",
            },
            "description": {
                "type": "string",
                "description": "一句话描述,未来据此判断相关性,写具体些",
            },
            "type": {
                "type": "string",
                "enum": list(MEMORY_TYPES),
                "description": "记忆类型",
            },
            "content": {
                "type": "string",
                "description": "记忆正文;feedback/project 类型请包含 Why 和 How to apply",
            },
        },
        "required": ["name", "description", "type", "content"],
    },
    call=lambda args, runtime: save_memory(**args, runtime=runtime),
    # 写自己的记忆是低风险的有意行为,默认放行(沿用 Tool 的默认 allow 权限)。
)

search_memory_tool = Tool(
    name="search_memory",
    description=(
        "列出当前所有长期记忆(文件名 + 类型 + 描述)。在保存新记忆前用它检查是否已有"
        "可更新的记忆,或在需要回忆时查看有哪些记忆可读。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "想查找的主题(可选,当前返回全量清单供你筛选)",
            },
        },
        "required": [],
    },
    call=lambda args, runtime: search_memory(**args, runtime=runtime),
    is_concurrency_safe=lambda args: True,
)
