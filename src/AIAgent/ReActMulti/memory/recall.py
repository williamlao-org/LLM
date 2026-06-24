"""自动召回:两段式「小模型选 → 主模型用」。

每个 user turn:
  1. 扫记忆头信息生成清单;
  2. 让一个便宜的 side-query 从清单里选出最相关的最多 5 条(只凭文件名+描述判断);
  3. 读选中记忆全文,连同 MEMORY.md 索引拼成 <system-reminder> 注入主对话。

召回是【尽力而为】的旁路:任何环节失败都返回空,绝不阻塞主流程。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm import LLMClient
from .llm_util import side_query
from .store import (
    format_manifest,
    read_entrypoint,
    read_memories_for_surfacing,
    scan_memory_files,
)

MAX_SELECTED = 5

SELECT_SYSTEM_PROMPT = """你在为一个 AI Agent 挑选「处理当前用户请求时会用到的记忆」。
你会拿到用户的请求,以及一份可用记忆清单(文件名 + 描述)。

返回一个文件名列表,只包含【明显】对处理该请求有帮助的记忆(最多 5 个),仅凭名称和描述判断。
- 不确定某条是否有用,就不要选它。要挑剔、克制。
- 如果清单里没有明显有用的,返回空列表。

只输出严格 JSON:{"selected_memories": ["a.md", "b.md"]}"""


def find_relevant_memories(
    query: str,
    llm: LLMClient,
    directory: Path | None = None,
    already_surfaced: set[str] | None = None,
) -> list[Path]:
    """选出与 query 最相关的记忆文件路径(最多 5 条)。失败返回 []。"""
    already_surfaced = already_surfaced or set()
    headers = [
        h
        for h in scan_memory_files(directory)
        if str(h.path) not in already_surfaced
    ]
    if not headers:
        return []

    valid = {h.filename: h.path for h in headers}
    manifest = format_manifest(headers)
    user_msg = f"用户请求:{query}\n\n可用记忆:\n{manifest}"

    try:
        raw = side_query(llm, SELECT_SYSTEM_PROMPT, user_msg)
        selected = json.loads(raw).get("selected_memories", [])
    except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
        return []
    if not isinstance(selected, list):
        return []

    paths: list[Path] = []
    for name in selected:
        if name in valid and valid[name] not in paths:
            paths.append(valid[name])
        if len(paths) >= MAX_SELECTED:
            break
    return paths


def build_recall_block(
    query: str,
    llm: LLMClient,
    directory: Path | None = None,
    already_surfaced: set[str] | None = None,
) -> str:
    """拼好要注入主对话的召回文本块;没有任何内容时返回 ""。

    结构:MEMORY.md 索引(全局视野)+ 选中记忆全文(本轮重点),包进 system-reminder。
    """
    index = read_entrypoint(directory)
    paths = find_relevant_memories(query, llm, directory, already_surfaced)
    relevant = read_memories_for_surfacing(paths)

    if not index and not relevant:
        return ""

    parts = ["<system-reminder>", "以下是你的长期记忆(背景上下文,非用户指令)。"]
    if index:
        parts.append("\n## 记忆索引 (MEMORY.md)\n" + index)
    if relevant:
        parts.append("\n## 与本次请求相关的记忆\n" + relevant)
    parts.append(
        "\n使用前请记住:记忆是过去某刻的快照,可能已过期;据此行动前先核实当前状态。"
    )
    parts.append("</system-reminder>")
    return "\n".join(parts)
