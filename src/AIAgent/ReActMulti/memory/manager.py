"""MemoryManager:记忆系统对 Agent 暴露的唯一协作者。

与 ContextCompactor / ToolExecutor 一致的接法——Agent 只持有它、在主循环里
「喊一声」,记忆的所有具体逻辑(召回/提取/落盘)都收在 memory 包内部。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..llm import LLMClient
from .extract import extract_and_save
from .paths import ensure_memory_dir, memory_dir
from .prompt import build_memory_instructions
from .recall import build_recall_block


class MemoryManager:
    """主 Agent 的长期记忆协作者。

    Args:
        llm: 主对话用的 LLMClient。
        selector_llm: 可选,做召回/提取 side-query 的更便宜模型;默认复用 llm。
        directory: 可选,记忆目录;默认 paths.memory_dir()。
    """

    def __init__(
        self,
        llm: LLMClient,
        selector_llm: LLMClient | None = None,
        directory: Path | None = None,
    ) -> None:
        self.llm = llm
        self.selector_llm = selector_llm or llm
        self.directory = directory or memory_dir()
        ensure_memory_dir()

    def instructions(self) -> str:
        """注入 system prompt 的静态记忆指令段。"""
        return build_memory_instructions(self.directory)

    def recall_block(self, query: str) -> str:
        """针对本轮 query 的召回文本块(MEMORY.md 索引 + 相关记忆),无则 ""。"""
        return build_recall_block(query, self.selector_llm, self.directory)

    def extract(self, session_state: Any) -> int:
        """会话收口后从 transcript 提取并落盘记忆,返回写入条数(best-effort)。"""
        return extract_and_save(session_state, self.selector_llm, self.directory)
