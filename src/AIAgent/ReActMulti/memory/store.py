"""记忆落盘 + 索引维护:纯文件操作,不依赖 LLM,可独立单测。

文件布局(对标 memdir):
    memory_store/
    ├── MEMORY.md          ← 索引,一行一条指针,始终注入上下文
    ├── user-role.md       ← 单条记忆,带 frontmatter
    └── ...

frontmatter 只用三字段(name/description/type),所以手写极简解析,不引 PyYAML。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .paths import MEMORY_INDEX, entrypoint_path, memory_dir
from .types import MemoryType, parse_memory_type

# 索引(MEMORY.md)注入上下文的体量上限,超过则截断并附警告(对标 memdir)。
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000
# 扫描时只读前若干行取 frontmatter,避免把整个大文件读进来。
FRONTMATTER_MAX_LINES = 30
# 召回时单条记忆全文注入的体量上限,防止一条巨型记忆撑爆上下文。
MAX_MEMORY_CHARS = 4_000
# 一次最多纳入索引/清单的记忆条数。
MAX_MEMORY_FILES = 200


@dataclass
class MemoryHeader:
    """一条记忆的轻量头信息(不含正文),用于清单/索引/召回选择。"""

    filename: str  # 相对 memory_dir 的文件名,如 "user-role.md"
    path: Path
    mtime: float
    description: str | None
    type: MemoryType | None


# ── frontmatter 解析 / 生成 ───────────────────────────────────────────
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """拆出 (frontmatter dict, 正文)。没有合法 frontmatter 时返回 ({}, 原文)。

    只解析 `key: value` 形式的单行字段(name/description/type 够用),value 去除
    首尾空白和成对引号。
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_fm, body = m.group(1), m.group(2)
    fm: dict[str, str] = {}
    for line in raw_fm.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            fm[key] = value
    return fm, body


def dump_frontmatter(name: str, description: str, type_: str, body: str) -> str:
    """把三字段 + 正文拼成带 frontmatter 的完整文件文本。"""
    # 描述里若含换行/冒号会破坏单行格式,压成一行。
    desc = " ".join(description.splitlines()).strip()
    return (
        "---\n"
        f"name: {name.strip()}\n"
        f"description: {desc}\n"
        f"type: {type_.strip()}\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


# ── 文件名 sanitize ───────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """把记忆名归一成安全的文件名 slug(小写、短横线)。"""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "memory"


# ── 扫描 ──────────────────────────────────────────────────────────────
def _read_head(path: Path, max_lines: int) -> str:
    """只读文件前 max_lines 行(够覆盖 frontmatter)。"""
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line)
    return "".join(lines)


def scan_memory_files(directory: Path | None = None) -> list[MemoryHeader]:
    """扫描记忆目录,返回头信息列表(跳过 MEMORY.md),按 mtime 倒序。

    目录不存在或读不了时返回 []。对标 memoryScan.ts 的 scanMemoryFiles。
    """
    directory = directory or memory_dir()
    if not directory.is_dir():
        return []

    headers: list[MemoryHeader] = []
    for path in directory.rglob("*.md"):
        if path.name == MEMORY_INDEX or not path.is_file():
            continue
        try:
            head = _read_head(path, FRONTMATTER_MAX_LINES)
            fm, _ = parse_frontmatter(head)
            headers.append(
                MemoryHeader(
                    filename=str(path.relative_to(directory)),
                    path=path,
                    mtime=path.stat().st_mtime,
                    description=fm.get("description") or None,
                    type=parse_memory_type(fm.get("type")),
                )
            )
        except OSError:
            continue  # 单个文件读失败不影响其余

    headers.sort(key=lambda h: h.mtime, reverse=True)
    return headers[:MAX_MEMORY_FILES]


def format_manifest(headers: list[MemoryHeader]) -> str:
    """把头信息列成清单:`- [type] file.md: description`,供选择器/搜索使用。"""
    lines: list[str] = []
    for h in headers:
        tag = f"[{h.type}] " if h.type else ""
        desc = f": {h.description}" if h.description else ""
        lines.append(f"- {tag}{h.filename}{desc}")
    return "\n".join(lines)


# ── 写入 + 索引 ───────────────────────────────────────────────────────
def write_memory_file(
    name: str,
    description: str,
    type_: str,
    content: str,
    directory: Path | None = None,
) -> Path:
    """写一条记忆到 `<slug>.md`(同名覆盖,兼作 update),返回文件路径。

    写完不自动 rebuild_index——调用方(工具/提取器)负责在写完后调一次,
    便于批量写入只重建一次索引。
    """
    directory = directory or memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slugify(name)}.md"
    path.write_text(
        dump_frontmatter(name, description, type_, content), encoding="utf-8"
    )
    return path


def rebuild_index(directory: Path | None = None) -> Path:
    """扫描所有记忆文件,重新生成 MEMORY.md 索引(一行一条指针)。

    每次写记忆后调用。从磁盘真值重建而非增量追加——自愈、永不与文件漂移。
    """
    directory = directory or memory_dir()
    directory.mkdir(parents=True, exist_ok=True)
    headers = scan_memory_files(directory)

    lines = ["# MEMORY.md", "", "记忆索引(每行一条指针,正文在各自文件里)。", ""]
    for h in headers:
        title = h.filename[:-3] if h.filename.endswith(".md") else h.filename
        desc = f" — {h.description}" if h.description else ""
        lines.append(f"- [{title}]({h.filename}){desc}")
    if not headers:
        lines.append("_(暂无记忆)_")

    index_path = directory / MEMORY_INDEX
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


# ── 读取(注入上下文) ─────────────────────────────────────────────────
def read_entrypoint(directory: Path | None = None) -> str:
    """读 MEMORY.md,套行数/字节上限截断(对标 truncateEntrypointContent)。

    文件不存在返回 ""。
    """
    path = (directory / MEMORY_INDEX) if directory else entrypoint_path()
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not raw:
        return ""

    lines = raw.split("\n")
    truncated = False
    if len(lines) > MAX_INDEX_LINES:
        lines = lines[:MAX_INDEX_LINES]
        truncated = True
    out = "\n".join(lines)
    if len(out.encode("utf-8")) > MAX_INDEX_BYTES:
        out = out.encode("utf-8")[:MAX_INDEX_BYTES].decode("utf-8", "ignore")
        truncated = True
    if truncated:
        out += (
            f"\n\n> 警告:{MEMORY_INDEX} 超出上限,仅加载了部分。"
            "请把索引条目压到一行、细节移进各自的记忆文件。"
        )
    return out


def read_memories_for_surfacing(paths: list[Path]) -> str:
    """读选中记忆的全文(单条超限则截断),拼成可注入的文本块。

    每条以 `### filename` 起头,便于模型分辨来源。
    """
    blocks: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if len(text) > MAX_MEMORY_CHARS:
            text = text[:MAX_MEMORY_CHARS] + "\n…(已截断)"
        blocks.append(f"### {path.name}\n{text}")
    return "\n\n".join(blocks)
