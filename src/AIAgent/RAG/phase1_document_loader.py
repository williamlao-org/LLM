"""
文档加载器

职责：从 docs/ 目录读取所有文档，返回结构化的文档列表。

设计思路：
  每个文档加载后变成一个 Document 对象，包含：
  - content: 文档的原始文本内容
  - metadata: 元数据（文件名、路径等），后续检索时可以追溯来源
"""

import hashlib
import mimetypes
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, get_args

try:
    # pyrefly: ignore [missing-import]
    from loaders.csv_loader import CsvLoader
    from loaders.docx_loader import DocxLoader
    from loaders.json_loader import JsonLoader
    from loaders.pdf_loader import PdfLoader
    from loaders.types import DocumentPart, MetadataValue
except (ImportError, ModuleNotFoundError):
    from .loaders.csv_loader import CsvLoader
    from .loaders.docx_loader import DocxLoader
    from .loaders.json_loader import JsonLoader
    from .loaders.pdf_loader import PdfLoader
    from .loaders.types import DocumentPart, MetadataValue


# ── 支持的文件格式（单一真实来源） ──────────────────────────────
# 新增格式时：1) 在此处添加扩展名  2) 在 DocumentLoader.loaders 中注册对应的加载函数
SupportedFormat = Literal[
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".html",
    ".csv",
    ".json",
    ".jsonl",
    ".xlsx",
]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(get_args(SupportedFormat))


@dataclass(frozen=True)
class LoaderSpec:
    """文件扩展名对应的加载器注册信息"""

    name: str
    load: Callable[[Path], list[DocumentPart]]


@dataclass
class Document:
    """表示一个加载后的文档"""

    content: str  # 文档文本内容
    metadata: dict[str, MetadataValue] = field(default_factory=dict)  # 元数据

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        source = self.metadata.get("source", "?")
        page = self.metadata.get("page")
        part_index = self.metadata.get("part_index")
        part_total = self.metadata.get("part_total")

        parts = [f"source={source}"]
        if page is not None:
            parts.append(f"page={page}")
        if (
            isinstance(part_index, int)
            and isinstance(part_total, int)
            and part_total > 1
        ):
            parts.append(f"part={part_index + 1}/{part_total}")

        parts.append(f"len={len(self.content)}")
        parts.append(f"preview='{preview}...'")

        return f"Document({', '.join(parts)})"


class DocumentLoader:
    def __init__(self):
        self.csv_loader = CsvLoader()
        self.docx_loader = DocxLoader()
        self.json_loader = JsonLoader()
        self.pdf_loader = PdfLoader()
        self.loaders: dict[SupportedFormat, LoaderSpec] = {
            ".txt": LoaderSpec("text", self._load_text),
            ".md": LoaderSpec("markdown", self._load_markdown),
            ".csv": LoaderSpec("csv", self.csv_loader.load),
            ".json": LoaderSpec("json", self.json_loader.load),
            ".pdf": LoaderSpec("pdf", self.pdf_loader.load),
            ".docx": LoaderSpec("docx", self.docx_loader.load),
        }

        # 运行时校验：loaders 注册的格式必须是 SupportedFormat 的子集
        undeclared_extensions = set(self.loaders.keys()) - SUPPORTED_EXTENSIONS
        if undeclared_extensions:
            raise RuntimeError(
                f"loaders 中注册了未在 SupportedFormat 中声明的格式: {undeclared_extensions}"
            )

    def load_documents(self, docs_dir: str | Path) -> list[Document]:
        docs_dir = Path(docs_dir).resolve()
        if not docs_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

        documents: list[Document] = []
        for filepath in sorted(docs_dir.iterdir()):
            # 获取文件扩展名
            if not filepath.is_file():
                continue

            ext = filepath.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            if ext not in self.loaders:
                print(f"  ⚠️ 跳过: {filepath.name} (格式 {ext} 已声明但未实现 loader)")
                continue

            try:
                file_documents = self.load_document(filepath)
                char_count = sum(len(document.content) for document in file_documents)
                print(
                    f"  ✅ 已加载: {filepath.name} "
                    f"({len(file_documents)} 个片段, {char_count} 字符)"
                )

            except Exception as e:
                print(f"  ⚠️ 跳过失败文件: {filepath.name} ({e})")
                continue

            documents.extend(file_documents)

        print(f"共加载 {len(documents)} 个文档")
        return documents

    def load_document(self, doc_path: Path | str) -> list[Document]:
        doc_path = Path(doc_path).resolve()

        if not doc_path.exists():
            raise FileNotFoundError(f"文件不存在: {doc_path}")

        if not doc_path.is_file():
            raise ValueError(f"不是文件: {doc_path}")

        loader_spec = self.loaders.get(doc_path.suffix.lower(), None)
        if loader_spec is None:
            raise ValueError(f"不支持 {doc_path.suffix} 类型文件")

        parts = [
            part
            for part in loader_spec.load(doc_path)
            if part.content.strip()
        ]
        if not parts:
            raise ValueError(f"文件{doc_path.resolve()}内容为空")

        documents: list[Document] = []
        part_total = len(parts)
        for part_index, part in enumerate(parts):
            part_type = str(part.metadata.get("part_type", "text"))
            metadata = {
                **part.metadata,
                **self._build_metadata(doc_path, part.content, loader_spec.name),
                "part_index": part_index,
                "part_total": part_total,
                "part_type": part_type,
            }
            documents.append(Document(content=part.content, metadata=metadata))

        return documents

    def _load_text(self, filepath: Path) -> list[DocumentPart]:
        return [
            DocumentPart(
                content=filepath.read_text(encoding="utf-8"),
                metadata={"part_type": "text"},
            )
        ]

    def _load_markdown(self, filepath: Path) -> list[DocumentPart]:
        return [
            DocumentPart(
                content=filepath.read_text(encoding="utf-8"),
                metadata={"part_type": "markdown"},
            )
        ]

    def _build_metadata(
        self,
        doc_path: Path,
        content: str,
        loader_name: str,
    ) -> dict[str, MetadataValue]:
        resolved_path = doc_path.resolve()
        stat = resolved_path.stat()
        mime_type, _ = mimetypes.guess_type(resolved_path.name)

        return {
            "document_id": self._document_id(resolved_path),
            "source": resolved_path.name,
            "filename": resolved_path.name,
            "filepath": str(resolved_path),
            "extension": resolved_path.suffix.lower(),
            "mime_type": mime_type or "application/octet-stream",
            "loader": loader_name,
            "char_count": len(content),
            "file_size": stat.st_size,
            "file_modified_at": datetime.fromtimestamp(
                stat.st_mtime,
                timezone.utc,
            ).isoformat(),
        }

    def _document_id(self, doc_path: Path) -> str:
        return hashlib.sha256(str(doc_path).encode("utf-8")).hexdigest()[:16]


def load_documents(docs_dir: str | Path) -> list[Document]:
    return DocumentLoader().load_documents(docs_dir)


# ===== 测试 =====
if __name__ == "__main__":
    # 可以直接运行这个文件来测试文档加载
    docs_dir = Path(__file__).resolve().parent / "docs"
    docs = load_documents(docs_dir)
    for doc in docs:
        print(doc)
