"""DOCX 加载器。"""

from pathlib import Path

try:
    # pyrefly: ignore [missing-import]
    from loaders.table_formatter import rows_to_markdown
    from loaders.types import DocumentPart
except (ImportError, ModuleNotFoundError):
    from .table_formatter import rows_to_markdown
    from .types import DocumentPart


class DocxLoader:
    """加载 DOCX 文件并输出适合 RAG 使用的文本。"""

    def load(self, filepath: Path) -> list[DocumentPart]:
        try:
            from docx import Document as DocxDocument
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as e:
            raise ImportError("DOCX 加载需要安装 python-docx: uv add python-docx") from e

        try:
            doc = DocxDocument(filepath)
        except Exception as e:
            raise ValueError(f"无法打开 DOCX 文件 {filepath.name}: {e}") from e

        parts: list[DocumentPart] = []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                text = Paragraph(child, doc).text.strip()
                if text:
                    parts.append(
                        DocumentPart(
                            content=text,
                            metadata={"part_type": "paragraph"},
                        )
                    )
            elif isinstance(child, CT_Tbl):
                table_text = rows_to_markdown(
                    [self._row_to_text(row.cells) for row in Table(child, doc).rows]
                )
                if table_text:
                    parts.append(
                        DocumentPart(
                            content=table_text,
                            metadata={"part_type": "table"},
                        )
                    )

        if not parts:
            raise ValueError(f"DOCX 文件不包含可读取文本: {filepath.name}")
        return parts

    def _row_to_text(self, cells) -> list[str]:
        return [cell.text.strip() for cell in cells]
