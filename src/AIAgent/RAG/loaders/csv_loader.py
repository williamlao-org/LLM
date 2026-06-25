"""CSV 加载器。"""

import csv
import io
from pathlib import Path

try:
    # pyrefly: ignore [missing-import]
    from loaders.table_formatter import rows_to_markdown
    from loaders.types import DocumentPart
except (ImportError, ModuleNotFoundError):
    from .table_formatter import rows_to_markdown
    from .types import DocumentPart


class CsvLoader:
    """加载 CSV 文件并输出 Markdown-ish 表格文本。"""

    def load(self, filepath: Path) -> list[DocumentPart]:
        try:
            text = filepath.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError(f"无法按 UTF-8 读取 CSV 文件 {filepath.name}: {e}") from e

        try:
            dialect = csv.Sniffer().sniff(text[:4096])
        except csv.Error:
            dialect = csv.excel

        try:
            rows = list(csv.reader(io.StringIO(text), dialect))
        except csv.Error as e:
            raise ValueError(f"无法解析 CSV 文件 {filepath.name}: {e}") from e

        content = rows_to_markdown(rows)
        if not content.strip():
            return []
        return [DocumentPart(content=content, metadata={"part_type": "table"})]
