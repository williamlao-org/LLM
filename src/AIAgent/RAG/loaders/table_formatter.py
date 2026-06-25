"""表格文本格式化工具。"""

from collections.abc import Iterable
from typing import Any


def rows_to_markdown(rows: Iterable[Iterable[Any]]) -> str:
    normalized_rows = [
        [format_table_cell(cell) for cell in row]
        for row in rows
        if any("" if cell is None else str(cell).strip() for cell in row)
    ]
    if not normalized_rows:
        return ""

    column_count = max(len(row) for row in normalized_rows)
    padded_rows = [
        row + [""] * (column_count - len(row))
        for row in normalized_rows
    ]

    header = padded_rows[0]
    body = padded_rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(column_count)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def format_table_cell(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )
