"""JSON 加载器。"""

import json
from pathlib import Path

try:
    # pyrefly: ignore [missing-import]
    from loaders.types import DocumentPart
except (ImportError, ModuleNotFoundError):
    from .types import DocumentPart


class JsonLoader:
    """加载 JSON 文件并输出格式化文本。"""

    def load(self, filepath: Path) -> list[DocumentPart]:
        try:
            data = json.loads(filepath.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as e:
            raise ValueError(f"无法按 UTF-8 读取 JSON 文件 {filepath.name}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON 文件 {filepath.name}: {e}") from e

        content = json.dumps(data, ensure_ascii=False, indent=2)
        return [DocumentPart(content=content, metadata={"part_type": "json"})]
