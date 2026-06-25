"""Loader 层共享类型。"""

from dataclasses import dataclass, field

MetadataValue = str | int | float | bool


@dataclass
class DocumentPart:
    """文件内部的一个可索引片段。"""

    content: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)
