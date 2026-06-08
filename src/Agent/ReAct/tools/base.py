from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., "ToolResult"]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolResult:
    ok: bool
    err: str = ""
    data: Any = None

    @classmethod
    def success(cls, data=None) -> "ToolResult":
        return cls(ok=True, err="", data=data)

    @classmethod
    def fail(cls, err: str, data=None) -> "ToolResult":
        return cls(ok=False, err=err, data=data)

    def to_dict(self):
        return {
            "ok": self.ok,
            "err": self.err,
            "data": self.data,
        }
