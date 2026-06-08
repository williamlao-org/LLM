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
class ToolCall:
    name: str
    arguments: dict

    @classmethod
    def from_dict(cls, tool_call_dict) -> "ToolCall":
        if not isinstance(tool_call_dict, dict):
            raise ValueError(f"tool_call 不是对象: {tool_call_dict!r}")
        name = tool_call_dict.get("name")

        if not isinstance(name, str) or not name:
            raise ValueError(f"tool_call 缺少合法 name: {tool_call_dict!r}")
        args = tool_call_dict.get("arguments", {})  # 缺省给空 dict,不让它 KeyError
        
        if not isinstance(args, dict):
            raise ValueError(f"arguments 必须是对象: {args!r}")
        return cls(name=name, arguments=args)


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
