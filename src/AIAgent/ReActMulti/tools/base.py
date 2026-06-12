from uuid import uuid4
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
    id: str = ""
    # 解析阶段失败时,造一个带 error 的占位 ToolCall;正常调用此字段恒为空。
    # 执行阶段据此识别"这条 call 在解析时就废了",直接吐 fail,不查表。
    error: str = ""

    @classmethod
    def from_dict(cls, tool_call_dict, call_id: str | None) -> "ToolCall":
        if not isinstance(tool_call_dict, dict):
            raise ValueError(f"tool_call 不是对象: {tool_call_dict!r}")

        name = tool_call_dict.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool_call 缺少合法 name: {tool_call_dict!r}")

        args = tool_call_dict.get(
            "arguments", {}
        )  # 缺省给空 dict,不让它 KeyError, args允许 llm 返回空字典
        if not isinstance(args, dict):
            raise ValueError(f"arguments 必须是对象: {args!r}")

        if call_id is None:
            call_id = f"call_{uuid4().hex[:6]}"

        return cls(name=name, arguments=args, id=call_id)


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
