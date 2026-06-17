"""回合协议层:把模型一轮的原始输出解析 + 校验成结构化的 ParsedTurn。

这一层是"契约"的代码化身,而且是【唯一事实来源】。在此之前,契约的形状在两处
各写一份——prompt 里手写的 JSON 示例 + 这里手写的校验逻辑——改一处忘另一处就漂。
现在改成路线 a:用 pydantic 模型 AgentTurn 定义一次,两端都从它派生——
  - 给模型看的格式描述  = AgentTurn.model_json_schema()(见 prompt.py)
  - 回包校验            = AgentTurn.model_validate()(见 parse_turn)
一份定义喂两端,不可能漂移。

两条边界仍然成立:
  - 服务端的 json_object 保证"是合法 JSON",本层只管"形状/语义对不对"。
    顶层 JSON 非法 / 不符 schema / 二选一违规 → TurnAbort(整轮中止,喂回模型重答)。
  - 工具"存不存在"不归本层,归 executor 查 registry;本层只验形状。

设计变更(相对手写版):tool_calls 现在被 pydantic 严格逐条校验,单条坏掉
即整轮 TurnAbort(带 pydantic 的精确定位),不再造 error 占位单独 fail。
严格校验与逐条优雅降级本质冲突,这里选了前者——回包既已是合法 JSON,单条
schema 错属模型笔误,把精确报错喂回比静默占位更可操作。
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .tools.base import ToolCall


class TurnAbort(Exception):
    """本轮 LLM 输出无法解析或校验,这一轮没法继续。

    与"单个工具失败"区分开:工具失败是数据(ToolResult.fail),整轮照常;
    TurnAbort 是整轮中止,只能把错误喂回 LLM 让它重答。
    主循环专门捕获它,其它异常一律放行(那是真 bug,不该被静默吞掉)。
    """


class ToolCallSpec(BaseModel):
    """单个工具调用的形状契约:name + arguments。

    它和 prompt 里"调用长什么样"的描述同源——model_json_schema() 生成给模型看的
    部分,model_validate() 做回包校验。改这里两端一起变。
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentTurn(BaseModel):
    """一轮模型输出的信封契约——本协议的唯一事实来源。

    注意:"二选一"是【语义约束】,JSON Schema 表达不了,所以它既要在下面的
    model_validator 里用代码兜底,也要在 prompt 里用散文讲一遍(schema 生成不出
    这条规则)。这正是"形状能同源,语义约束难完全同源"的活例子。
    """

    reasoning: str = ""
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    final_answer: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "AgentTurn":
        has_tools = len(self.tool_calls) > 0
        has_final = self.final_answer is not None
        # 恰好一个:两者同真(都有)或同假(都无)都违规。
        if has_tools == has_final:
            raise ValueError(
                "每轮必须恰好二选一:非空 tool_calls 或 非空 final_answer"
            )
        return self


@dataclass
class ParsedTurn:
    """一轮模型输出解析校验后的结构化结果。

    kind 决定主循环走哪条路:final 直接收尾返回 final_answer;tool_calls 把
    tool_calls 交给执行器。parsed 保留原始 dict,供 session 记账留档。
    """

    kind: Literal["final", "tool_calls"]
    parsed: dict
    final_answer: Any = None
    tool_calls: list[ToolCall] = field(default_factory=list)


def parse_turn(raw: str) -> ParsedTurn:
    """解析 + 校验模型一轮原始输出。任何形状/语义级错误抛 TurnAbort。"""
    data = _loads(raw)
    try:
        turn = AgentTurn.model_validate(data)
    except ValidationError as e:
        # pydantic 报错带精确定位(哪个字段、缺什么、类型错在哪),原样喂回最有用
        raise TurnAbort(f"回合不符合 schema:{e}") from e

    # _exactly_one 已保证恰好一侧成立,这里据 final_answer 是否为 None 分流即可。
    if turn.final_answer is not None:
        return ParsedTurn(kind="final", parsed=data, final_answer=turn.final_answer)

    # id 由系统盖章(执行阶段靠 id 对账);spec 已是校验过的合法调用。
    tool_calls = [
        ToolCall(
            name=spec.name,
            arguments=spec.arguments,
            id=f"call_{uuid.uuid4().hex[:6]}",
        )
        for spec in turn.tool_calls
    ]
    return ParsedTurn(kind="tool_calls", parsed=data, tool_calls=tool_calls)


def _loads(raw: str) -> dict:
    """解析合法 JSON 并确认顶层是对象。

    json_object 模式下服务端已保证可解析,这里的 try 是客户端兜底
    (端点降级/空响应时仍接得住)。
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise TurnAbort(f"LLM 输出不是合法 JSON: {e}") from e
    if not isinstance(data, dict):
        raise TurnAbort(f"LLM 输出顶层必须是对象,得到 {type(data).__name__}")
    return data
