"""
Phase 4.1: 滑动窗口短期记忆。

这一步故意只按「完整对话轮」限制窗口，不做 Token 精确计数、
历史摘要或持久化。目的是先把最基础的遗忘机制看清楚。
"""

from collections import deque
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ConversationTurn:
    """一个完整的用户/助手问答轮次。"""

    user: str
    assistant: str


class WorkingMemory(Protocol):
    """Agent 所依赖的最小短期记忆协议。"""

    def add_turn(self, user: str, assistant: str) -> None:
        """写入一个完整问答。"""
        ...

    def get_context_messages(self) -> list[dict[str, str]]:
        """返回应注入模型上下文的历史消息。"""
        ...

    def clear(self) -> None:
        """清空当前会话。"""
        ...

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """返回当前保留的完整问答。"""
        ...


class ConversationWindowMemory:
    """只保留最近 ``max_turns`` 个完整问答的短期记忆。"""

    def __init__(self, max_turns: int = 3):
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")

        self.max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add_turn(self, user: str, assistant: str) -> None:
        """写入一个完整问答；超出窗口时自动淘汰最早的一轮。"""
        self._turns.append(
            ConversationTurn(user=str(user), assistant=str(assistant))
        )

    def get_context_messages(self) -> list[dict[str, str]]:
        """将当前窗口展开为 OpenAI messages 格式的新列表。"""
        messages: list[dict[str, str]] = []
        for turn in self._turns:
            messages.extend([
                {"role": "user", "content": turn.user},
                {"role": "assistant", "content": turn.assistant},
            ])
        return messages

    def clear(self) -> None:
        """清空当前会话。"""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """返回只读快照，便于 CLI 观察当前窗口。"""
        return tuple(self._turns)
