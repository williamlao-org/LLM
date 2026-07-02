"""
Phase 4.2: 按 Token 预算裁剪的短期记忆。

``max_tokens`` 只表示历史问答可使用的预算，不是模型完整的
上下文上限。system prompt、tools、当前问题和预期回答都应在
分配这个预算前预留空间。
"""

from collections import deque
from math import ceil
from typing import Callable

from phase4_working_memory import ConversationTurn


TokenCounter = Callable[[str], int]


def estimate_text_tokens(text: str) -> int:
    """用可解释的本地规则粗估 Token 数。

    - 非 ASCII 字符：约 1 字符 / token
    - ASCII 字符：约 4 字符 / token

    它用于学习和离线测试，不保证与服务端计数完全一致。
    """
    ascii_count = sum(1 for char in text if ord(char) < 128)
    non_ascii_count = len(text) - ascii_count
    return non_ascii_count + ceil(ascii_count / 4)


class TokenBudgetMemory:
    """保留不超过 Token 预算的连续、最新完整问答。"""

    def __init__(
        self,
        max_tokens: int,
        token_counter: TokenCounter | None = None,
        tokens_per_message: int = 4,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if tokens_per_message < 0:
            raise ValueError("tokens_per_message 不能小于 0")

        self.max_tokens = max_tokens
        self.tokens_per_message = tokens_per_message
        self.token_counter = token_counter or estimate_text_tokens
        self.current_tokens = 0
        self._turns: deque[tuple[ConversationTurn, int]] = deque()

    def _count_text(self, text: str) -> int:
        count = self.token_counter(text)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("token_counter 必须返回 int")
        if count < 0:
            raise ValueError("token_counter 不能返回负数")
        return count

    def _count_turn(self, turn: ConversationTurn) -> int:
        content_tokens = (
            self._count_text(turn.user) + self._count_text(turn.assistant)
        )
        return content_tokens + 2 * self.tokens_per_message

    def add_turn(self, user: str, assistant: str) -> None:
        """写入完整问答，然后永久淘汰最早轮次直到回到预算内。"""
        turn = ConversationTurn(user=str(user), assistant=str(assistant))
        turn_tokens = self._count_turn(turn)

        self._turns.append((turn, turn_tokens))
        self.current_tokens += turn_tokens

        while self.current_tokens > self.max_tokens and self._turns:
            _, evicted_tokens = self._turns.popleft()
            self.current_tokens -= evicted_tokens

    def get_context_messages(self) -> list[dict[str, str]]:
        """将当前 Token 窗口展开为 OpenAI messages 格式。"""
        messages: list[dict[str, str]] = []
        for turn, _ in self._turns:
            messages.extend([
                {"role": "user", "content": turn.user},
                {"role": "assistant", "content": turn.assistant},
            ])
        return messages

    def clear(self) -> None:
        """清空问答和 Token 计数。"""
        self._turns.clear()
        self.current_tokens = 0

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        """返回只读问答快照。"""
        return tuple(turn for turn, _ in self._turns)
