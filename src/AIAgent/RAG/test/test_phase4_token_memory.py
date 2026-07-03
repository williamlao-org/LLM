from types import SimpleNamespace

import pytest

import phase4_token_memory
from phase4_token_memory import (
    DEEPSEEK_V4_ASSISTANT_TOKEN,
    DEEPSEEK_V4_EOS_TOKEN,
    DEEPSEEK_V4_THINKING_END_TOKEN,
    DEEPSEEK_V4_USER_TOKEN,
    DeepSeekV4TokenCounter,
    TokenBudgetMemory,
    estimate_text_tokens,
)
from phase4_working_memory import ConversationTurn


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return SimpleNamespace(ids=list(range(len(text))))


def test_default_estimator_handles_ascii_and_non_ascii():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2
    assert estimate_text_tokens("你好") == 2
    assert estimate_text_tokens("你好abcd") == 3


def test_deepseek_counter_encodes_complete_chat_turn_once():
    tokenizer = FakeTokenizer()
    counter = DeepSeekV4TokenCounter(tokenizer)
    turn = ConversationTurn(user="question", assistant="answer")

    result = counter.count_turn(turn)

    expected = (
        f"{DEEPSEEK_V4_USER_TOKEN}question"
        f"{DEEPSEEK_V4_ASSISTANT_TOKEN}{DEEPSEEK_V4_THINKING_END_TOKEN}"
        f"answer{DEEPSEEK_V4_EOS_TOKEN}"
    )
    assert tokenizer.calls == [(expected, False)]
    assert result == len(expected)


def test_deepseek_counter_counts_summary_as_plain_text():
    tokenizer = FakeTokenizer()
    counter = DeepSeekV4TokenCounter(tokenizer)

    result = counter.count_text("历史摘要")

    assert tokenizer.calls == [("历史摘要", False)]
    assert result == len("历史摘要")


def test_deepseek_counter_load_failure_is_explicit(monkeypatch):
    def fail_to_load(model):
        raise OSError("offline")

    monkeypatch.setattr(
        phase4_token_memory,
        "Tokenizer",
        SimpleNamespace(from_pretrained=fail_to_load),
    )

    with pytest.raises(RuntimeError, match="DeepSeek tokenizer"):
        DeepSeekV4TokenCounter.from_pretrained("deepseek/test-tokenizer")


def test_exact_budget_keeps_turn_and_excess_evicts_oldest_complete_turn():
    memory = TokenBudgetMemory(
        max_tokens=8,
        token_counter=len,
        tokens_per_message=1,
    )

    memory.add_turn("a", "b")  # 1 + 1 + 2 message overhead = 4
    memory.add_turn("c", "d")

    assert memory.current_tokens == 8
    assert [(turn.user, turn.assistant) for turn in memory.turns] == [
        ("a", "b"),
        ("c", "d"),
    ]

    memory.add_turn("e", "f")

    assert memory.current_tokens == 8
    assert [(turn.user, turn.assistant) for turn in memory.turns] == [
        ("c", "d"),
        ("e", "f"),
    ]


def test_content_length_changes_eviction_even_with_same_turn_count():
    memory = TokenBudgetMemory(
        max_tokens=14,
        token_counter=len,
        tokens_per_message=0,
    )

    memory.add_turn("a", "b")
    memory.add_turn("long", "12345678")
    memory.add_turn("c", "d")

    assert len(memory) == 2
    assert [turn.user for turn in memory.turns] == ["long", "c"]
    assert memory.current_tokens == 14


def test_oversized_newest_turn_is_not_partially_retained():
    memory = TokenBudgetMemory(
        max_tokens=5,
        token_counter=len,
        tokens_per_message=0,
    )

    memory.add_turn("123", "456")

    assert memory.turns == ()
    assert memory.get_context_messages() == []
    assert memory.current_tokens == 0


def test_injected_counter_is_used_for_both_messages():
    calls = []

    def fake_counter(text: str) -> int:
        calls.append(text)
        return 2

    memory = TokenBudgetMemory(
        max_tokens=10,
        token_counter=fake_counter,
        tokens_per_message=1,
    )

    memory.add_turn("question", "answer")

    assert calls == ["question", "answer"]
    assert memory.current_tokens == 6


def test_exact_turn_counter_replaces_text_and_fixed_message_overhead():
    calls = []

    def exact_turn_counter(turn):
        calls.append(turn)
        return 5

    memory = TokenBudgetMemory(
        max_tokens=5,
        token_counter=lambda text: pytest.fail("不应计算单条正文"),
        turn_token_counter=exact_turn_counter,
        tokens_per_message=999,
    )

    memory.add_turn("q1", "a1")
    memory.add_turn("q2", "a2")

    assert [(turn.user, turn.assistant) for turn in calls] == [
        ("q1", "a1"),
        ("q2", "a2"),
    ]
    assert memory.current_tokens == 5
    assert [turn.user for turn in memory.turns] == ["q2"]


@pytest.mark.parametrize(
    ("invalid_count", "error_type", "message"),
    [
        (True, TypeError, "turn_token_counter"),
        (-1, ValueError, "负数"),
    ],
)
def test_invalid_turn_counter_result_does_not_mutate_memory(
    invalid_count,
    error_type,
    message,
):
    memory = TokenBudgetMemory(
        max_tokens=10,
        turn_token_counter=lambda turn: invalid_count,
    )

    with pytest.raises(error_type, match=message):
        memory.add_turn("question", "answer")

    assert memory.turns == ()
    assert memory.current_tokens == 0


def test_invalid_counter_result_does_not_mutate_memory():
    memory = TokenBudgetMemory(
        max_tokens=10,
        token_counter=lambda text: -1,
    )

    with pytest.raises(ValueError, match="负数"):
        memory.add_turn("question", "answer")

    assert memory.turns == ()
    assert memory.current_tokens == 0


def test_clear_removes_turns_and_resets_token_count():
    memory = TokenBudgetMemory(
        max_tokens=20,
        token_counter=len,
        tokens_per_message=0,
    )
    memory.add_turn("question", "answer")

    memory.clear()

    assert memory.turns == ()
    assert memory.current_tokens == 0
