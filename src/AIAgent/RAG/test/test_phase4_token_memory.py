import pytest

from phase4_token_memory import TokenBudgetMemory, estimate_text_tokens


def test_default_estimator_handles_ascii_and_non_ascii():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2
    assert estimate_text_tokens("你好") == 2
    assert estimate_text_tokens("你好abcd") == 3


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
