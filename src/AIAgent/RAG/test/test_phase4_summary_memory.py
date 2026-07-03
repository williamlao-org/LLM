import json
from types import SimpleNamespace

import pytest

from phase3_agentic_rag import AgenticRAG
from phase4_summary_memory import (
    LLMConversationSummarizer,
    SUMMARY_CONTEXT_PREFIX,
    SummaryBufferMemory,
)


class FakeSummarizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def summarize(self, existing_summary, turns, max_tokens):
        self.calls.append({
            "existing_summary": existing_summary,
            "turns": turns,
            "max_tokens": max_tokens,
        })
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def make_memory(summarizer, *, recent=8, summary=20):
    return SummaryBufferMemory(
        max_recent_tokens=recent,
        max_summary_tokens=summary,
        summarizer=summarizer,
        token_counter=len,
        tokens_per_message=0,
    )


def test_no_summary_before_recent_buffer_exceeds_budget():
    summarizer = FakeSummarizer([])
    memory = make_memory(summarizer)

    memory.add_turn("u1", "a1")
    memory.add_turn("u2", "a2")

    assert summarizer.calls == []
    assert memory.summary == ""
    assert [turn.user for turn in memory.turns] == ["u1", "u2"]
    assert memory.recent_tokens == 8


def test_overflow_summarizes_only_oldest_complete_turn():
    summarizer = FakeSummarizer(["summary one"])
    memory = make_memory(summarizer)
    memory.add_turn("u1", "a1")
    memory.add_turn("u2", "a2")

    memory.add_turn("u3", "a3")

    assert len(summarizer.calls) == 1
    evicted = summarizer.calls[0]["turns"]
    assert [(turn.user, turn.assistant) for turn in evicted] == [("u1", "a1")]
    assert [turn.user for turn in memory.turns] == ["u2", "u3"]
    assert memory.summary == "summary one"
    assert memory.recent_tokens == 8


def test_exact_turn_counter_controls_summary_trigger_without_fixed_overhead():
    summarizer = FakeSummarizer(["summary one"])
    counted_turns = []

    def exact_turn_counter(turn):
        counted_turns.append(turn)
        return 5

    memory = SummaryBufferMemory(
        max_recent_tokens=5,
        max_summary_tokens=20,
        summarizer=summarizer,
        token_counter=len,
        turn_token_counter=exact_turn_counter,
        tokens_per_message=999,
    )

    memory.add_turn("q1", "a1")
    memory.add_turn("q2", "a2")

    assert [turn.user for turn in counted_turns] == ["q1", "q2"]
    assert [turn.user for turn in memory.turns] == ["q2"]
    assert memory.recent_tokens == 5
    assert [turn.user for turn in summarizer.calls[0]["turns"]] == ["q1"]


def test_invalid_exact_turn_count_does_not_mutate_summary_memory():
    memory = SummaryBufferMemory(
        max_recent_tokens=10,
        max_summary_tokens=20,
        summarizer=FakeSummarizer([]),
        turn_token_counter=lambda turn: True,
    )

    with pytest.raises(TypeError, match="turn_token_counter"):
        memory.add_turn("question", "answer")

    assert memory.turns == ()
    assert memory.recent_tokens == 0


def test_repeated_compaction_merges_existing_summary():
    summarizer = FakeSummarizer(["S1", "S1 + S2"])
    memory = make_memory(summarizer)
    memory.add_turn("u1", "a1")
    memory.add_turn("u2", "a2")
    memory.add_turn("u3", "a3")

    memory.add_turn("u4", "a4")

    assert len(summarizer.calls) == 2
    assert summarizer.calls[1]["existing_summary"] == "S1"
    assert [turn.user for turn in summarizer.calls[1]["turns"]] == ["u2"]
    assert memory.summary == "S1 + S2"


def test_summary_is_injected_before_recent_raw_turns():
    summarizer = FakeSummarizer(["用户名叫小林"])
    memory = make_memory(summarizer, recent=4)
    memory.add_turn("我叫小林", "好的")
    memory.add_turn("u2", "a2")

    messages = memory.get_context_messages()

    assert messages[0] == {
        "role": "system",
        "content": SUMMARY_CONTEXT_PREFIX + "用户名叫小林",
    }
    assert messages[1:] == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_single_oversized_turn_is_summarized_without_partial_raw_messages():
    summarizer = FakeSummarizer(["compressed"])
    memory = make_memory(summarizer, recent=3)

    memory.add_turn("123", "456")

    assert memory.turns == ()
    assert memory.recent_tokens == 0
    assert [(turn.user, turn.assistant) for turn in summarizer.calls[0]["turns"]] == [
        ("123", "456")
    ]
    assert len(memory.get_context_messages()) == 1


def test_overlong_summary_is_hard_truncated_to_summary_budget():
    summarizer = FakeSummarizer(["abcdef"])
    memory = make_memory(summarizer, recent=1, summary=3)

    memory.add_turn("u", "a")

    assert memory.summary == "abc"
    assert memory.summary_tokens == 3


def test_summary_failure_keeps_budget_and_does_not_break_agent_answer():
    summarizer = FakeSummarizer([RuntimeError("summary unavailable")])
    memory = make_memory(summarizer, recent=1)
    completions = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="answer", tool_calls=[])
            )]
        )
    )
    agent = object.__new__(AgenticRAG)
    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent.llm_model = "test-model"
    agent.use_multi_hop = False
    agent.max_iterations = 5
    agent.max_tool_calls = 3
    agent.max_retrieval_retries = 2

    result = agent.query("q", verbose=False, memory=memory)

    assert result["answer"] == "answer"
    assert memory.turns == ()
    assert memory.recent_tokens == 0
    assert memory.summary == ""
    assert "summary unavailable" in memory.last_summary_error


def test_clear_resets_summary_recent_turns_counters_and_error():
    summarizer = FakeSummarizer(["summary", RuntimeError("failed")])
    memory = make_memory(summarizer, recent=1)
    memory.add_turn("u", "a")
    memory.add_turn("x", "y")

    assert memory.summary == "summary"
    assert memory.summary_tokens == len("summary")
    assert "failed" in memory.last_summary_error

    memory.clear()

    assert memory.summary == ""
    assert memory.summary_tokens == 0
    assert memory.recent_tokens == 0
    assert memory.turns == ()
    assert memory.last_summary_error is None


def test_llm_summarizer_uses_low_temperature_and_output_limit():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="  merged summary  ")
        )])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    summarizer = LLMConversationSummarizer(client, "test-model")

    result = summarizer.summarize(
        existing_summary="old",
        turns=(SimpleNamespace(user="u", assistant="a"),),
        max_tokens=42,
    )

    assert result == "merged summary"
    assert calls[0]["temperature"] == 0.1
    assert calls[0]["max_tokens"] == 42
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["existing_summary"] == "old"
    assert payload["new_turns"] == [{"user": "u", "assistant": "a"}]
