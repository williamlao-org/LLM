import json
from types import SimpleNamespace

import pytest

from phase3_agentic_rag import AgenticRAG
from phase4_token_memory import TokenBudgetMemory
from phase4_working_memory import ConversationWindowMemory


def llm_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("LLM failed")
        return self.responses.pop(0)


def make_agent(responses, *, max_iterations=5):
    agent = object.__new__(AgenticRAG)
    completions = FakeCompletions(responses)
    agent.llm_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    agent.llm_model = "test-model"
    agent.use_multi_hop = False
    agent.max_iterations = max_iterations
    agent.max_tool_calls = 3
    agent.max_retrieval_retries = 2
    return agent, completions


def test_window_evicts_oldest_complete_turn():
    memory = ConversationWindowMemory(max_turns=3)

    for index in range(1, 5):
        memory.add_turn(f"q{index}", f"a{index}")

    assert [(turn.user, turn.assistant) for turn in memory.turns] == [
        ("q2", "a2"),
        ("q3", "a3"),
        ("q4", "a4"),
    ]
    assert memory.get_context_messages() == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
    ]


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError, match="max_turns"):
        ConversationWindowMemory(max_turns=0)


def test_clear_removes_all_turns():
    memory = ConversationWindowMemory()
    memory.add_turn("question", "answer")

    memory.clear()

    assert len(memory) == 0
    assert memory.get_context_messages() == []


def test_query_injects_history_then_commits_final_answer():
    memory = ConversationWindowMemory(max_turns=3)
    memory.add_turn("我叫小林", "好的，小林。")
    agent, completions = make_agent([llm_response(content="你叫小林。")])

    result = agent.query("我叫什么？", verbose=False, memory=memory)

    assert result["answer"] == "你叫小林。"
    assert completions.calls[0]["messages"][1:] == [
        {"role": "user", "content": "我叫小林"},
        {"role": "assistant", "content": "好的，小林。"},
        {"role": "user", "content": "我叫什么？"},
    ]
    assert memory.turns[-1].user == "我叫什么？"
    assert memory.turns[-1].assistant == "你叫小林。"


def test_query_without_memory_remains_stateless():
    agent, completions = make_agent([
        llm_response(content="first"),
        llm_response(content="second"),
    ])

    agent.query("q1", verbose=False)
    agent.query("q2", verbose=False)

    assert completions.calls[0]["messages"][1:] == [
        {"role": "user", "content": "q1"}
    ]
    assert completions.calls[1]["messages"][1:] == [
        {"role": "user", "content": "q2"}
    ]


def test_query_accepts_token_budget_memory():
    memory = TokenBudgetMemory(
        max_tokens=50,
        token_counter=len,
        tokens_per_message=0,
    )
    memory.add_turn("previous question", "previous answer")
    agent, completions = make_agent([llm_response(content="current answer")])

    agent.query("current question", verbose=False, memory=memory)

    assert completions.calls[0]["messages"][1:] == [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "current question"},
    ]
    assert memory.turns[-1].assistant == "current answer"


def test_direct_answer_is_committed_without_internal_tool_messages():
    memory = ConversationWindowMemory()
    agent, _ = make_agent([
        llm_response(tool_calls=[
            tool_call(
                "direct_answer",
                json.dumps({"answer": "2"}),
            )
        ])
    ])

    result = agent.query("1+1=?", verbose=False, memory=memory)

    assert result["answer"] == "2"
    assert memory.get_context_messages() == [
        {"role": "user", "content": "1+1=?"},
        {"role": "assistant", "content": "2"},
    ]


def test_failed_query_does_not_commit_partial_turn():
    memory = ConversationWindowMemory()
    agent, _ = make_agent([])

    with pytest.raises(RuntimeError, match="LLM failed"):
        agent.query("will fail", verbose=False, memory=memory)

    assert len(memory) == 0


def test_forced_final_answer_is_committed():
    memory = ConversationWindowMemory()
    agent, _ = make_agent(
        [llm_response(content="forced answer")],
        max_iterations=0,
    )

    result = agent.query("question", verbose=False, memory=memory)

    assert result["answer"] == "forced answer"
    assert memory.turns[-1].assistant == "forced answer"
