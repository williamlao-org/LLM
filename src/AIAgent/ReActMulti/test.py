"""可靠性路径的回归测试。

超时/失败路径平时跑不到——不写测试它就等于不存在,第一次线上工具挂死时才爆。
LLMClient 用假参数构造(不发起任何网络请求),整套测试离线可跑。

运行(项目根目录下):
    python -m src.AIAgent.ReActMulti.test
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

from .agent import Agent
from .events import ContentDelta, ContentDone, UsageEvent
from .llm import LLMClient
from .renderer import SilentRenderer
from .session import SessionState, UsageRecord
from .tools.base import Tool, ToolCall, ToolResult


class RecordingRenderer(SilentRenderer):
    def __init__(self):
        self.context_compacts = []

    def on_context_compact(
        self,
        folded_count: int,
        prompt_tokens: int | None,
        context_limit: int | None,
        context_watermark: float,
    ) -> None:
        self.context_compacts.append(
            {
                "folded_count": folded_count,
                "prompt_tokens": prompt_tokens,
                "context_limit": context_limit,
                "context_watermark": context_watermark,
            }
        )


def _make_session(user_goal: str = "") -> SessionState:
    return SessionState.create(
        user_goal=user_goal,
        workspace_dir=Path(__file__).resolve().parent / "workspace",
    )


def _make_agent(
    tools: list[Tool], tool_timeout: float, keep_recent_tool_results: int = 3
) -> Agent:
    llm = LLMClient(
        base_url="http://x", api_key="sk-x", model="m", context_limit=128_000
    )
    return Agent(
        llm,
        tools,
        _make_session(),
        SilentRenderer(),
        tool_timeout=tool_timeout,
        keep_recent_tool_results=keep_recent_tool_results,
    )


def test_agent_does_not_duplicate_system_prompt_for_existing_session():
    llm = LLMClient(
        base_url="http://x", api_key="sk-x", model="m", context_limit=128_000
    )
    session = _make_session()

    first_agent = Agent(llm, [], session, SilentRenderer(), tool_timeout=5)
    second_agent = Agent(llm, [], session, SilentRenderer(), tool_timeout=5)

    system_messages = [msg for msg in session.messages if msg.get("role") == "system"]
    assert first_agent.messages is second_agent.messages
    assert len(system_messages) == 1


def test_parallel_timeout_fills_fail():
    """超时的调用必须以 fail 占位留在结果里,不能蒸发,也不能炸穿。"""

    def fast():
        return ToolResult.success("fast done")

    def slow():
        time.sleep(3)
        return ToolResult.success("slow done")

    agent = _make_agent(
        [Tool("fast", "", {}, fast), Tool("slow", "", {}, slow)],
        tool_timeout=0.5,
    )
    calls = [ToolCall("fast", {}, "c1"), ToolCall("slow", {}, "c2")]

    t0 = time.time()
    results = agent.execute_tool_calls_parallel(calls)
    elapsed = time.time() - t0

    assert len(results) == len(calls), "模型靠 id 对账,结果少一条都不行"
    assert results[0][1].ok and results[0][1].data == "fast done"
    assert results[0][2] == "succeeded"
    assert not results[1][1].ok and "timeout" in results[1][1].err
    assert results[1][2] == "timeout"
    assert elapsed < 2, f"应在预算 0.5s 附近返回,实际等了 {elapsed:.1f}s"


def test_inner_timeout_clamped_to_budget():
    """模型传的内层 timeout 必须被钳到外层预算内,否则外层先掐。"""
    captured = {}

    def spy(timeout: int = 20):
        captured["timeout"] = timeout
        return ToolResult.success(None)

    agent = _make_agent([Tool("spy", "", {}, spy)], tool_timeout=30)
    agent.execute_tool_calls([ToolCall("spy", {"timeout": 300}, "c1")])

    assert captured["timeout"] == 30


def test_tool_exception_becomes_fail():
    """工具抛异常是数据(fail),不是事故,整轮照常。"""

    def boom():
        raise RuntimeError("炸了")

    agent = _make_agent([Tool("boom", "", {}, boom)], tool_timeout=5)
    results = agent.execute_tool_calls([ToolCall("boom", {}, "c1")])

    assert not results[0][1].ok
    assert "RuntimeError" in results[0][1].err


def test_run_turn_records_usage():
    """UsageEvent 要被 Agent 接住:上下文取最近值,输出 token 做累计。"""

    class FakeLLM:
        def __init__(self):
            self.calls = 0
            self.context_limit = 128_000

        def __call__(self, messages):
            self.calls += 1
            yield ContentDone(content=f"content {self.calls}")
            yield UsageEvent(
                SimpleNamespace(
                    prompt_tokens=100 * self.calls,
                    completion_tokens=10 * self.calls,
                    total_tokens=110 * self.calls,
                )
            )

    agent = Agent(FakeLLM(), [], _make_session(), SilentRenderer(), tool_timeout=5)

    content, usage = agent._run_turn()
    assert content == "content 1"
    assert usage == UsageRecord(prompt_tokens=100, completion_tokens=10, total_tokens=110)

    content, usage = agent._run_turn()
    assert content == "content 2"
    assert usage == UsageRecord(prompt_tokens=200, completion_tokens=20, total_tokens=220)


def test_run_turn_records_dict_usage():
    """兼容 dict 形态的 usage,不少 OpenAI-compatible 接口会这样返回。"""

    class FakeLLM:
        context_limit = 128_000

        def __call__(self, messages):
            yield ContentDone(content="ok")
            yield UsageEvent(
                {
                    "completion_tokens": 1541,
                    "prompt_tokens": 11,
                    "total_tokens": 1552,
                    "completion_tokens_details": {
                        "reasoning_tokens": 1426,
                    },
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                    },
                    "prompt_cache_hit_tokens": 0,
                    "prompt_cache_miss_tokens": 11,
                }
            )

    agent = Agent(FakeLLM(), [], _make_session(), SilentRenderer(), tool_timeout=5)

    content, usage = agent._run_turn()
    assert content == "ok"
    assert usage == UsageRecord(
        prompt_tokens=11, completion_tokens=1541, total_tokens=1552
    )


def test_stream_usage_event_can_live_on_choice_chunk():
    """兼容接口可能把 usage 挂在带 choices 的流式 chunk 上,不能漏读。"""

    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))],
        usage=usage,
    )

    class FakeCompletions:
        def create(self, **kwargs):
            return iter([chunk])

    llm = LLMClient.__new__(LLMClient)
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    llm.model = "fake-model"
    llm.max_attempts = 1
    llm.base_wait = 0
    llm.max_wait = 0

    events = list(llm._call_stream([{"role": "user", "content": "hello"}]))

    assert isinstance(events[0], ContentDelta) and events[0].piece == "hi"
    assert isinstance(events[1], UsageEvent)
    assert isinstance(events[-1], ContentDone)


def test_session_records_tool_turn_and_execution():
    session = _make_session("inspect file")
    call = ToolCall("read_file", {"file": "a.py"}, "call_1")

    turn = session.record_assistant_turn(
        assistant_raw='{"tool_calls":[{"name":"read_file"}],"final_answer":null}',
        parsed={"tool_calls": [{"name": "read_file"}], "final_answer": None},
        route="tool_calls",
        tool_calls=[call],
    )

    assert session.step_count == 1
    assert turn.step == 1
    assert turn.tool_execution_ids == ["call_1"]
    assert session.tool_executions["call_1"].call is call
    assert session.tool_executions["call_1"].result is None
    assert session.tool_executions["call_1"].status == "pending"

    result = ToolResult.success({"content": "hello"})
    execution = session.record_tool_execution("call_1", result)

    assert execution.result is result
    assert execution.status == "succeeded"


def test_session_rejects_duplicate_tool_ids_without_partial_state():
    session = _make_session("duplicate calls")
    calls = [
        ToolCall("read_file", {"file": "a.py"}, "call_dup"),
        ToolCall("read_file", {"file": "b.py"}, "call_dup"),
    ]

    try:
        session.record_assistant_turn(
            assistant_raw='{"tool_calls":[{},{}],"final_answer":null}',
            parsed={"tool_calls": [{}, {}], "final_answer": None},
            route="tool_calls",
            tool_calls=calls,
        )
    except ValueError as exc:
        assert "重复的 tool_call id" in str(exc)
    else:
        raise AssertionError("duplicate tool_call ids should be rejected")

    assert session.step_count == 0
    assert session.turns == []
    assert session.tool_executions == {}


def test_session_records_invalid_usage_and_status():
    session = _make_session("bad output")

    turn = session.record_invalid_turn("not json", "LLM 输出不是合法 JSON")
    assert session.step_count == 1
    assert turn.route == "invalid"
    assert turn.error == "LLM 输出不是合法 JSON"
    assert turn.tool_execution_ids == []

    session.record_usage_for_turn(
        turn,
        UsageRecord(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )
    assert turn.usage == UsageRecord(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    assert session.last_usage == turn.usage
    assert session.total_usage == UsageRecord(
        prompt_tokens=10, completion_tokens=2, total_tokens=12
    )

    final_turn = session.record_assistant_turn(
        assistant_raw='{"tool_calls":[],"final_answer":"done"}',
        parsed={"tool_calls": [], "final_answer": "done"},
        route="final",
    )
    assert final_turn.step == 2
    assert final_turn.tool_execution_ids == []

    session.mark_completed()
    assert session.status == "completed"
    session.mark_max_steps()
    assert session.status == "max_steps"


def test_run_defaults_to_session_max_steps():
    class InvalidLLM:
        context_limit = 128_000

        def __call__(self, messages):
            yield ContentDone(content="not json")

    session = _make_session()
    session.max_steps = 2
    agent = Agent(InvalidLLM(), [], session, SilentRenderer(), tool_timeout=5)

    assert agent.run("keep failing") is None
    assert session.status == "max_steps"
    assert session.step_count == 2
    assert session.max_steps == 2


def test_is_tool_result_message_only_accepts_valid_tool_results():
    agent = _make_agent([], tool_timeout=5)

    assert agent._is_tool_result_message(
        {"role": "user", "content": json.dumps({"tool_results": []})}
    )
    assert not agent._is_tool_result_message(
        {"role": "assistant", "content": json.dumps({"tool_results": []})}
    )
    assert not agent._is_tool_result_message({"role": "user", "content": "not json"})
    assert not agent._is_tool_result_message(
        {"role": "user", "content": json.dumps("tool_results")}
    )
    assert not agent._is_tool_result_message(
        {"role": "user", "content": json.dumps({"tool_results": {}})}
    )
    assert not agent._is_tool_result_message({"role": "user", "content": None})


def test_fold_old_tool_results_keeps_recent_and_roles():
    agent = _make_agent([], tool_timeout=5, keep_recent_tool_results=2)

    original_messages = list(agent.messages)
    for i in range(5):
        agent.messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tool_results": [
                            {
                                "id": f"call_{i}",
                                "name": "read_file",
                                "result": {
                                    "ok": True,
                                    "err": "",
                                    "data": f"large result {i}",
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                "_test_extra_field": f"keep {i}",
            }
        )
        agent.messages.append({"role": "assistant", "content": f"assistant {i}"})

    before_len = len(agent.messages)
    before_roles = [msg["role"] for msg in agent.messages]

    assert agent._fold_old_tool_results() == 3
    assert len(agent.messages) == before_len
    assert [msg["role"] for msg in agent.messages] == before_roles
    assert agent.messages[: len(original_messages)] == original_messages

    tool_result_messages = [
        json.loads(msg["content"])
        for msg in agent.messages
        if agent._is_tool_result_message(msg)
    ]

    assert len(tool_result_messages) == 5

    for folded in tool_result_messages[:3]:
        assert folded["folded"] is True
        result = folded["tool_results"][0]["result"]
        assert result["ok"] is True
        assert result["err"] == ""
        assert result["data"] == "[旧工具结果已折叠以节省上下文]"

    folded_messages = [
        msg
        for msg in agent.messages
        if agent._is_tool_result_message(msg) and json.loads(msg["content"]).get("folded")
    ]
    assert [msg["_test_extra_field"] for msg in folded_messages] == [
        "keep 0",
        "keep 1",
        "keep 2",
    ]

    for recent_idx, recent in enumerate(tool_result_messages[3:], start=3):
        assert "folded" not in recent
        assert recent["tool_results"][0]["result"]["data"] == f"large result {recent_idx}"


def test_fold_old_tool_results_is_idempotent():
    agent = _make_agent([], tool_timeout=5, keep_recent_tool_results=1)
    for i in range(3):
        agent.messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tool_results": [
                            {
                                "id": f"call_{i}",
                                "name": "read_file",
                                "result": {"ok": False, "err": f"err {i}", "data": "x"},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        )

    assert agent._fold_old_tool_results() == 2
    after_first_fold = list(agent.messages)
    assert agent._fold_old_tool_results() == 0
    assert agent.messages == after_first_fold


def _append_tool_result_message(agent: Agent, idx: int) -> None:
    agent.messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "tool_results": [
                        {
                            "id": f"call_{idx}",
                            "name": "read_file",
                            "result": {
                                "ok": True,
                                "err": "",
                                "data": f"large result {idx}",
                            },
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        }
    )


def test_compact_context_if_needed_folds_and_reports():
    renderer = RecordingRenderer()
    llm = LLMClient(
        base_url="http://x", api_key="sk-x", model="m", context_limit=100
    )
    agent = Agent(
        llm,
        [],
        _make_session(),
        renderer,
        tool_timeout=5,
        context_watermark=0.75,
        keep_recent_tool_results=2,
    )
    agent.session_state.last_usage = UsageRecord(
        prompt_tokens=80, completion_tokens=10, total_tokens=90
    )

    for i in range(5):
        _append_tool_result_message(agent, i)

    assert agent._compact_context_if_needed() == 3
    assert agent.session_state.last_usage is None
    assert renderer.context_compacts == [
        {
            "folded_count": 3,
            "prompt_tokens": 80,
            "context_limit": 100,
            "context_watermark": 0.75,
        }
    ]


def test_compact_context_if_needed_reports_when_nothing_to_fold():
    renderer = RecordingRenderer()
    llm = LLMClient(
        base_url="http://x", api_key="sk-x", model="m", context_limit=100
    )
    agent = Agent(
        llm,
        [],
        _make_session(),
        renderer,
        tool_timeout=5,
        context_watermark=0.75,
        keep_recent_tool_results=3,
    )
    agent.session_state.last_usage = UsageRecord(
        prompt_tokens=80, completion_tokens=10, total_tokens=90
    )

    assert agent._compact_context_if_needed() == 0
    assert agent.session_state.last_usage == UsageRecord(
        prompt_tokens=80, completion_tokens=10, total_tokens=90
    )
    assert renderer.context_compacts == [
        {
            "folded_count": 0,
            "prompt_tokens": 80,
            "context_limit": 100,
            "context_watermark": 0.75,
        }
    ]


def test_compact_context_if_needed_skips_below_watermark():
    renderer = RecordingRenderer()
    llm = LLMClient(
        base_url="http://x", api_key="sk-x", model="m", context_limit=100
    )
    agent = Agent(
        llm,
        [],
        _make_session(),
        renderer,
        tool_timeout=5,
        context_watermark=0.75,
        keep_recent_tool_results=1,
    )
    agent.session_state.last_usage = UsageRecord(prompt_tokens=75, completion_tokens=0, total_tokens=75)

    assert agent._compact_context_if_needed() == 0
    assert renderer.context_compacts == []


if __name__ == "__main__":
    test_agent_does_not_duplicate_system_prompt_for_existing_session()
    test_parallel_timeout_fills_fail()
    test_inner_timeout_clamped_to_budget()
    test_tool_exception_becomes_fail()
    test_run_turn_records_usage()
    test_run_turn_records_dict_usage()
    test_stream_usage_event_can_live_on_choice_chunk()
    test_session_records_tool_turn_and_execution()
    test_session_rejects_duplicate_tool_ids_without_partial_state()
    test_session_records_invalid_usage_and_status()
    test_run_defaults_to_session_max_steps()
    test_is_tool_result_message_only_accepts_valid_tool_results()
    test_fold_old_tool_results_keeps_recent_and_roles()
    test_fold_old_tool_results_is_idempotent()
    test_compact_context_if_needed_folds_and_reports()
    test_compact_context_if_needed_reports_when_nothing_to_fold()
    test_compact_context_if_needed_skips_below_watermark()
    print("all tests passed")
