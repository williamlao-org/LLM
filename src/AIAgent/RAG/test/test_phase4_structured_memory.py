import json
from types import SimpleNamespace

import pytest

from phase4_structured_memory import (
    LLMWorkingStateExtractor,
    MemoryExtraction,
    MemoryOperation,
    TokenAndBreakUpdatePolicy,
    StructuredWorkingMemory,
)
from phase4_summary_memory import SummaryBufferMemory
from phase4_working_memory import ConversationTurn, ConversationWindowMemory


class FakeExtractor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def extract(self, turns, existing_entries):
        self.calls.append((turns, existing_entries))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def extraction(*operations):
    return MemoryExtraction(operations=list(operations))


def upsert(category, key, value):
    return MemoryOperation(
        action="upsert",
        category=category,
        key=key,
        value=value,
    )


def delete(category, key):
    return MemoryOperation(action="delete", category=category, key=key)


def make_memory(outputs, *, min_tokens=1, min_tool_calls=1, max_entries=30):
    extractor = FakeExtractor(outputs)
    memory = StructuredWorkingMemory(
        base_memory=ConversationWindowMemory(max_turns=20),
        extractor=extractor,
        update_policy=TokenAndBreakUpdatePolicy(
            min_tokens_between_updates=min_tokens,
            min_tool_calls_between_updates=min_tool_calls,
        ),
        max_entries=max_entries,
    )
    return memory, extractor


def test_trigger_on_natural_break_once_token_threshold_is_met():
    # 设定 min_tokens=45, min_tool_calls=3
    # 每轮问题+回答估计约 8 tokens. 6轮累计约 48 tokens.
    memory, extractor = make_memory([extraction()], min_tokens=45, min_tool_calls=3)

    for index in range(1, 6):
        memory.add_turn(f"普通问题 {index}", f"回答 {index}")
        assert extractor.calls == []  # 前5轮累计 tokens < 45，不触发
        assert len(memory.pending_turns) == index

    # 第6轮，累计 tokens 达 48 >= 45，且助手回复是自然停顿，触发！
    memory.add_turn("普通问题 6", "回答 6")
    assert len(extractor.calls) == 1
    assert memory.pending_turns == ()


def test_trigger_on_tool_call_threshold_and_token_threshold_even_if_no_natural_break():
    # 设定 min_tokens=70, min_tool_calls=3
    # 模拟助手回复带有 "tool_call"，不是自然停顿（每轮一个 tool_call）
    memory, extractor = make_memory([extraction()], min_tokens=70, min_tool_calls=3)

    # 前2轮：虽然助手回复是非自然停顿，但 token 增长还不够，不触发
    memory.add_turn("普通问题 1", "执行 tool_call 动作 1")
    memory.add_turn("普通问题 2", "执行 tool_call 动作 2")
    assert extractor.calls == []

    # 第3轮：累计 tool calls 达到 3，但累计 tokens 约 36，未到 70，不触发
    memory.add_turn("普通问题 3", "执行 tool_call 动作 3")
    assert extractor.calls == []

    # 第4轮、第5轮继续积攒
    memory.add_turn("普通问题 4", "执行 tool_call 动作 4")
    memory.add_turn("普通问题 5", "执行 tool_call 动作 5")
    assert extractor.calls == []

    # 第6轮：累计 tool calls = 6 >= 3，累计 tokens 达 78 >= 70
    # 满足 (has_met_token_threshold and has_met_tool_call_threshold) 条件，强行触发！
    memory.add_turn("普通问题 6", "执行 tool_call 动作 6")
    assert len(extractor.calls) == 1
    assert memory.pending_turns == ()


def test_upsert_update_and_delete_preserve_creation_turn():
    memory, _ = make_memory([
        extraction(upsert("preference", "user.favorite_color", "蓝色")),
        extraction(upsert("preference", "user.favorite_color", "绿色")),
        extraction(delete("preference", "user.favorite_color")),
    ])

    memory.add_turn("我喜欢蓝色", "好的")
    first = memory.entries[0]
    memory.add_turn("把颜色改成绿色", "已更新")
    updated = memory.entries[0]

    assert first.value == "蓝色"
    assert updated.value == "绿色"
    assert updated.created_turn == first.created_turn == 1
    assert updated.updated_turn == 2

    memory.add_turn("忘记我的颜色偏好", "已删除")
    assert memory.entries == ()


def test_empty_or_idempotent_operations_keep_state_version_and_prompt_stable():
    memory, _ = make_memory([
        extraction(upsert("identity", "user.name", "小林")),
        extraction(upsert("identity", "user.name", "小林")),
        extraction(),
    ])
    memory.add_turn("我叫小林", "你好")
    version = memory.state_version
    content = memory.get_context_messages()[0]["content"]

    memory.add_turn("我是小林", "知道了")
    memory.add_turn("我决定什么都不改", "好的")

    assert memory.state_version == version
    assert memory.get_context_messages()[0]["content"] == content


def test_sensitive_operations_are_filtered_deterministically():
    memory, _ = make_memory([extraction(
        upsert("fact", "credentials.api_key", "sk-secret"),
        upsert("identity", "user.name", "小林"),
    )])

    memory.add_turn("我叫小林，API Key 是 sk-secret", "收到")

    assert [(entry.key, entry.value) for entry in memory.entries] == [
        ("user.name", "小林")
    ]


def test_extraction_failure_keeps_state_and_clears_batch_without_retry_storm():
    memory, extractor = make_memory([
        extraction(upsert("identity", "user.name", "小林")),
        RuntimeError("extractor down"),
    ])
    memory.add_turn("我叫小林", "好的")
    before = memory.entries

    memory.add_turn("我喜欢蓝色", "好的")

    assert memory.entries == before
    assert memory.pending_turns == ()
    assert "extractor down" in memory.last_extraction_error

    memory.add_turn("今天天气怎么样", "晴天")
    assert len(extractor.calls) == 3
    assert memory.pending_turns == ()


def test_capacity_evicts_least_recently_updated_entry():
    memory, _ = make_memory([
        extraction(upsert("fact", "item.one", "1")),
        extraction(upsert("fact", "item.two", "2")),
        extraction(upsert("fact", "item.three", "3")),
    ], max_entries=2)

    memory.add_turn("我决定记录一", "好")
    memory.add_turn("我决定记录二", "好")
    memory.add_turn("我决定记录三", "好")

    assert [entry.key for entry in memory.entries] == ["item.three", "item.two"]


def test_context_places_structured_state_before_base_memory():
    memory, _ = make_memory([
        extraction(upsert("identity", "user.name", "小林")),
    ])
    memory.add_turn("我叫小林", "你好")

    messages = memory.get_context_messages()

    assert messages[0]["role"] == "system"
    assert "user.name" in messages[0]["content"]
    assert messages[1:] == [
        {"role": "user", "content": "我叫小林"},
        {"role": "assistant", "content": "你好"},
    ]


def test_context_order_is_structured_then_summary_then_recent_raw_turns():
    summarizer = FakeExtractor(["早期摘要"])

    class SummaryAdapter:
        def summarize(self, existing_summary, turns, max_tokens):
            return summarizer.outputs.pop(0)

    base = SummaryBufferMemory(
        max_recent_tokens=4,
        max_summary_tokens=20,
        summarizer=SummaryAdapter(),
        token_counter=len,
        tokens_per_message=0,
    )
    extractor = FakeExtractor([
        extraction(upsert("identity", "user.name", "小林")),
    ])
    memory = StructuredWorkingMemory(
        base,
        extractor,
        update_policy=TokenAndBreakUpdatePolicy(
            min_tokens_between_updates=1,
            min_tool_calls_between_updates=1,
        )
    )
    memory.add_turn("我叫小林", "你好")
    memory.add_turn("q", "a")

    messages = memory.get_context_messages()

    assert "user.name" in messages[0]["content"]
    assert "历史对话摘要" in messages[1]["content"]
    assert messages[2:] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_manual_flush_forget_and_clear():
    memory, extractor = make_memory([
        extraction(upsert("constraint", "response.language", "中文")),
    ], min_tokens=100)
    memory.add_turn("普通问题", "普通回答")

    assert memory.flush_pending() is True
    assert len(extractor.calls) == 1
    assert memory.forget("constraint", "response.language") is True
    assert memory.forget("constraint", "response.language") is False

    memory.add_turn("另一个普通问题", "回答")
    memory.clear()

    assert memory.entries == ()
    assert memory.pending_turns == ()
    assert memory.turns == ()
    assert memory.state_version == 0


def test_llm_extractor_uses_schema_and_parses_tool_arguments():
    arguments = json.dumps({
        "operations": [{
            "action": "upsert",
            "category": "preference",
            "key": "response.style",
            "value": "简洁",
        }]
    }, ensure_ascii=False)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        tool_call = SimpleNamespace(
            function=SimpleNamespace(arguments=arguments)
        )
        message = SimpleNamespace(tool_calls=[tool_call], content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    extractor = LLMWorkingStateExtractor(client, "test-model")

    result = extractor.extract(
        turns=(ConversationTurn("我偏好简洁回答", "好的"),),
        existing_entries=(),
    )

    assert result.operations[0].key == "response.style"
    assert calls[0]["temperature"] == 0.1
    assert calls[0]["tools"][0]["function"]["parameters"] == (
        MemoryExtraction.model_json_schema()
    )


def test_invalid_extractor_json_is_rejected():
    message = SimpleNamespace(tool_calls=[], content="not-json")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: response)
    ))
    extractor = LLMWorkingStateExtractor(client, "test-model")

    with pytest.raises(ValueError, match="无法解析"):
        extractor.extract((ConversationTurn("u", "a"),), ())


def test_failed_agent_turn_does_not_enter_pending_or_call_extractor():
    memory, extractor = make_memory([])

    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("agent failed")

    from phase3_agentic_rag import AgenticRAG

    agent = object.__new__(AgenticRAG)
    agent.llm_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    agent.llm_model = "test-model"
    agent.use_multi_hop = False
    agent.max_iterations = 5
    agent.max_tool_calls = 3
    agent.max_retrieval_retries = 2

    with pytest.raises(RuntimeError, match="agent failed"):
        agent.query("我喜欢蓝色", verbose=False, memory=memory)

    assert memory.pending_turns == ()
    assert extractor.calls == []


def test_cli_builds_structured_wrapper_around_selected_base_strategy():
    from phase4_main import build_memory, get_base_memory, parse_args

    args = parse_args([
        "--strategy",
        "summary",
        "--structured-state",
        "--token-budget",
        "120",
        "--summary-token-budget",
        "80",
    ])

    class Summarizer:
        def summarize(self, existing_summary, turns, max_tokens):
            return "summary"

    state_extractor = FakeExtractor([])
    memory = build_memory(
        args,
        summarizer=Summarizer(),
        state_extractor=state_extractor,
    )

    assert isinstance(memory, StructuredWorkingMemory)
    base = get_base_memory(memory)
    assert isinstance(base, SummaryBufferMemory)
    assert base.max_recent_tokens == 120
    assert base.max_summary_tokens == 80


def test_structured_memory_file_persistence(tmp_path):
    import os
    filepath = str(tmp_path / "memory.json")

    # 1. 初始阶段：创建带有文件路径的记忆系统
    extractor = FakeExtractor([
        extraction(upsert("identity", "user.name", "小林")),
    ])
    memory = StructuredWorkingMemory(
        base_memory=ConversationWindowMemory(max_turns=20),
        extractor=extractor,
        update_policy=TokenAndBreakUpdatePolicy(
            min_tokens_between_updates=1,
            min_tool_calls_between_updates=1,
        ),
        filepath=filepath,
    )

    # 文件应该不存在
    assert not os.path.exists(filepath)

    # 2. 添加问答轮次触发提取，会触发写入文件
    memory.add_turn("我叫小林", "你好")
    assert len(memory.entries) == 1
    assert memory.entries[0].key == "user.name"
    assert memory.entries[0].value == "小林"

    # 文件现在应该已经存在并包含了序列化的条目
    assert os.path.exists(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["key"] == "user.name"
        assert data[0]["value"] == "小林"

    # 3. 创建一个新的记忆实例，传入同样的文件路径，验证自动加载
    extractor2 = FakeExtractor([])
    memory2 = StructuredWorkingMemory(
        base_memory=ConversationWindowMemory(max_turns=20),
        extractor=extractor2,
        update_policy=TokenAndBreakUpdatePolicy(
            min_tokens_between_updates=1,
            min_tool_calls_between_updates=1,
        ),
        filepath=filepath,
    )

    # 应该无需调用 LLM，直接从文件加载了已有的状态
    assert len(memory2.entries) == 1
    assert memory2.entries[0].key == "user.name"
    assert memory2.entries[0].value == "小林"

    # 4. 验证 clear 行为能同步清空文件
    memory2.clear()
    assert memory2.entries == ()
    with open(filepath, "r", encoding="utf-8") as f:
        data2 = json.load(f)
        assert data2 == []
