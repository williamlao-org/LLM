import json
from types import SimpleNamespace

import pytest

from phase3_agentic_rag import AgenticRAG
from phase4_episodic_memory import EpisodeReflection, EpisodicAgent, EpisodicMemory
from phase4_semantic_memory import (
    SEMANTIC_MEMORY_PREFIX,
    SemanticAgent,
    SemanticMemory,
    SemanticOperation,
)
from phase4_structured_memory import (
    MemoryExtraction,
    MemoryOperation,
    StructuredWorkingMemory,
    TokenAndBreakUpdatePolicy,
)
from phase4_working_memory import ConversationWindowMemory


class FakeEmbedder:
    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.texts = []

    def _vector(self, text):
        for keyword, vector in self.vectors.items():
            if keyword in text:
                return list(vector)
        return [0.0, 1.0]

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, query):
        return self._vector(query)


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


def operation(action="upsert", key="user.city", value="上海"):
    return SemanticOperation(
        action=action,
        category="fact",
        key=key,
        value=value,
    )


def structured_operation(action="upsert", key="user.city", value="上海"):
    return MemoryOperation(
        action=action,
        category="fact",
        key=key,
        value=value,
    )


def make_semantic(tmp_path, **kwargs):
    return SemanticMemory(
        tmp_path / "semantic.json",
        kwargs.pop("embedder", FakeEmbedder()),
        **kwargs,
    )


def test_versioned_persistence_update_delete_and_clear(tmp_path):
    memory = make_semantic(tmp_path)
    assert memory.apply_operations([operation()])
    first = memory.entries[0]

    payload = json.loads((tmp_path / "semantic.json").read_text("utf-8"))
    assert payload["schema_version"] == 1
    assert payload["entries"][0]["value"] == "上海"

    assert memory.apply_operations([operation(value="纽约")])
    updated = memory.entries[0]
    assert updated.created_at == first.created_at
    assert updated.value == "纽约"

    loaded = make_semantic(tmp_path)
    assert loaded.entries == memory.entries
    assert loaded.delete("user.city") is True
    assert loaded.delete("user.city") is False
    assert loaded.apply_operations([operation(key="user.country", value="中国")])
    assert loaded.clear() is True
    assert loaded.entries == ()


def test_capacity_evicts_least_recently_updated(tmp_path):
    memory = make_semantic(tmp_path, max_entries=2)
    memory.apply_operations([
        operation(key="fact.one", value="1"),
        operation(key="fact.two", value="2"),
        operation(key="fact.three", value="3"),
    ])
    assert [entry.key for entry in memory.entries] == ["fact.three", "fact.two"]


def test_recall_sorts_filters_and_handles_dimension_mismatch(tmp_path):
    embedder = FakeEmbedder({
        "上海": [1.0, 0.0],
        "杭州": [0.8, 0.6],
        "Python": [0.0, 1.0],
    })
    memory = make_semantic(
        tmp_path,
        embedder=embedder,
        top_k=2,
        min_similarity=0.7,
    )
    memory.apply_operations([
        operation(key="user.city", value="上海"),
        operation(key="user.travel_city", value="杭州"),
        operation(key="user.language", value="Python"),
    ])

    recalled = memory.recall("上海附近")
    assert [item.entry.key for item in recalled] == [
        "user.city",
        "user.travel_city",
    ]
    assert recalled[0].score == pytest.approx(1.0)

    memory.embedder = SimpleNamespace(
        embed_texts=lambda texts: [[1.0, 0.0, 0.0]],
        embed_query=lambda query: [1.0, 0.0, 0.0],
    )
    assert memory.recall("上海") == ()
    assert "维度" in memory.last_recall_error


def test_corrupt_file_and_write_failure_preserve_state(tmp_path):
    path = tmp_path / "semantic.json"
    path.write_text("not-json", encoding="utf-8")
    corrupt = SemanticMemory(path, FakeEmbedder())
    assert corrupt.entries == ()
    assert "JSONDecodeError" in corrupt.last_load_error
    assert path.read_text("utf-8") == "not-json"

    path.unlink()
    memory = SemanticMemory(path, FakeEmbedder())
    memory.apply_operations([operation()])
    before = memory.entries
    memory._save = lambda entries: (_ for _ in ()).throw(OSError("disk full"))
    assert memory.apply_operations([operation(value="纽约")]) == ()
    assert memory.entries == before
    assert "disk full" in memory.last_write_error


def test_sensitive_fact_never_reaches_embedding_or_disk(tmp_path):
    embedder = FakeEmbedder()
    memory = make_semantic(tmp_path, embedder=embedder)

    assert memory.apply_operations([
        operation(key="credentials.api_key", value="sk-abcdefgh"),
    ]) == ()

    assert memory.entries == ()
    assert embedder.texts == []
    assert not (tmp_path / "semantic.json").exists()


def test_fact_stays_in_structured_working_memory_without_semantic_write(tmp_path):
    semantic = make_semantic(tmp_path)
    extractor = FakeExtractor([
        MemoryExtraction(operations=[structured_operation()]),
    ])
    memory = StructuredWorkingMemory(
        ConversationWindowMemory(max_turns=5),
        extractor,
        TokenAndBreakUpdatePolicy(min_tokens_between_updates=1),
    )

    memory.add_turn("我住在上海", "知道了")

    assert semantic.entries == ()
    assert [(entry.category, entry.key, entry.value) for entry in memory.entries] == [
        ("fact", "user.city", "上海")
    ]
    assert memory.state_version == 1
    messages = memory.get_context_messages()
    assert messages[0]["role"] == "system"
    assert "user.city" in messages[0]["content"]
    assert messages[1:] == [
        {"role": "user", "content": "我住在上海"},
        {"role": "assistant", "content": "知道了"},
    ]


def test_explicit_signal_bypasses_token_threshold_for_structured_fact():
    extractor = FakeExtractor([
        MemoryExtraction(operations=[structured_operation()]),
    ])
    memory = StructuredWorkingMemory(
        ConversationWindowMemory(max_turns=5),
        extractor,
        TokenAndBreakUpdatePolicy(min_tokens_between_updates=10_000),
    )

    memory.add_turn("请记住我住在上海", "记住了")

    assert len(extractor.calls) == 1
    assert memory.entries[0].value == "上海"


def test_extractor_does_not_mix_durable_semantic_facts_into_working_state(tmp_path):
    semantic = make_semantic(tmp_path)
    semantic.apply_operations([operation()])
    extractor = FakeExtractor([
        MemoryExtraction(operations=[structured_operation(value="纽约")]),
    ])
    memory = StructuredWorkingMemory(
        ConversationWindowMemory(max_turns=5),
        extractor,
        TokenAndBreakUpdatePolicy(min_tokens_between_updates=1),
    )

    memory.add_turn("我搬到纽约了", "已更新")

    existing_entries = extractor.calls[0][1]
    assert existing_entries == ()
    assert semantic.entries[0].value == "上海"
    assert memory.entries[0].value == "纽约"


def test_legacy_structured_fact_loads_without_automatic_migration(tmp_path):
    structured_path = tmp_path / "structured.json"
    structured_path.write_text(json.dumps([{
        "category": "fact",
        "key": "user.city",
        "value": "上海",
        "created_turn": 1,
        "updated_turn": 1,
    }], ensure_ascii=False), encoding="utf-8")
    semantic = make_semantic(tmp_path)
    memory = StructuredWorkingMemory(
        ConversationWindowMemory(),
        FakeExtractor([]),
        filepath=str(structured_path),
    )

    assert memory.entries[0].key == "user.city"
    assert semantic.entries == ()


class FakeReflector:
    def reflect(self, task, answer, steps, error=None):
        return EpisodeReflection(
            outcome="success",
            summary="旧经验",
            strategy="复用历史",
        )


class CapturingAgent:
    def __init__(self):
        self.context_messages = None

    def query(self, question, verbose=True, memory=None):
        self.context_messages = memory.get_context_messages()
        memory.add_turn(question, "answer")
        return {"answer": "answer", "steps": [], "used_retrieval": False}


def test_semantic_then_episodic_context_order(tmp_path):
    embedder = FakeEmbedder({"任务": [1.0, 0.0]})
    semantic = make_semantic(
        tmp_path,
        embedder=embedder,
        min_similarity=0.1,
    )
    semantic.apply_operations([operation(key="project.stack", value="任务技术栈")])
    episodic = EpisodicMemory(
        tmp_path / "episodes.json",
        embedder,
        FakeReflector(),
        min_similarity=0.1,
    )
    episodic.record("任务旧经验", result={})
    base = CapturingAgent()
    wrapped = SemanticAgent(EpisodicAgent(base, episodic), semantic)
    working = ConversationWindowMemory(max_turns=3)
    working.add_turn("旧问题", "旧回答")

    wrapped.query("任务新问题", verbose=False, memory=working)

    contents = [message["content"] for message in base.context_messages]
    semantic_index = next(
        index for index, content in enumerate(contents)
        if content.startswith(SEMANTIC_MEMORY_PREFIX)
    )
    episodic_index = next(
        index for index, content in enumerate(contents)
        if content.startswith("【历史任务经验")
    )
    assert semantic_index < episodic_index


def test_external_semantic_tool_registration_and_execution(tmp_path):
    semantic = make_semantic(tmp_path, min_similarity=-1.0)
    semantic.apply_operations([operation()])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name="search_semantic_memory",
                    arguments=json.dumps({"query": "住在哪里"}),
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
        else:
            message = SimpleNamespace(content="住在上海", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    agent = object.__new__(AgenticRAG)
    agent.llm_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent.llm_model = "test-model"
    agent.use_multi_hop = False
    agent.max_iterations = 3
    agent.max_tool_calls = 2
    agent.max_retrieval_retries = 1
    agent._external_tools = {}
    agent.register_tool(
        semantic.tool_spec(),
        semantic.execute_tool,
        "- **search_semantic_memory**: 检索长期事实",
    )

    result = agent.query("我住在哪里？", verbose=False)

    assert result["answer"] == "住在上海"
    assert result["steps"][0]["tool"] == "search_semantic_memory"
    assert SEMANTIC_MEMORY_PREFIX in calls[1]["messages"][-1]["content"]
    with pytest.raises(ValueError, match="重复"):
        agent.register_tool(semantic.tool_spec(), semantic.execute_tool)


def test_cli_exposes_semantic_defaults_without_structured_wrapper():
    from phase4_main import build_memory, parse_args

    args = parse_args(["--semantic-memory-file", "semantic.json"])
    memory = build_memory(args)

    assert args.semantic_top_k == 3
    assert args.semantic_min_score == 0.35
    assert args.semantic_max_entries == 500
    assert isinstance(memory, ConversationWindowMemory)


def test_structured_memory_routes_fact_to_semantic_sink(tmp_path):
    semantic = make_semantic(tmp_path)
    extractor = FakeExtractor([
        MemoryExtraction(operations=[structured_operation(value="北京")]),
    ])
    memory = StructuredWorkingMemory(
        ConversationWindowMemory(max_turns=5),
        extractor,
        TokenAndBreakUpdatePolicy(min_tokens_between_updates=1),
        semantic_sink=semantic,
    )

    memory.add_turn("我住在北京", "记住了")

    assert len(semantic.entries) == 1
    assert semantic.entries[0].key == "user.city"
    assert semantic.entries[0].value == "北京"
    assert memory.entries == ()


def test_structured_memory_evicts_legacy_fact_from_working_state(tmp_path):
    structured_path = tmp_path / "structured.json"
    structured_path.write_text(json.dumps([{
        "category": "fact",
        "key": "user.city",
        "value": "旧地址",
        "created_turn": 1,
        "updated_turn": 1,
    }], ensure_ascii=False), encoding="utf-8")

    semantic = make_semantic(tmp_path)
    extractor = FakeExtractor([
        MemoryExtraction(operations=[structured_operation(value="新地址")]),
    ])
    memory = StructuredWorkingMemory(
        ConversationWindowMemory(max_turns=5),
        extractor,
        TokenAndBreakUpdatePolicy(min_tokens_between_updates=1),
        filepath=str(structured_path),
        semantic_sink=semantic,
    )

    assert len(memory.entries) == 1
    assert memory.entries[0].value == "旧地址"

    memory.add_turn("我搬到新地址了", "已更新")

    assert len(semantic.entries) == 1
    assert semantic.entries[0].value == "新地址"
    assert memory.entries == ()

