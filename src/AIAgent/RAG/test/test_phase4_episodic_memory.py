import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from phase4_episodic_memory import (
    EPISODIC_MEMORY_PREFIX,
    EpisodeReflection,
    EpisodicAgent,
    EpisodicMemory,
    LLMEpisodeReflector,
    redact_sensitive_text,
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


class FakeReflector:
    def __init__(self, outputs=None):
        self.outputs = list(outputs or [])
        self.calls = []

    def reflect(self, task, answer, steps, error=None):
        self.calls.append((task, answer, steps, error))
        if self.outputs:
            output = self.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output
        return reflection()


def reflection(outcome="success", summary="完成任务"):
    return EpisodeReflection(
        outcome=outcome,
        summary=summary,
        strategy="先检索再回答",
        lessons=["复用可靠证据"],
        pitfalls=["避免无依据推断"],
    )


def make_memory(tmp_path, **kwargs):
    return EpisodicMemory(
        filepath=tmp_path / "episodes.json",
        embedder=kwargs.pop("embedder", FakeEmbedder()),
        reflector=kwargs.pop("reflector", FakeReflector()),
        **kwargs,
    )


def test_record_creates_versioned_json_and_loads_across_instances(tmp_path):
    embedder = FakeEmbedder({"部署": [1.0, 0.0]})
    reflector = FakeReflector([reflection(summary="部署成功")])
    memory = make_memory(tmp_path, embedder=embedder, reflector=reflector)

    episode = memory.record(
        "部署服务",
        result={
            "answer": "完整答案不应持久化",
            "steps": [{
                "tool": "search_knowledge_base",
                "args": {"query": "部署"},
                "result_preview": "找到部署文档",
            }],
        },
    )

    assert episode is not None
    payload = json.loads((tmp_path / "episodes.json").read_text("utf-8"))
    assert payload["schema_version"] == 2
    assert "完整答案不应持久化" not in json.dumps(payload, ensure_ascii=False)

    loaded = EpisodicMemory(
        tmp_path / "episodes.json",
        embedder=embedder,
        reflector=reflector,
    )
    assert loaded.episodes == (episode,)
    assert loaded.last_load_error is None


def test_capacity_delete_and_clear_are_persisted(tmp_path):
    memory = make_memory(tmp_path, max_episodes=2)
    first = memory.record("任务一", result={})
    second = memory.record("任务二", result={})
    third = memory.record("任务三", result={})

    assert first is not None and second is not None and third is not None
    assert [item.task for item in memory.episodes] == ["任务二", "任务三"]
    assert memory.delete(second.id) is True
    assert memory.delete("missing") is False
    assert [item.task for item in memory.episodes] == ["任务三"]
    assert memory.clear() is True
    assert memory.episodes == ()

    payload = json.loads((tmp_path / "episodes.json").read_text("utf-8"))
    assert payload["episodes"] == []


def test_corrupt_or_wrong_version_file_loads_as_empty_without_overwrite(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text("not-json", encoding="utf-8")

    memory = EpisodicMemory(path, FakeEmbedder(), FakeReflector())

    assert memory.episodes == ()
    assert "JSONDecodeError" in memory.last_load_error
    assert path.read_text("utf-8") == "not-json"

    path.write_text(json.dumps({"schema_version": 999, "episodes": []}))
    memory = EpisodicMemory(path, FakeEmbedder(), FakeReflector())
    assert memory.episodes == ()
    assert "schema_version" in memory.last_load_error


def test_recall_sorts_filters_and_honors_top_k(tmp_path):
    embedder = FakeEmbedder({
        "Python": [1.0, 0.0],
        "Java": [0.8, 0.6],
        "烹饪": [0.0, 1.0],
    })
    memory = make_memory(
        tmp_path,
        embedder=embedder,
        top_k=2,
        min_similarity=0.7,
    )
    memory.record("Python 部署", result={})
    memory.record("Java 部署", result={})
    memory.record("烹饪晚餐", result={})

    recalled = memory.recall("Python 新任务")

    assert [item.episode.task for item in recalled] == [
        "Python 部署",
        "Java 部署",
    ]
    assert recalled[0].score == pytest.approx(1.0)
    assert memory.recall("Python", top_k=1)[0].episode.task == "Python 部署"


def test_empty_and_dimension_mismatch_recall_degrade_to_empty(tmp_path):
    memory = make_memory(tmp_path)
    assert memory.recall("anything") == ()

    memory.record("任务", result={})
    memory.embedder = FakeEmbedder({"查询": [1.0, 0.0, 0.0]})
    assert memory.recall("查询") == ()
    assert "维度" in memory.last_recall_error


def test_reflection_failure_uses_fallback_and_still_records(tmp_path):
    reflector = FakeReflector([RuntimeError("reflector down")])
    memory = make_memory(tmp_path, reflector=reflector)

    episode = memory.record("任务", result={"answer": "done"})

    assert episode is not None
    assert episode.outcome == "partial"
    assert "reflector down" in memory.last_reflection_error
    assert memory.last_recording_error is None


def test_embedding_or_save_failure_does_not_mutate_existing_state(tmp_path):
    memory = make_memory(tmp_path)
    first = memory.record("任务一", result={})
    assert first is not None

    memory.embedder = SimpleNamespace(
        embed_texts=lambda texts: [[1.0, 2.0, 3.0]],
        embed_query=lambda query: [1.0, 2.0, 3.0],
    )
    assert memory.record("维度错误", result={}) is None
    assert memory.episodes == (first,)

    memory.embedder = FakeEmbedder()
    memory._save = lambda episodes: (_ for _ in ()).throw(OSError("disk full"))
    assert memory.record("写入错误", result={}) is None
    assert memory.episodes == (first,)
    assert "disk full" in memory.last_recording_error


def test_sensitive_data_is_redacted_before_reflection_embedding_and_disk(tmp_path):
    reflector = FakeReflector([EpisodeReflection(
        outcome="success",
        summary="使用 API Key=sk-abcdefgh 完成",
        strategy="Bearer abc.def.ghi",
        lessons=["password: hunter2"],
    )])
    embedder = FakeEmbedder()
    memory = make_memory(tmp_path, reflector=reflector, embedder=embedder)

    episode = memory.record(
        "调用接口 API Key=sk-abcdefgh",
        result={
            "answer": "password: hunter2",
            "steps": [{
                "tool": "call_api",
                "args": {"access_token": "secret-token"},
                "result_preview": "Bearer abc.def.ghi",
            }],
        },
    )

    assert episode is not None
    persisted = (tmp_path / "episodes.json").read_text("utf-8")
    for secret in ("sk-abcdefgh", "hunter2", "abc.def.ghi", "secret-token"):
        assert secret not in persisted
        assert secret not in " ".join(embedder.texts)
    call = reflector.calls[0]
    assert "sk-abcdefgh" not in call[0]
    assert "hunter2" not in call[1]
    assert call[2][0]["args"]["access_token"] == "[REDACTED]"
    assert "[REDACTED]" in redact_sensitive_text("password=hunter2")


def test_llm_reflector_parses_forced_tool_call():
    arguments = reflection().model_dump_json()
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            function=SimpleNamespace(arguments=arguments)
        )],
    )
    completions = SimpleNamespace(
        calls=[],
        create=lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        ),
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    output = LLMEpisodeReflector(client, "test-model").reflect(
        "任务", "回答", (), None
    )

    assert output.outcome == "success"
    assert output.summary == "完成任务"


class CapturingAgent:
    def __init__(self, *, error=None):
        self.error = error
        self.context_messages = None

    def query(self, question, verbose=True, memory=None):
        self.context_messages = memory.get_context_messages()
        if self.error:
            raise self.error
        memory.add_turn(question, "answer")
        return {
            "answer": "answer",
            "steps": [{"tool": "direct_answer", "args": {}, "result_preview": ""}],
            "used_retrieval": False,
        }


def test_agent_injects_experience_after_short_term_context_and_records(tmp_path):
    embedder = FakeEmbedder({"相似": [1.0, 0.0]})
    reflector = FakeReflector([
        reflection(summary="旧经验"),
        reflection(summary="新经验"),
    ])
    episodic_memory = make_memory(
        tmp_path,
        embedder=embedder,
        reflector=reflector,
        min_similarity=0.1,
    )
    episodic_memory.record("相似旧任务", result={})
    short_term = ConversationWindowMemory(max_turns=3)
    short_term.add_turn("旧问题", "旧回答")
    agent = CapturingAgent()

    result = EpisodicAgent(agent, episodic_memory).query(
        "相似新任务", verbose=False, memory=short_term
    )

    assert result["answer"] == "answer"
    assert agent.context_messages[:2] == [
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
    ]
    assert agent.context_messages[2]["role"] == "system"
    assert agent.context_messages[2]["content"].startswith(EPISODIC_MEMORY_PREFIX)
    assert [turn.user for turn in short_term.turns] == ["旧问题", "相似新任务"]
    assert episodic_memory.episodes[-1].reflection.summary == "新经验"


def test_agent_records_runtime_failure_and_reraises_original_error(tmp_path):
    # 即使反思器误判为 success，运行时异常也必须确定性标为 failure。
    memory = make_memory(tmp_path, reflector=FakeReflector([reflection("success")]))
    agent = CapturingAgent(error=RuntimeError("agent failed"))

    with pytest.raises(RuntimeError, match="agent failed"):
        EpisodicAgent(agent, memory).query("失败任务", verbose=False)

    assert len(memory) == 1
    assert memory.episodes[0].outcome == "failure"


def test_memory_failures_never_replace_successful_agent_answer(tmp_path):
    embedder = SimpleNamespace(
        embed_texts=lambda texts: (_ for _ in ()).throw(RuntimeError("embed down")),
        embed_query=lambda query: (_ for _ in ()).throw(RuntimeError("query down")),
    )
    memory = make_memory(tmp_path, embedder=embedder)
    agent = CapturingAgent()

    result = EpisodicAgent(agent, memory).query("任务", verbose=False)

    assert result["answer"] == "answer"
    assert memory.episodes == ()
    assert "embed down" in memory.last_recording_error


def test_cli_exposes_episodic_defaults():
    from phase4_main import parse_args

    args = parse_args(["--episodic-memory-file", "episodes.json"])

    assert args.episodic_memory_file == "episodes.json"
    assert args.episodic_top_k == 3
    assert args.episodic_min_score == 0.35
    assert args.episodic_max_episodes == 200
    assert args.episodic_retention_days == 30


def test_agent_does_not_record_when_no_steps(tmp_path):
    episodic_memory = make_memory(tmp_path)
    agent = SimpleNamespace(
        query=lambda question, verbose=True, memory=None: {
            "answer": "answer",
            "steps": [],
            "used_retrieval": False,
        }
    )

    result = EpisodicAgent(agent, episodic_memory).query(
        "简单闲聊", verbose=False
    )

    assert result["answer"] == "answer"
    assert len(episodic_memory.episodes) == 0


def test_v1_episodic_memory_loads_with_forgetting_defaults(tmp_path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "episodes": [{
            "id": "old",
            "task": "旧任务",
            "outcome": "success",
            "reflection": reflection().model_dump(),
            "steps": [],
            "created_at": "2026-01-01T00:00:00+00:00",
            "embedding": [1.0, 0.0],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    memory = EpisodicMemory(path, FakeEmbedder(), FakeReflector())

    assert memory.episodes[0].importance == 0.5
    assert memory.episodes[0].recall_count == 0
    assert memory.episodes[0].last_recalled_at is None


def test_episodic_prune_and_recall_access_statistics(tmp_path):
    memory = make_memory(tmp_path, min_similarity=-1.0)
    normal = memory.record("普通任务", result={})
    failed = memory.record("失败任务", result={}, error=RuntimeError("failed"))
    assert normal is not None and failed is not None
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memory._episodes = [
        normal.model_copy(update={
            "created_at": start.isoformat(),
            "importance": 0.5,
        }),
        failed.model_copy(update={
            "created_at": start.isoformat(),
            "recall_count": 5,
        }),
    ]

    removed = memory.prune(start + timedelta(days=46))

    assert [episode.task for episode in removed] == ["普通任务"]
    assert memory.recall("失败")
    assert memory.episodes[0].recall_count == 6
    assert memory.episodes[0].last_recalled_at is not None
    memory._episodes[0] = memory.episodes[0].model_copy(update={
        "last_recalled_at": (start + timedelta(days=46)).isoformat(),
    })
    assert memory.prune(start + timedelta(days=149))[0].task == "失败任务"


def test_episodic_recall_access_save_failure_keeps_retrieval_available(tmp_path):
    memory = make_memory(tmp_path, min_similarity=-1.0)
    assert memory.record("任务", result={})
    before = memory.episodes
    memory._save = lambda episodes: (_ for _ in ()).throw(OSError("disk full"))

    assert memory.recall("任务")
    assert memory.episodes == before
    assert "disk full" in memory.last_access_error


def test_episodic_record_auto_prunes_highest_forgetting_score(tmp_path):
    memory = make_memory(tmp_path, max_episodes=2)
    stale = memory.record("易忘任务", result={})
    important = memory.record("重要任务", result={}, error=RuntimeError("失败"))
    assert stale is not None and important is not None
    now = datetime.now(timezone.utc)
    memory._episodes = [
        stale.model_copy(update={
            "created_at": (now - timedelta(days=40)).isoformat(),
            "importance": 0.0,
        }),
        important.model_copy(update={
            "created_at": (now - timedelta(days=40)).isoformat(),
            "importance": 0.9,
        }),
    ]

    assert memory.record("新任务", result={})

    assert [episode.task for episode in memory.episodes] == ["重要任务", "新任务"]
    assert [episode.task for episode in memory.last_pruned] == ["易忘任务"]
