import json
from types import SimpleNamespace

import pytest

from phase1_chunker import Chunk
from phase1_dense_retriever import SearchResult
from phase3_agentic_rag import AgenticRAG
from phase3_hop_assessor import HopAssessment, HopAssessor
from phase3_query_decomposer import QueryDecomposer, QueryPlan, QueryStep
from phase3_router import KnowledgeBase, KnowledgeRouter
from phase3_self_rag import RetrievalAssessment, SelfRAGAssessor


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def llm_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("LLM mock 没有剩余响应")
        return self.responses.pop(0)


class FakeAssessor:
    def __init__(self, assessments):
        self.assessments = list(assessments)
        self.calls = []

    def assess(self, question, results, verbose=True):
        self.calls.append((question, list(results)))
        if not self.assessments:
            raise AssertionError("Assessor mock 没有剩余响应")
        return self.assessments.pop(0)


class FakeHopAssessor:
    def __init__(self, assessments):
        self.assessments = list(assessments)
        self.calls = []

    def assess(self, **kwargs):
        self.calls.append(kwargs)
        if not self.assessments:
            raise AssertionError("HopAssessor mock 没有剩余响应")
        value = self.assessments.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def assessment(action="answer", suggested_query=None):
    return RetrievalAssessment(
        relevance="relevant" if action == "answer" else "partially_relevant",
        sufficiency="sufficient" if action == "answer" else "insufficient",
        action=action,
        reason=f"decision={action}",
        suggested_query=suggested_query,
    )


def hop_assessment(
    relevance="relevant",
    sufficiency="sufficient",
    can_answer=False,
    facts=(),
    entities=None,
    suggested_query=None,
    reason="hop assessment",
):
    return HopAssessment(
        relevance=relevance,
        sufficiency=sufficiency,
        can_answer_question=can_answer,
        extracted_facts=list(facts),
        resolved_entities=entities or {},
        suggested_query=suggested_query,
        reason=reason,
    )


def search_result(content="evidence", source="doc.md", score=0.9):
    return SearchResult(
        chunk=Chunk(content=content, metadata={"source": source}),
        score=score,
    )


def make_agent(responses, assessments=(), **overrides):
    agent = object.__new__(AgenticRAG)
    completions = FakeCompletions(responses)
    agent.llm_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    agent.llm_model = "test-model"
    agent.top_k = 3
    agent.use_reranker = False
    agent.use_router = False
    agent.use_multi_hop = True
    agent.max_iterations = 5
    agent.max_tool_calls = 3
    agent.max_retrieval_retries = 2
    agent.max_hop_retries = 1
    agent.max_replans = 1
    agent.max_multi_hop_steps = 6
    agent.reranker = None
    agent.router = None
    agent.decomposer = None
    agent.hop_assessor = None
    agent._knowledge_bases = {}
    agent.assessor = FakeAssessor(assessments)
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent, completions


def test_direct_answer_does_not_retrieve():
    agent, completions = make_agent([
        llm_response(tool_calls=[
            tool_call("direct_answer", json.dumps({"answer": "2"}))
        ])
    ])

    result = agent.query("1+1=?", verbose=False)

    assert result["answer"] == "2"
    assert result["used_retrieval"] is False
    assert [step["tool"] for step in result["steps"]] == ["direct_answer"]
    assert all(
        tool["function"]["name"] != "assess_retrieval_quality"
        for tool in completions.calls[0]["tools"]
    )


def test_search_always_runs_assessment_before_final_answer():
    agent, _ = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("search_knowledge_base", json.dumps({"query": "RAG"}))
            ]),
            llm_response(content="grounded answer"),
        ],
        assessments=[assessment("answer")],
    )
    evidence = search_result()
    agent._exec_search = lambda query, verbose=True: ("context", [evidence])

    result = agent.query("什么是 RAG？", verbose=False)

    assert result["answer"] == "grounded answer"
    assert result["used_retrieval"] is True
    assert [step["tool"] for step in result["steps"]] == [
        "search_knowledge_base",
        "assess_retrieval_quality",
    ]
    assert agent.assessor.calls == [("什么是 RAG？", [evidence])]


def test_refine_uses_suggested_query_and_honors_retry_limit():
    agent, _ = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("search_knowledge_base", json.dumps({"query": "q1"}))
            ]),
            llm_response(content="best effort"),
        ],
        assessments=[
            assessment("refine", "q2"),
            assessment("refine", "q3"),
            assessment("refine", "q4"),
        ],
        max_retrieval_retries=2,
    )
    queries = []

    def fake_search(query, verbose=True):
        queries.append(query)
        return f"context:{query}", [search_result(query)]

    agent._exec_search = fake_search

    result = agent.query("complex", verbose=False)

    assert result["answer"] == "best effort"
    assert queries == ["q1", "q2", "q3"]
    assert [step["tool"] for step in result["steps"]].count(
        "assess_retrieval_quality"
    ) == 3


def test_empty_knowledge_base_falls_back_without_index_error():
    agent, completions = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("search_knowledge_base", json.dumps({"query": "missing"}))
            ]),
            llm_response(content="没有可靠资料"),
        ]
    )
    agent.assessor = SelfRAGAssessor(llm_client=None, model="unused")

    result = agent.query("missing", verbose=False)

    assert result["answer"] == "没有可靠资料"
    assert "未找到相关信息" in completions.calls[1]["messages"][-1]["content"]
    assert "fallback" in completions.calls[1]["messages"][-1]["content"]


def test_consecutive_queries_do_not_share_retrieval_results():
    agent, _ = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("search_knowledge_base", json.dumps({"query": "first"}))
            ]),
            llm_response(content="first answer"),
            llm_response(tool_calls=[
                tool_call("search_knowledge_base", json.dumps({"query": "second"}))
            ]),
            llm_response(content="second answer"),
        ],
        assessments=[assessment("answer"), assessment("fallback")],
    )
    first_result = search_result("first evidence")
    batches = iter([("first context", [first_result]), ("no context", [])])
    agent._exec_search = lambda query, verbose=True: next(batches)

    agent.query("first question", verbose=False)
    agent.query("second question", verbose=False)

    assert agent.assessor.calls[0][1] == [first_result]
    assert agent.assessor.calls[1][1] == []
    assert not hasattr(agent, "_last_results")


def test_disabling_multi_hop_removes_tool():
    agent, _ = make_agent([], use_multi_hop=False)

    names = [tool["function"]["name"] for tool in agent._available_tools()]

    assert names == ["search_knowledge_base", "direct_answer"]
    assert "multi_hop_search" not in agent._system_prompt()


def test_multi_hop_uses_only_declared_dependencies_and_deduplicates():
    rewrite_response = llm_response(content="rewritten step 3")
    agent, completions = make_agent([rewrite_response])
    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=1, query="step 1", depends_on=None),
            QueryStep(step_id=2, query="step 2", depends_on=None),
            QueryStep(step_id=3, query="placeholder", depends_on=[1]),
        ])
    )
    agent.hop_assessor = FakeHopAssessor([
        hop_assessment(facts=["fact A"], entities={"entity": "A"}),
        hop_assessment(facts=["fact B"], entities={"entity": "B"}),
        hop_assessment(facts=["fact A reused"]),
    ])
    result_a = search_result("evidence A", "a.md")
    result_b = search_result("evidence B", "b.md")
    result_by_query = {
        "step 1": ("context A", [result_a]),
        "step 2": ("context B", [result_b]),
        "rewritten step 3": ("context A duplicate", [result_a]),
    }
    agent._exec_search = lambda query, verbose=False: result_by_query[query]
    step_log = []

    output, results = agent._exec_multi_hop_search(
        "complex", verbose=False, step_log=step_log
    )

    rewrite_prompt = completions.calls[0]["messages"][0]["content"]
    assert "fact A" in rewrite_prompt
    assert "fact B" not in rewrite_prompt
    assert results == [result_a, result_b]
    assert [item["tool"] for item in step_log].count("multi_hop_step") == 3
    assert [item["tool"] for item in step_log].count("assess_multi_hop_step") == 3
    assert "rewritten step 3" in output


def test_missing_multi_hop_dependency_stops_when_replan_is_unavailable():
    agent, completions = make_agent([])
    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=2, query="safe fallback", depends_on=[99]),
        ])
    )
    agent.hop_assessor = FakeHopAssessor([])
    queries = []
    agent._exec_search = lambda query, verbose=False: (
        queries.append(query) or "context",
        [search_result()],
    )
    step_log = []

    agent._exec_multi_hop_search("complex", verbose=False, step_log=step_log)

    assert queries == []
    assert completions.calls == []
    assert step_log[0]["args"]["status"] == "invalid_dependency"
    assert step_log[1]["tool"] == "replan_multi_hop"
    assert step_log[1]["args"]["status"] == "failed"


def test_multi_hop_retries_current_step_with_suggested_query():
    agent, _ = make_agent([], max_hop_retries=1)
    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=1, query="q1", depends_on=None),
        ])
    )
    agent.hop_assessor = FakeHopAssessor([
        hop_assessment(
            relevance="partially_relevant",
            sufficiency="insufficient",
            suggested_query="q1 refined",
        ),
        hop_assessment(facts=["confirmed fact"]),
    ])
    queries = []
    agent._exec_search = lambda query, verbose=False: (
        queries.append(query) or f"context:{query}",
        [search_result(query)],
    )
    step_log = []

    output, _ = agent._exec_multi_hop_search(
        "complex", verbose=False, step_log=step_log
    )

    assert queries == ["q1", "q1 refined"]
    assert any(item["tool"] == "retry_multi_hop_step" for item in step_log)
    assert "confirmed fact" in output


def test_retry_failure_replans_only_remaining_steps_and_keeps_completed_facts():
    agent, _ = make_agent([], max_hop_retries=1, max_replans=1)
    replan_calls = []

    def replan(**kwargs):
        replan_calls.append(kwargs)
        return QueryPlan(steps=[
            QueryStep(step_id=4, query="replacement", depends_on=[1]),
        ])

    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=1, query="first", depends_on=None),
            QueryStep(step_id=2, query="failing", depends_on=[1]),
            QueryStep(step_id=3, query="stale remaining", depends_on=[2]),
        ]),
        replan=replan,
    )
    agent.hop_assessor = FakeHopAssessor([
        hop_assessment(facts=["first fact"]),
        hop_assessment(
            relevance="partially_relevant",
            sufficiency="insufficient",
            suggested_query="failing refined",
        ),
        hop_assessment(
            relevance="irrelevant",
            sufficiency="insufficient",
        ),
        hop_assessment(facts=["replacement fact"]),
    ])
    queries = []
    agent._resolve_multi_hop_query = lambda step, dependencies: step.query
    agent._exec_search = lambda query, verbose=False: (
        queries.append(query) or f"context:{query}",
        [search_result(query, f"{query}.md")],
    )
    step_log = []

    output, _ = agent._exec_multi_hop_search(
        "complex", verbose=False, step_log=step_log
    )

    assert queries == ["first", "failing", "failing refined", "replacement"]
    assert replan_calls[0]["completed_steps"][0]["facts"] == ["first fact"]
    assert [step.query for step in replan_calls[0]["remaining_steps"]] == [
        "stale remaining"
    ]
    assert "replacement fact" in output
    assert any(
        item["tool"] == "replan_multi_hop"
        and item["args"]["status"] == "success"
        for item in step_log
    )


def test_multi_hop_finishes_early_when_accumulated_evidence_is_enough():
    agent, _ = make_agent([])
    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=1, query="enough", depends_on=None),
            QueryStep(step_id=2, query="must not run", depends_on=[1]),
        ])
    )
    agent.hop_assessor = FakeHopAssessor([
        hop_assessment(can_answer=True, facts=["complete answer fact"]),
    ])
    queries = []
    agent._exec_search = lambda query, verbose=False: (
        queries.append(query) or "context",
        [search_result()],
    )
    step_log = []

    output, _ = agent._exec_multi_hop_search(
        "complex", verbose=False, step_log=step_log
    )

    assert queries == ["enough"]
    assert "已足以回答原问题" in output
    assert any(item["tool"] == "finish_multi_hop" for item in step_log)


def test_hop_assessment_error_with_evidence_continues_conservatively():
    agent, completions = make_agent([llm_response(content="resolved next")])
    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=1, query="first", depends_on=None),
            QueryStep(step_id=2, query="next", depends_on=[1]),
        ])
    )
    agent.hop_assessor = FakeHopAssessor([
        ValueError("bad assessment"),
        hop_assessment(facts=["second fact"]),
    ])
    agent._exec_search = lambda query, verbose=False: (
        f"context:{query}",
        [search_result(query)],
    )

    output, results = agent._exec_multi_hop_search("complex", verbose=False)

    resolver_prompt = completions.calls[0]["messages"][0]["content"]
    assert "fallback_evidence" in resolver_prompt
    assert len(results) == 2
    assert "评估失败" in output


def test_multi_hop_step_limit_is_hard_bound():
    agent, _ = make_agent([], max_multi_hop_steps=1)
    agent.decomposer = SimpleNamespace(
        decompose=lambda question, verbose=False: QueryPlan(steps=[
            QueryStep(step_id=1, query="one", depends_on=None),
            QueryStep(step_id=2, query="two", depends_on=None),
        ])
    )
    agent.hop_assessor = FakeHopAssessor([hop_assessment(facts=["one fact"])])
    queries = []
    agent._exec_search = lambda query, verbose=False: (
        queries.append(query) or "context",
        [search_result()],
    )

    output, _ = agent._exec_multi_hop_search("complex", verbose=False)

    assert queries == ["one"]
    assert "达到多跳步骤上限" in output


def test_final_crag_refine_uses_targeted_search_without_rerunning_plan():
    agent, _ = make_agent(
        [],
        assessments=[
            assessment("refine", "targeted gap"),
            assessment("answer"),
        ],
        max_retrieval_retries=2,
    )
    multi_hop_calls = []
    targeted_calls = []
    base_result = search_result("base", "base.md")
    target_result = search_result("target", "target.md")
    agent._exec_multi_hop_search = lambda query, verbose=True, step_log=None: (
        multi_hop_calls.append(query) or "multi context",
        [base_result],
    )
    agent._exec_search = lambda query, verbose=True: (
        targeted_calls.append(query) or "target context",
        [target_result],
    )
    step_log = []

    output = agent._run_retrieval_cycle(
        question="complex",
        initial_query="complex",
        tool_name="multi_hop_search",
        steps=step_log,
        verbose=False,
    )

    assert multi_hop_calls == ["complex"]
    assert targeted_calls == ["targeted gap"]
    assert "target context" in output
    assert any(
        item["args"].get("mode") == "multi_hop_targeted_refine"
        for item in step_log
        if item["tool"] == "search_knowledge_base"
    )


def test_query_plan_rejects_future_dependencies():
    invalid_plan = QueryPlan(steps=[
        QueryStep(step_id=1, query="first", depends_on=[2]),
        QueryStep(step_id=2, query="second", depends_on=None),
    ])

    with pytest.raises(ValueError, match="尚不可用的依赖"):
        QueryDecomposer._validate_plan(invalid_plan)


def test_hop_assessor_parses_structured_facts_without_chain_of_thought_field():
    expected = hop_assessment(
        can_answer=True,
        facts=["A founded B"],
        entities={"company": "B"},
    )
    completions = FakeCompletions([
        llm_response(tool_calls=[
            tool_call(
                "submit_hop_assessment",
                json.dumps(expected.model_dump()),
            )
        ])
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assessor = HopAssessor(client, "test-model")

    result = assessor.assess(
        original_question="question",
        step=QueryStep(step_id=1, query="query"),
        executed_query="query",
        dependency_facts=[],
        accumulated_facts=[],
        results=[search_result()],
    )

    assert result == expected
    assert "chain_of_thought" not in HopAssessment.model_json_schema()["properties"]


def test_replan_accepts_dependencies_on_completed_steps():
    plan_payload = {
        "steps": [
            {"step_id": 4, "query": "replacement", "depends_on": [1]},
        ]
    }
    completions = FakeCompletions([
        llm_response(tool_calls=[
            tool_call("submit_query_plan", json.dumps(plan_payload))
        ])
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    decomposer = QueryDecomposer(client, "test-model")

    plan = decomposer.replan(
        original_question="question",
        completed_steps=[
            {"step_id": 1, "query": "first", "facts": ["fact"], "entities": {}},
        ],
        failed_step=QueryStep(step_id=2, query="failed", depends_on=[1]),
        remaining_steps=[QueryStep(step_id=3, query="stale", depends_on=[2])],
        next_step_id=4,
    )

    assert plan.steps[0].step_id == 4
    assert plan.steps[0].depends_on == [1]


def test_tool_limit_completes_all_tool_messages_then_forces_answer():
    agent, completions = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("unknown_one", "{}", "call-1"),
                tool_call("unknown_two", "{}", "call-2"),
            ]),
            llm_response(content="forced answer"),
        ],
        max_tool_calls=1,
    )

    result = agent.query("question", verbose=False)

    assert result["answer"] == "forced answer"
    forced_messages = completions.calls[1]["messages"]
    tool_messages = [message for message in forced_messages if isinstance(message, dict) and message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call-1", "call-2"]
    assert "未执行" in tool_messages[1]["content"]


def test_malformed_arguments_fall_back_to_original_question():
    agent, _ = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("search_knowledge_base", "not-json")
            ]),
            llm_response(content="answer"),
        ],
        assessments=[assessment("answer")],
    )
    queries = []
    agent._exec_search = lambda query, verbose=True: (
        queries.append(query) or "context",
        [search_result()],
    )

    agent.query("original question", verbose=False)

    assert queries == ["original question"]


def test_mixed_direct_and_search_calls_do_not_return_direct_candidate_early():
    agent, _ = make_agent(
        [
            llm_response(tool_calls=[
                tool_call("direct_answer", json.dumps({"answer": "premature"}), "direct"),
                tool_call("search_knowledge_base", json.dumps({"query": "evidence"}), "search"),
            ]),
            llm_response(content="grounded final"),
        ],
        assessments=[assessment("answer")],
    )
    agent._exec_search = lambda query, verbose=True: (
        "context",
        [search_result()],
    )

    result = agent.query("question", verbose=False)

    assert result["answer"] == "grounded final"
    assert result["used_retrieval"] is True


class StaticRetriever:
    def search(self, question, top_k=3):
        return []


def make_router(selected_json):
    completions = FakeCompletions([llm_response(content=selected_json)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    router = KnowledgeRouter(client, "test-model")
    for name in ("tech", "general"):
        router.add_kb(KnowledgeBase(name, name, StaticRetriever()))
    return router


def test_router_preserves_intentional_empty_selection():
    router = make_router(json.dumps({"selected": [], "reason": "无需检索"}))

    decision = router.route("hello", verbose=False)

    assert decision.selected_kbs == []


def test_router_invalid_target_falls_back_to_all_knowledge_bases():
    router = make_router(json.dumps({"selected": ["invented"], "reason": "bad"}))

    decision = router.route("question", verbose=False)

    assert decision.selected_kbs == ["tech", "general"]
