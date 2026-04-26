"""
基于 rewrite_graph.GraphFlow 的 Multi-Agent v2。

这个版本刻意保留 rewrite_graph 的学习模型：

    State(TypedDict) + node_xxx(state) + route_after_xxx(state) + GraphFlow 连边

流程：

    planner -> coder -> reviewer -> tester -> arbiter -> route
                                                | accept / max_rounds -> END
                                                | reject              -> coder

和 realistic_multi_agent_v2.py 的区别：
- 这里不另写一个独立 Engine；
- 所有阶段都落在 GraphFlow 的节点和条件边上；
- 仍然可以从 rewrite_graph 的 add_node / add_edge / add_conditional_edges 理解整套系统。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

try:
    from Agent.StateMachine_GraphFlow.rewrite_graph import END, GraphFlow
except ModuleNotFoundError:
    # 兼容直接运行本文件：python src/Agent/MultiAgent/realistic_multi_agent_graph_v2.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from StateMachine_GraphFlow.rewrite_graph import END, GraphFlow


BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
MODEL = os.getenv("OPENAI_MODEL", "zai-org/GLM-4.6")


PLANNER_PROMPT = """
你是一名资深软件架构师。
把任务拆成真实可执行的开发计划。
只返回 JSON，字段：
- objective: string
- milestones: string[]
- acceptance_criteria: string[]
- risks: string[]
- candidate_test_commands: string[]
""".strip()

CODER_PROMPT = """
你是一名高级 Python 工程师。
根据任务、计划和上一轮反馈，产出可审查的代码。
只返回 JSON，字段：
- summary: string
- code: string
- notes: string[]
""".strip()

REVIEWER_PROMPT = """
你是一名严格的代码审查员。
检查候选代码的正确性、边界条件、可维护性和测试风险。
只返回 JSON，字段：
- approved: boolean
- issues: object[]，每项包含 severity: "high|medium|low", detail: string
- suggestions: string[]
- confidence: number
""".strip()

TESTER_PROMPT = """
你是一名测试工程师。
从允许列表中选择一个安全测试命令；如果没有合适命令，返回 run_tests=false。
只返回 JSON，字段：
- run_tests: boolean
- command: string
- rationale: string

允许的命令前缀只有：
- python -m pytest
- pytest
- python -m unittest
""".strip()

ARBITER_PROMPT = """
你是一名发版仲裁者。
综合计划、代码摘要、聚合审查和测试报告判断能否验收。
只返回 JSON，字段：
- accept: boolean
- reason: string
- remaining_risks: string[]
""".strip()


@dataclass
class TaskSpec:
    objective: str
    constraints: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)


class MultiAgentV2State(TypedDict):
    # rewrite_graph.State 的基础字段
    user_input: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    answer: str
    trace: list[str]

    # Multi-Agent 业务字段
    task: dict[str, Any]
    plan: dict[str, Any]
    coding_result: dict[str, Any]
    draft_code: str
    individual_reviews: list[dict[str, Any]]
    aggregated_review: dict[str, Any]
    tester_decision: dict[str, Any]
    test_report: dict[str, Any]
    verdict: dict[str, Any]
    rounds: list[dict[str, Any]]
    round_index: int
    max_rounds: int
    final_code: str


class JsonUtils:
    @staticmethod
    def parse_object(text: str) -> dict[str, Any]:
        if not text:
            return {}

        content = text.strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
                if content.startswith("json"):
                    content = content[4:].strip()

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(content[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}


class LLMGateway:
    def __init__(
        self,
        model: str = MODEL,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def ask_json(
        self, system_prompt: str, payload: dict[str, Any], temperature: float = 0.1
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "只返回有效 JSON。\n输入：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    timeout=self.timeout_seconds,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                parsed = JsonUtils.parse_object(raw)
                if parsed:
                    return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

            if attempt < self.max_retries:
                time.sleep(attempt * 1.1)

        raise RuntimeError(f"LLM JSON response failed after retries: {last_error}")


class SafeCommandRunner:
    def __init__(
        self, allow_execute: bool = True, allowed_prefixes: list[str] | None = None
    ) -> None:
        self.allow_execute = allow_execute
        self.allowed_prefixes = allowed_prefixes or [
            "python -m pytest",
            "pytest",
            "python -m unittest",
        ]

    def _is_allowed(self, command: str) -> bool:
        cmd = command.strip()
        return any(cmd.startswith(prefix) for prefix in self.allowed_prefixes)

    def run(self, command: str, timeout_seconds: int = 120) -> dict[str, Any]:
        command = command.strip()
        if not command:
            return {
                "ok": True,
                "skipped": True,
                "reason": "no command provided",
                "command": "",
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        if not self.allow_execute:
            return {
                "ok": True,
                "skipped": True,
                "reason": "execution disabled",
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        if not self._is_allowed(command):
            return {
                "ok": False,
                "skipped": True,
                "reason": "command blocked by allow-list policy",
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        try:
            completed = subprocess.run(
                command,
                shell=True,
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
            )
            return {
                "ok": completed.returncode == 0,
                "skipped": False,
                "reason": "",
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "skipped": False,
                "reason": "command timeout",
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "timeout",
            }


llm: LLMGateway | None = None
runner = SafeCommandRunner()


def get_llm() -> LLMGateway:
    global llm
    if llm is None:
        llm = LLMGateway()
    return llm


def trace(state: MultiAgentV2State, message: str) -> None:
    print(message)
    state["trace"].append(message)


def _task_spec(state: MultiAgentV2State) -> TaskSpec:
    return TaskSpec(**state["task"])


def _history_view(state: MultiAgentV2State) -> list[dict[str, Any]]:
    return [
        {
            "round": item["round_index"],
            "review": item["aggregated_review"],
            "test": {
                "ok": item["test_report"].get("ok"),
                "reason": item["test_report"].get("reason"),
                "command": item["test_report"].get("command"),
            },
            "verdict": item["verdict"],
        }
        for item in state["rounds"][-5:]
    ]


def node_planner(state: MultiAgentV2State) -> None:
    trace(state, "\n[planner] 制定计划")
    task = _task_spec(state)
    state["plan"] = get_llm().ask_json(
        PLANNER_PROMPT,
        {"task": asdict(task)},
        temperature=0.1,
    )
    trace(state, json.dumps(state["plan"], ensure_ascii=False, indent=2))


def node_coder(state: MultiAgentV2State) -> None:
    state["round_index"] += 1
    trace(state, f"\n[coder] 第 {state['round_index']} 轮生成代码")
    task = _task_spec(state)
    coding = get_llm().ask_json(
        CODER_PROMPT,
        {
            "task": asdict(task),
            "plan": state["plan"],
            "history": _history_view(state),
        },
        temperature=0.2,
    )
    state["coding_result"] = coding
    state["draft_code"] = str(coding.get("code", "")).strip() or "# empty code"
    trace(state, str(coding.get("summary", "")))
    trace(state, state["draft_code"][:1200])


def _single_review(
    reviewer_id: int, task: TaskSpec, plan: dict[str, Any], code_text: str
) -> dict[str, Any]:
    review = get_llm().ask_json(
        REVIEWER_PROMPT,
        {
            "reviewer_id": reviewer_id,
            "task": asdict(task),
            "plan": plan,
            "candidate_code": code_text,
        },
        temperature=0.1,
    )
    review.setdefault("approved", False)
    review.setdefault("issues", [])
    review.setdefault("suggestions", [])
    review.setdefault("confidence", 0.5)
    review["reviewer_id"] = reviewer_id
    return review


def node_reviewer(state: MultiAgentV2State) -> None:
    trace(state, "\n[reviewer] 多审查员评审")
    task = _task_spec(state)
    reviews = [
        _single_review(i + 1, task, state["plan"], state["draft_code"])
        for i in range(2)
    ]

    issues: list[dict[str, Any]] = []
    suggestions: list[str] = []
    approvals = 0
    for review in reviews:
        if bool(review.get("approved", False)):
            approvals += 1
        issues.extend(x for x in review.get("issues", []) if isinstance(x, dict))
        suggestions.extend(x for x in review.get("suggestions", []) if isinstance(x, str))

    high_count = sum(
        1 for issue in issues if str(issue.get("severity", "")).lower() == "high"
    )
    medium_count = sum(
        1 for issue in issues if str(issue.get("severity", "")).lower() == "medium"
    )
    state["individual_reviews"] = reviews
    state["aggregated_review"] = {
        "approved": approvals == len(reviews) and high_count == 0,
        "approval_ratio": f"{approvals}/{len(reviews)}",
        "issues": issues,
        "suggestions": suggestions,
        "issue_stats": {
            "high": high_count,
            "medium": medium_count,
            "low": len(issues) - high_count - medium_count,
        },
    }
    trace(state, json.dumps(state["aggregated_review"], ensure_ascii=False, indent=2))


def node_tester(state: MultiAgentV2State) -> None:
    trace(state, "\n[tester] 选择并执行安全测试")
    task = _task_spec(state)
    tester_decision = get_llm().ask_json(
        TESTER_PROMPT,
        {
            "task": asdict(task),
            "plan": state["plan"],
            "candidate_code": state["draft_code"],
            "aggregated_review": state["aggregated_review"],
        },
        temperature=0.0,
    )
    state["tester_decision"] = tester_decision

    if bool(tester_decision.get("run_tests", False)):
        state["test_report"] = runner.run(str(tester_decision.get("command", "")))
    else:
        state["test_report"] = {
            "ok": True,
            "skipped": True,
            "reason": "tester skipped execution",
            "command": str(tester_decision.get("command", "")),
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    state["test_report"]["rationale"] = tester_decision.get("rationale", "")
    trace(state, json.dumps(state["test_report"], ensure_ascii=False, indent=2))


def _hard_gate(state: MultiAgentV2State) -> dict[str, Any]:
    review = state["aggregated_review"]
    test_report = state["test_report"]
    high_count = int(review.get("issue_stats", {}).get("high", 0))

    if high_count > 0:
        return {"passed": False, "reason": "存在 high 级别审查问题"}
    if not bool(review.get("approved", False)):
        return {"passed": False, "reason": "聚合审查未通过"}
    if not bool(test_report.get("ok", False)):
        return {"passed": False, "reason": "测试未通过或命令被拦截"}
    return {"passed": True, "reason": "硬规则通过"}


def node_arbiter(state: MultiAgentV2State) -> None:
    trace(state, "\n[arbiter] 验收裁决")
    task = _task_spec(state)
    hard_gate = _hard_gate(state)
    verdict = get_llm().ask_json(
        ARBITER_PROMPT,
        {
            "task": asdict(task),
            "plan": state["plan"],
            "coding_summary": state["coding_result"].get("summary", ""),
            "aggregated_review": state["aggregated_review"],
            "test_report": {
                "ok": state["test_report"].get("ok"),
                "skipped": state["test_report"].get("skipped"),
                "reason": state["test_report"].get("reason"),
                "command": state["test_report"].get("command"),
                "returncode": state["test_report"].get("returncode"),
            },
            "hard_gate": hard_gate,
        },
        temperature=0.0,
    )

    if not hard_gate["passed"]:
        verdict["accept"] = False
        verdict["reason"] = f"{hard_gate['reason']}；{verdict.get('reason', '')}".strip()

    verdict.setdefault("accept", False)
    verdict.setdefault("reason", "")
    verdict.setdefault("remaining_risks", [])
    state["verdict"] = verdict

    round_record = {
        "round_index": state["round_index"],
        "coding_result": state["coding_result"],
        "individual_reviews": state["individual_reviews"],
        "aggregated_review": state["aggregated_review"],
        "tester_decision": state["tester_decision"],
        "test_report": state["test_report"],
        "verdict": state["verdict"],
    }
    state["rounds"].append(round_record)

    if bool(verdict.get("accept", False)):
        state["final_code"] = state["draft_code"]
        state["answer"] = state["final_code"]

    trace(state, json.dumps(state["verdict"], ensure_ascii=False, indent=2))


def route_after_arbiter(state: MultiAgentV2State) -> str:
    if bool(state["verdict"].get("accept", False)):
        return END
    if state["round_index"] >= state["max_rounds"]:
        state["final_code"] = state["draft_code"]
        state["answer"] = state["final_code"]
        state["trace"].append("route -> max_rounds reached")
        return END
    return "coder"


def build_multi_agent_v2() -> GraphFlow:
    g = GraphFlow()

    g.add_node("planner", node_planner)
    g.add_node("coder", node_coder)
    g.add_node("reviewer", node_reviewer)
    g.add_node("tester", node_tester)
    g.add_node("arbiter", node_arbiter)

    g.add_edge("planner", "coder")
    g.add_edge("coder", "reviewer")
    g.add_edge("reviewer", "tester")
    g.add_edge("tester", "arbiter")
    g.add_conditional_edges(
        "arbiter",
        route_after_arbiter,
        {
            "coder": "coder",
            END: END,
        },
    )
    g.set_start("planner")
    return g


def make_state(task: TaskSpec, max_rounds: int = 4) -> MultiAgentV2State:
    return {
        "user_input": task.objective,
        "messages": [],
        "tool_calls": [],
        "answer": "",
        "trace": [],
        "task": asdict(task),
        "plan": {},
        "coding_result": {},
        "draft_code": "",
        "individual_reviews": [],
        "aggregated_review": {},
        "tester_decision": {},
        "test_report": {},
        "verdict": {},
        "rounds": [],
        "round_index": 0,
        "max_rounds": max(1, max_rounds),
        "final_code": "",
    }

def build_default_task(task_text: str) -> TaskSpec:
    return TaskSpec(
        objective=task_text,
        constraints=[
            "优先处理边界情况",
            "代码应清晰、可维护",
            "必要时给出最小测试",
        ],
        deliverables=["可审查的 Python 代码", "简短实现说明"],
        context=["这是基于 rewrite_graph.GraphFlow 的多智能体学习版本"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphFlow-based Multi-Agent v2")
    parser.add_argument(
        "--task",
        type=str,
        default="写一个 Python 函数 quick_sort(items)，要求不修改原列表并处理空列表。",
        help="任务目标",
    )
    parser.add_argument("--rounds", type=int, default=4, help="最大迭代轮数")
    parser.add_argument(
        "--no-test-exec",
        action="store_true",
        help="只让 tester 规划测试，不实际执行命令",
    )
    return parser.parse_args()


def main() -> None:
    global runner

    args = parse_args()
    runner = SafeCommandRunner(allow_execute=not args.no_test_exec)

    task = build_default_task(args.task)
    graph = build_multi_agent_v2()
    state = make_state(task, max_rounds=args.rounds)

    final_state = graph.run(cast(Any, state), max_steps=1 + args.rounds * 4)

    print("\n=== Final Code ===")
    print(final_state["final_code"] or final_state["draft_code"])

    print("\n=== Graph Trace ===")
    for item in final_state["trace"]:
        print(item)


if __name__ == "__main__":
    main()
