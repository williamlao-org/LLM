"""
Modular multi-agent example powered by agent_graph_engine.py.

这个文件现在只负责定义“业务模块”和“业务图”：
- 引擎能力在 agent_graph_engine.py
- 当前文件展示如何像搭积木一样组装 planner / coder / reviewer / tester / arbiter

运行：
    python src/Agent/MultiAgent/modular_agent_graph.py --mode mock
    python src/Agent/MultiAgent/modular_agent_graph.py --mode llm --task "写 quick_sort"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

try:
    from Agent.MultiAgent.agent_graph_engine import (
        END,
        AgentGraph,
        AgentModule,
        ModuleContext,
        RunState,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Agent.MultiAgent.agent_graph_engine import (
        END,
        AgentGraph,
        AgentModule,
        ModuleContext,
        RunState,
    )


BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
MODEL = os.getenv("OPENAI_MODEL", "zai-org/GLM-4.6")


class JsonUtils:
    @staticmethod
    def parse_object(text: str) -> dict[str, Any]:
        content = text.strip()
        if not content:
            return {}

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
                parsed = JsonUtils.parse_object(response.choices[0].message.content or "")
                if parsed:
                    return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

            if attempt < self.max_retries:
                time.sleep(attempt * 1.1)

        raise RuntimeError(f"LLM JSON response failed after retries: {last_error}")


class SafeCommandRunner:
    def __init__(self, allow_execute: bool = False) -> None:
        self.allow_execute = allow_execute
        self.allowed_prefixes = [
            "python -m pytest",
            "pytest",
            "python -m unittest",
        ]

    def run(self, command: str) -> dict[str, Any]:
        command = command.strip()
        if not command:
            return {"ok": True, "skipped": True, "reason": "no command"}
        if not self.allow_execute:
            return {
                "ok": True,
                "skipped": True,
                "reason": "execution disabled",
                "command": command,
            }
        if not any(command.startswith(prefix) for prefix in self.allowed_prefixes):
            return {
                "ok": False,
                "skipped": True,
                "reason": "blocked by allow-list",
                "command": command,
            }

        completed = subprocess.run(
            command,
            shell=True,
            timeout=120,
            capture_output=True,
            text=True,
        )
        return {
            "ok": completed.returncode == 0,
            "skipped": False,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


_llm: LLMGateway | None = None
_runner = SafeCommandRunner()


def get_llm() -> LLMGateway:
    global _llm
    if _llm is None:
        _llm = LLMGateway()
    return _llm


def llm_json_module(
    *,
    name: str,
    inputs: list[str],
    outputs: list[str],
    prompt: str,
    optional_inputs: list[str] | None = None,
    temperature: float = 0.1,
    description: str = "",
) -> AgentModule:
    def run(ctx: ModuleContext) -> dict[str, Any]:
        ctx.emit("llm_request", {"keys": list(ctx.inputs)})
        return get_llm().ask_json(prompt, ctx.inputs, temperature=temperature)

    return AgentModule(
        name=name,
        inputs=inputs,
        optional_inputs=optional_inputs or [],
        outputs=outputs,
        run=run,
        description=description,
        require_all_outputs=True,
    )


PLANNER_PROMPT = """
你是软件架构师。把任务拆成可执行计划。
只返回 JSON：
{
  "plan": {
    "objective": "...",
    "steps": ["..."],
    "acceptance_criteria": ["..."],
    "risks": ["..."]
  }
}
""".strip()

CODER_PROMPT = """
你是 Python 工程师。根据 task、plan 和可选 review/verdict/test_report 编写或修改代码。
如果包含 review 或 verdict，说明上一轮需要返工。
只返回 JSON：
{
  "draft_code": "...",
  "coding_notes": ["..."]
}
""".strip()

REVIEWER_PROMPT = """
你是代码审查员。检查 draft_code 是否满足 task 和 plan。
只返回 JSON：
{
  "review": {
    "approved": true,
    "issues": [{"severity": "high|medium|low", "detail": "..."}],
    "suggestions": ["..."]
  }
}
""".strip()

TESTER_PROMPT = """
你是测试工程师。判断是否需要运行一个安全测试命令。
允许的命令前缀只有：python -m pytest, pytest, python -m unittest。
只返回 JSON：
{
  "test_plan": {
    "run_tests": false,
    "command": "",
    "rationale": "..."
  }
}
""".strip()

ARBITER_PROMPT = """
你是验收仲裁者。根据 review 和 test_report 决定是否验收。
只返回 JSON：
{
  "verdict": {
    "accept": true,
    "reason": "...",
    "remaining_risks": ["..."]
  }
}
""".strip()


def tester_runtime(ctx: ModuleContext) -> dict[str, Any]:
    plan_result = get_llm().ask_json(TESTER_PROMPT, ctx.inputs, temperature=0.0)
    test_plan = plan_result.get("test_plan", {})
    if bool(test_plan.get("run_tests", False)):
        report = _runner.run(str(test_plan.get("command", "")))
    else:
        report = {
            "ok": True,
            "skipped": True,
            "reason": "tester skipped execution",
            "command": str(test_plan.get("command", "")),
        }
    report["test_plan"] = test_plan
    return {"test_report": report}


def route_after_arbiter(state: RunState) -> str:
    verdict = state.data.get("verdict", {})
    if bool(verdict.get("accept", False)):
        state.answer = state.data.get("draft_code")
        return END
    return "coder"


def build_code_review_graph() -> AgentGraph:
    graph = AgentGraph()

    graph.add_module(
        llm_json_module(
            name="planner",
            inputs=["task"],
            outputs=["plan"],
            prompt=PLANNER_PROMPT,
            description="把任务变成计划",
        )
    )
    graph.add_module(
        llm_json_module(
            name="coder",
            inputs=["task", "plan"],
            optional_inputs=["review", "verdict", "test_report"],
            outputs=["draft_code", "coding_notes"],
            prompt=CODER_PROMPT,
            temperature=0.2,
            description="根据计划和反馈写代码",
        )
    )
    graph.add_module(
        llm_json_module(
            name="reviewer",
            inputs=["task", "plan", "draft_code"],
            outputs=["review"],
            prompt=REVIEWER_PROMPT,
            description="审查代码",
        )
    )
    graph.add_module(
        AgentModule(
            name="tester",
            inputs=["task", "plan", "draft_code", "review"],
            outputs=["test_report"],
            run=tester_runtime,
            description="规划并可选执行测试",
            require_all_outputs=True,
        )
    )
    graph.add_module(
        llm_json_module(
            name="arbiter",
            inputs=["review", "test_report"],
            outputs=["verdict"],
            prompt=ARBITER_PROMPT,
            temperature=0.0,
            description="决定是否验收",
        )
    )

    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_edge("reviewer", "tester")
    graph.add_edge("tester", "arbiter")
    graph.add_conditional_edges(
        "arbiter",
        route_after_arbiter,
        {
            "coder": "coder",
            END: END,
        },
    )
    graph.set_start("planner")
    return graph


def build_mock_graph() -> AgentGraph:
    def plan(ctx: ModuleContext) -> dict[str, Any]:
        return {
            "plan": {
                "objective": ctx.inputs["task"],
                "steps": ["write code", "review code", "test code"],
            }
        }

    def code(ctx: ModuleContext) -> dict[str, Any]:
        review = ctx.inputs.get("review")
        if review and review.get("issues"):
            code_text = "def add(a, b):\n    return a + b\n"
            notes = ["fixed after review"]
        else:
            code_text = "def add(a, b):\n    return a - b\n"
            notes = ["first draft intentionally imperfect"]
        return {"draft_code": code_text, "coding_notes": notes}

    def review(ctx: ModuleContext) -> dict[str, Any]:
        approved = "return a + b" in ctx.inputs["draft_code"]
        issues = [] if approved else [{"severity": "high", "detail": "uses subtraction"}]
        return {
            "review": {
                "approved": approved,
                "issues": issues,
                "suggestions": ["return a + b"] if issues else [],
            }
        }

    def test(ctx: ModuleContext) -> dict[str, Any]:
        ok = "return a + b" in ctx.inputs["draft_code"]
        return {
            "test_report": {
                "ok": ok,
                "skipped": False,
                "reason": "mock assertion passed" if ok else "mock assertion failed",
            }
        }

    def arbitrate(ctx: ModuleContext) -> dict[str, Any]:
        review_passed = bool(ctx.inputs["review"].get("approved", False))
        tests_passed = bool(ctx.inputs["test_report"].get("ok", False))
        return {
            "verdict": {
                "accept": review_passed and tests_passed,
                "reason": "accepted" if review_passed and tests_passed else "needs revision",
            }
        }

    graph = AgentGraph()
    graph.add_module(AgentModule("planner", ["task"], ["plan"], plan))
    graph.add_module(
        AgentModule(
            "coder",
            ["task", "plan"],
            ["draft_code", "coding_notes"],
            code,
            optional_inputs=["review", "verdict", "test_report"],
            require_all_outputs=True,
        )
    )
    graph.add_module(
        AgentModule("reviewer", ["task", "plan", "draft_code"], ["review"], review)
    )
    graph.add_module(
        AgentModule("tester", ["draft_code", "review"], ["test_report"], test)
    )
    graph.add_module(
        AgentModule("arbiter", ["review", "test_report"], ["verdict"], arbitrate)
    )

    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_edge("reviewer", "tester")
    graph.add_edge("tester", "arbiter")
    graph.add_conditional_edges(
        "arbiter",
        route_after_arbiter,
        {
            "coder": "coder",
            END: END,
        },
    )
    graph.set_start("planner")
    return graph


def run_task(
    task: str,
    *,
    mode: str,
    max_steps: int,
    allow_test_exec: bool = False,
) -> RunState:
    global _runner
    _runner = SafeCommandRunner(allow_execute=allow_test_exec)

    graph = build_mock_graph() if mode == "mock" else build_code_review_graph()
    compiled = graph.compile(initial_keys={"task"})
    state = RunState(data={"task": task})
    return compiled.run(state, max_steps=max_steps, error_policy="raise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular multi-agent graph")
    parser.add_argument(
        "--task",
        default="写一个 Python 函数 add(a, b)，返回两个数之和。",
        help="任务描述",
    )
    parser.add_argument("--mode", choices=["mock", "llm"], default="mock")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--allow-test-exec", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = run_task(
        args.task,
        mode=args.mode,
        max_steps=args.max_steps,
        allow_test_exec=args.allow_test_exec,
    )

    print("\n=== Status ===")
    print(state.status)

    print("\n=== Data ===")
    print(json.dumps(state.data, ensure_ascii=False, indent=2))

    print("\n=== Events ===")
    for event in state.events:
        print(f"{event.step:02d} {event.node} {event.kind} {event.detail}")

    if state.errors:
        print("\n=== Errors ===")
        print(json.dumps(state.errors, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
