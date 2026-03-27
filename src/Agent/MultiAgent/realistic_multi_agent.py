from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.64.0.4:8080/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "1234567890")
MODEL = os.getenv("OPENAI_MODEL", "Qwen3.5-27B-Opus4.6-Q4_K_M.gguf")


PLANNER_PROMPT = """
You are a senior software architect.
Break the user task into a clear execution plan.
Return valid JSON only with keys:
- objective: string
- steps: string[]
- acceptance_criteria: string[]
- risks: string[]
- suggested_test_commands: string[]
""".strip()

CODER_PROMPT = """
You are a principal Python engineer.
Implement code according to the plan and latest feedback.
Return valid JSON only with keys:
- summary: string
- code: string
- notes: string[]
""".strip()

REVIEWER_PROMPT = """
You are a strict code reviewer.
Review the candidate code for correctness, maintainability, style, and edge cases.
Return valid JSON only with keys:
- approved: boolean
- issues: list of objects {severity: "high|medium|low", detail: string}
- suggestions: string[]
""".strip()

TESTER_PROMPT = """
You are a test engineer.
Decide whether to run tests and provide exactly one safe command if needed.
Return valid JSON only with keys:
- run_tests: boolean
- command: string
- rationale: string
""".strip()

ARBITER_PROMPT = """
You are a release arbiter.
Based on review and test results, decide if current draft can be accepted.
Return valid JSON only with keys:
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


@dataclass
class RoundRecord:
    round_index: int
    draft_code: str
    review: dict[str, Any]
    test_report: dict[str, Any]
    verdict: dict[str, Any]


@dataclass
class Board:
    task: TaskSpec
    plan: dict[str, Any] = field(default_factory=dict)
    rounds: list[RoundRecord] = field(default_factory=list)
    final_code: str = ""


class LLMGateway:
    def __init__(
        self,
        model: str = MODEL,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self, system_prompt: str, payload: dict[str, Any], temperature: float
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Reply with valid JSON only. Input:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ]

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    timeout=self.timeout_seconds,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self.max_retries:
                    time.sleep(attempt * 1.2)

        raise RuntimeError(f"LLM request failed after retries: {last_error}")


class JsonParser:
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
            value = json.loads(content)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        left = content.find("{")
        right = content.rfind("}")
        if left >= 0 and right > left:
            try:
                value = json.loads(content[left : right + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                return {}

        return {}


class SafeCommandRunner:
    def __init__(self, allowed_prefixes: list[str] | None = None) -> None:
        self.allowed_prefixes = allowed_prefixes or [
            "pytest",
            "python -m pytest",
            "uv run pytest",
            "python -m unittest",
        ]

    def is_allowed(self, command: str) -> bool:
        cmd = command.strip()
        return any(cmd.startswith(prefix) for prefix in self.allowed_prefixes)

    def run(self, command: str, timeout_seconds: int = 120) -> dict[str, Any]:
        if not command:
            return {
                "ok": True,
                "skipped": True,
                "reason": "No command proposed",
                "command": "",
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        if not self.is_allowed(command):
            return {
                "ok": False,
                "skipped": True,
                "reason": "Command not allowed by policy",
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
                "reason": "Command timed out",
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "timeout",
            }


class MultiAgentEngine:
    def __init__(
        self, llm: LLMGateway, runner: SafeCommandRunner | None = None
    ) -> None:
        self.llm = llm
        self.runner = runner or SafeCommandRunner()

    def _ask_json(
        self, prompt: str, payload: dict[str, Any], temperature: float
    ) -> dict[str, Any]:
        raw = self.llm.chat(prompt, payload, temperature)
        return JsonParser.parse_object(raw)

    def build_plan(self, task: TaskSpec) -> dict[str, Any]:
        return self._ask_json(
            PLANNER_PROMPT,
            {
                "objective": task.objective,
                "constraints": task.constraints,
                "deliverables": task.deliverables,
                "context": task.context,
            },
            temperature=0.1,
        )

    def code(
        self, task: TaskSpec, plan: dict[str, Any], history: list[RoundRecord]
    ) -> dict[str, Any]:
        previous_feedback = [
            {
                "round": r.round_index,
                "review": r.review,
                "test_report": {
                    "ok": r.test_report.get("ok"),
                    "reason": r.test_report.get("reason"),
                },
                "verdict": r.verdict,
            }
            for r in history
        ]

        return self._ask_json(
            CODER_PROMPT,
            {
                "task": task.objective,
                "constraints": task.constraints,
                "plan": plan,
                "previous_feedback": previous_feedback,
            },
            temperature=0.2,
        )

    def review(
        self, task: TaskSpec, plan: dict[str, Any], code_text: str
    ) -> dict[str, Any]:
        return self._ask_json(
            REVIEWER_PROMPT,
            {
                "task": task.objective,
                "constraints": task.constraints,
                "plan": plan,
                "candidate_code": code_text,
            },
            temperature=0.1,
        )

    def test(
        self,
        task: TaskSpec,
        plan: dict[str, Any],
        code_text: str,
        review: dict[str, Any],
    ) -> dict[str, Any]:
        tester = self._ask_json(
            TESTER_PROMPT,
            {
                "task": task.objective,
                "plan": plan,
                "review": review,
                "candidate_code": code_text,
            },
            temperature=0.0,
        )

        run_tests = bool(tester.get("run_tests", False))
        command = str(tester.get("command", "")).strip()

        if run_tests:
            report = self.runner.run(command)
        else:
            report = {
                "ok": True,
                "skipped": True,
                "reason": "Tester decided not to run command",
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        report["rationale"] = tester.get("rationale", "")
        return report

    def arbitrate(
        self,
        task: TaskSpec,
        plan: dict[str, Any],
        code_text: str,
        review: dict[str, Any],
        test_report: dict[str, Any],
    ) -> dict[str, Any]:
        return self._ask_json(
            ARBITER_PROMPT,
            {
                "task": task.objective,
                "plan": plan,
                "candidate_code": code_text,
                "review": review,
                "test_report": {
                    "ok": test_report.get("ok"),
                    "skipped": test_report.get("skipped"),
                    "reason": test_report.get("reason"),
                    "command": test_report.get("command"),
                    "returncode": test_report.get("returncode"),
                },
            },
            temperature=0.0,
        )

    def run(self, task: TaskSpec, max_rounds: int = 4) -> Board:
        board = Board(task=task)
        board.plan = self.build_plan(task)

        print("\n=== Task ===")
        print(task.objective)
        print("\n=== Plan ===")
        print(json.dumps(board.plan, ensure_ascii=False, indent=2))

        for i in range(1, max_rounds + 1):
            print(f"\n=== Round {i}/{max_rounds} ===")

            coding_result = self.code(task, board.plan, board.rounds)
            draft_code = str(coding_result.get("code", "")).strip()
            if not draft_code:
                draft_code = "# Empty draft from coder"

            print("[Coder summary]", coding_result.get("summary", ""))
            print("[Draft preview]")
            print(draft_code[:1000])

            review = self.review(task, board.plan, draft_code)
            print("[Review]")
            print(json.dumps(review, ensure_ascii=False, indent=2))

            test_report = self.test(task, board.plan, draft_code, review)
            print("[Test report]")
            print(
                json.dumps(
                    {
                        "ok": test_report.get("ok"),
                        "skipped": test_report.get("skipped"),
                        "reason": test_report.get("reason"),
                        "command": test_report.get("command"),
                        "returncode": test_report.get("returncode"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

            verdict = self.arbitrate(task, board.plan, draft_code, review, test_report)
            print("[Arbiter verdict]")
            print(json.dumps(verdict, ensure_ascii=False, indent=2))

            board.rounds.append(
                RoundRecord(
                    round_index=i,
                    draft_code=draft_code,
                    review=review,
                    test_report=test_report,
                    verdict=verdict,
                )
            )

            if bool(verdict.get("accept", False)):
                board.final_code = draft_code
                print("\nAccepted by arbiter.")
                return board

        board.final_code = board.rounds[-1].draft_code if board.rounds else ""
        print("\nMax rounds reached. Returning last draft.")
        return board


def demo() -> None:
    task = TaskSpec(
        objective="Write a robust Python function fib(n) that returns the n-th Fibonacci number.",
        constraints=[
            "n is int",
            "raise ValueError for n < 0",
            "handle n = 0 and n = 1",
            "time complexity O(n) or better",
            "include at least two tests",
        ],
        deliverables=["production ready python code", "minimal tests"],
        context=["Prefer readability and explicit edge-case handling"],
    )

    engine = MultiAgentEngine(llm=LLMGateway())
    board = engine.run(task, max_rounds=4)

    print("\n=== Final Code ===")
    print(board.final_code)


if __name__ == "__main__":
    demo()
