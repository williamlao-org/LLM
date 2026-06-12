from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from openai import OpenAI


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.64.0.4:8080/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "1234567890")
MODEL = os.getenv("OPENAI_MODEL", "Qwen3.5-27B-Opus4.6-Q4_K_M.gguf")


PLANNER_PROMPT = """
You are a senior software architect.
Break task into realistic implementation milestones.
Return JSON with keys:
- objective: string
- milestones: string[]
- acceptance_criteria: string[]
- risks: string[]
- candidate_test_commands: string[]
""".strip()

CODER_PROMPT = """
You are a principal Python engineer.
Implement production-grade code according to the plan and feedback.
Return JSON with keys:
- summary: string
- code: string
- notes: string[]
""".strip()

REVIEWER_PROMPT = """
You are a strict reviewer.
Review candidate code and return JSON with keys:
- approved: boolean
- issues: list of objects {severity: "high|medium|low", detail: string}
- suggestions: string[]
- confidence: number
""".strip()

TESTER_PROMPT = """
You are a test engineer.
Decide whether to run one safe test command.
Return JSON with keys:
- run_tests: boolean
- command: string
- rationale: string
""".strip()

ARBITER_PROMPT = """
You are the release arbiter.
Decide accept/reject based on aggregated reviews and test report.
Return JSON with keys:
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
    coding_result: dict[str, Any]
    aggregated_review: dict[str, Any]
    individual_reviews: list[dict[str, Any]]
    test_report: dict[str, Any]
    verdict: dict[str, Any]


@dataclass
class Board:
    task: TaskSpec
    plan: dict[str, Any] = field(default_factory=dict)
    rounds: list[RoundRecord] = field(default_factory=list)
    final_code: str = ""


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
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def ask_json(
        self, system_prompt: str, payload: dict[str, Any], temperature: float
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Reply with valid JSON only.\nInput:\n"
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


class WorkspaceMemory:
    def __init__(self, file_path: str) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def load_recent(self, limit: int = 8) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-limit:]
        results: list[dict[str, Any]] = []
        for line in recent:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results


class SafeCommandRunner:
    def __init__(
        self, allow_execute: bool = True, allowed_prefixes: list[str] | None = None
    ) -> None:
        self.allow_execute = allow_execute
        self.allowed_prefixes = allowed_prefixes or [
            "pytest",
            "python -m pytest",
            "uv run pytest",
            "python -m unittest",
        ]

    def _is_allowed(self, command: str) -> bool:
        cmd = command.strip()
        return any(cmd.startswith(prefix) for prefix in self.allowed_prefixes)

    def run(self, command: str, timeout_seconds: int = 120) -> dict[str, Any]:
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
                "reason": "execution disabled by cli flag",
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


class MultiAgentEngineV2:
    def __init__(
        self,
        llm: LLMGateway,
        runner: SafeCommandRunner,
        memory: WorkspaceMemory,
        reviewer_count: int = 2,
    ) -> None:
        self.llm = llm
        self.runner = runner
        self.memory = memory
        self.reviewer_count = max(1, reviewer_count)

    def _log(self, stage: str, payload: dict[str, Any]) -> None:
        self.memory.append_event(
            {
                "ts": int(time.time()),
                "stage": stage,
                "payload": payload,
            }
        )

    def build_plan(self, task: TaskSpec) -> dict[str, Any]:
        recent_memory = self.memory.load_recent(limit=5)
        plan = self.llm.ask_json(
            PLANNER_PROMPT,
            {
                "task": asdict(task),
                "recent_memory": recent_memory,
            },
            temperature=0.1,
        )
        self._log("plan", plan)
        return plan

    def code(
        self, task: TaskSpec, plan: dict[str, Any], history: list[RoundRecord]
    ) -> dict[str, Any]:
        history_view = [
            {
                "round": r.round_index,
                "aggregated_review": r.aggregated_review,
                "verdict": r.verdict,
                "test_result": {
                    "ok": r.test_report.get("ok"),
                    "reason": r.test_report.get("reason"),
                },
            }
            for r in history
        ]

        coding = self.llm.ask_json(
            CODER_PROMPT,
            {
                "task": asdict(task),
                "plan": plan,
                "history": history_view,
            },
            temperature=0.2,
        )
        self._log("coding", {"summary": coding.get("summary", "")})
        return coding

    def _single_review(
        self, reviewer_id: int, task: TaskSpec, plan: dict[str, Any], code_text: str
    ) -> dict[str, Any]:
        review = self.llm.ask_json(
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
        self._log(f"reviewer_{reviewer_id}", review)
        return review

    def review(
        self, task: TaskSpec, plan: dict[str, Any], code_text: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        reviews = [
            self._single_review(i + 1, task, plan, code_text)
            for i in range(self.reviewer_count)
        ]

        all_issues: list[dict[str, Any]] = []
        all_suggestions: list[str] = []
        approvals = 0
        for review in reviews:
            if bool(review.get("approved", False)):
                approvals += 1
            for issue in review.get("issues", []):
                if isinstance(issue, dict):
                    all_issues.append(issue)
            for suggestion in review.get("suggestions", []):
                if isinstance(suggestion, str):
                    all_suggestions.append(suggestion)

        high_count = sum(
            1 for x in all_issues if str(x.get("severity", "")).lower() == "high"
        )
        medium_count = sum(
            1 for x in all_issues if str(x.get("severity", "")).lower() == "medium"
        )

        aggregated = {
            "approved": approvals == self.reviewer_count and high_count == 0,
            "approval_ratio": f"{approvals}/{self.reviewer_count}",
            "issues": all_issues,
            "suggestions": all_suggestions,
            "issue_stats": {
                "high": high_count,
                "medium": medium_count,
                "low": len(all_issues) - high_count - medium_count,
            },
        }
        self._log("review_aggregated", aggregated)
        return reviews, aggregated

    def test(
        self,
        task: TaskSpec,
        plan: dict[str, Any],
        code_text: str,
        aggregated_review: dict[str, Any],
    ) -> dict[str, Any]:
        tester = self.llm.ask_json(
            TESTER_PROMPT,
            {
                "task": asdict(task),
                "plan": plan,
                "aggregated_review": aggregated_review,
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
                "reason": "tester skipped execution",
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }

        report["rationale"] = tester.get("rationale", "")
        self._log(
            "test",
            {
                "ok": report.get("ok"),
                "reason": report.get("reason"),
                "command": report.get("command"),
            },
        )
        return report

    def arbitrate(
        self,
        task: TaskSpec,
        plan: dict[str, Any],
        coding_result: dict[str, Any],
        aggregated_review: dict[str, Any],
        test_report: dict[str, Any],
    ) -> dict[str, Any]:
        verdict = self.llm.ask_json(
            ARBITER_PROMPT,
            {
                "task": asdict(task),
                "plan": plan,
                "coding_summary": coding_result.get("summary", ""),
                "aggregated_review": aggregated_review,
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
        verdict.setdefault("accept", False)
        verdict.setdefault("reason", "")
        verdict.setdefault("remaining_risks", [])
        self._log("arbiter", verdict)
        return verdict

    def run(self, task: TaskSpec, max_rounds: int = 4) -> Board:
        board = Board(task=task)
        board.plan = self.build_plan(task)

        print("\n=== Task ===")
        print(task.objective)
        print("\n=== Plan ===")
        print(json.dumps(board.plan, ensure_ascii=False, indent=2))

        for i in range(1, max_rounds + 1):
            print(f"\n=== Round {i}/{max_rounds} ===")

            coding = self.code(task, board.plan, board.rounds)
            code_text = str(coding.get("code", "")).strip() or "# empty code"
            print("[Coder summary]", coding.get("summary", ""))
            print("[Draft preview]")
            print(code_text[:1200])

            individual_reviews, aggregated_review = self.review(
                task, board.plan, code_text
            )
            print("[Aggregated review]")
            print(json.dumps(aggregated_review, ensure_ascii=False, indent=2))

            test_report = self.test(task, board.plan, code_text, aggregated_review)
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

            verdict = self.arbitrate(
                task, board.plan, coding, aggregated_review, test_report
            )
            print("[Arbiter verdict]")
            print(json.dumps(verdict, ensure_ascii=False, indent=2))

            board.rounds.append(
                RoundRecord(
                    round_index=i,
                    coding_result=coding,
                    aggregated_review=aggregated_review,
                    individual_reviews=individual_reviews,
                    test_report=test_report,
                    verdict=verdict,
                )
            )

            if bool(verdict.get("accept", False)):
                board.final_code = code_text
                print("\nAccepted by arbiter.")
                return board

        board.final_code = (
            str(board.rounds[-1].coding_result.get("code", "")) if board.rounds else ""
        )
        print("\nMax rounds reached. Returning last draft.")
        return board


def build_default_task(task_text: str) -> TaskSpec:
    return TaskSpec(
        objective=task_text,
        constraints=[
            "Prefer robust edge-case handling",
            "Code should be readable and maintainable",
            "Include minimal tests when appropriate",
        ],
        deliverables=["production-ready python code", "review-ready rationale"],
        context=["This output may be used by another automation step"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realistic Multi-Agent Engine V2")
    parser.add_argument(
        "--task",
        type=str,
        default="Write a robust Python function fib(n) that returns the n-th Fibonacci number.",
        help="Task objective",
    )
    parser.add_argument(
        "--rounds", type=int, default=4, help="Max orchestration rounds"
    )
    parser.add_argument(
        "--reviewers", type=int, default=2, help="Number of reviewer agents"
    )
    parser.add_argument(
        "--memory-file",
        type=str,
        default="outputs/multi_agent/memory_events.jsonl",
        help="Path for persistent event memory",
    )
    parser.add_argument(
        "--export-final",
        type=str,
        default="",
        help="Optional file path to save final code",
    )
    parser.add_argument(
        "--no-test-exec",
        action="store_true",
        help="Disable actual test command execution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task = build_default_task(args.task)
    llm = LLMGateway()
    memory = WorkspaceMemory(args.memory_file)
    runner = SafeCommandRunner(allow_execute=not args.no_test_exec)

    engine = MultiAgentEngineV2(
        llm=llm,
        runner=runner,
        memory=memory,
        reviewer_count=args.reviewers,
    )

    board = engine.run(task=task, max_rounds=max(1, args.rounds))

    print("\n=== Final Code ===")
    print(board.final_code)

    if args.export_final:
        output_path = Path(args.export_final)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(board.final_code, encoding="utf-8")
        print(f"\nSaved final code to: {output_path}")


if __name__ == "__main__":
    main()
