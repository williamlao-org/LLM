"""交互式权限裁决:遇到 ask 就把请求打到终端,让人当场拍 y/n/a。

它和 RuleBasedApprovalHandler 是【同一类东西】——都满足 PermissionApprovalHandler
签名 `(PermissionRequest) -> PermissionCheckResult`,都能直接塞进
`PermissionResolver(approval_handler=...)`。区别只在"拍板的方式":一个查配置自动判,
一个问活人。所以执行器/resolver/子 Agent 那套全不用改,这就是把 handler 做成回调的回报。

相比规则式,交互式多出三个新关注点,代码里都会点到:
1. 状态:`a`(本会话总是允许)要被记住 → handler 实例持有一个 set(它第一次有记忆)。
2. 并发:http_request 是 parallel 工具,可能多线程同时触发 ask;两个线程同时 input()
   会把终端搅乱 → 用一把锁把"打印 + 读取"整段串起来。
3. 可测:不能在测试里真等人敲键盘 → 把 input/print 做成可注入参数(同 renderer 回调思路)。

fail-closed:空输入、看不懂的输入、读不到终端(EOF)一律当拒——拿不准就不放行。
"""

from __future__ import annotations

import threading
from typing import Callable

from .resolver import PermissionRequest
from .types import PermissionCheckResult


class InteractiveApprovalHandler:
    """把 ask 抛给终端前的人来裁决,可选记住"本会话总是允许某工具"。"""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        on_remember: Callable[[str], None] | None = None,
    ):
        # 注入 io 而非直接绑死内置 input/print:测试里塞个假的就能驱动整条交互,
        # 不用真去读 stdin。
        self._input = input_fn
        self._output = output_fn
        # "别再问"的落盘钩子:用户选 a 时调它把规则固化(如写回 settings.json)。
        # 做成回调而非在 handler 里直接写文件——handler 不该知道"配置存在哪、什么格式",
        # 那是装配层的事;不接这个钩子时 a 就只在本会话内存里生效。
        self._on_remember = on_remember
        # "总是允许"的记忆:按工具名记。本会话恒在内存里(下次同工具直接放行);
        # 若接了 on_remember,则同时落盘,跨会话也生效。
        # 粒度选工具名而非"工具+具体参数":后者几乎不会复用,工具名级才真正省事。
        # 代价是 a execute_command 等于放行该工具任意命令,所以只在低风险时给 a 选项。
        self._always_allow: set[str] = set()
        # parallel 工具(如 http_request)可能在线程池里并发问;锁保证终端一次只问一题。
        self._lock = threading.Lock()

    def __call__(self, request: PermissionRequest) -> PermissionCheckResult:
        tool_name = request.tool.name

        with self._lock:
            if tool_name in self._always_allow:
                return self._allow(request, f"本会话已记住:总是允许 {tool_name}")

            self._render(request)
            offer_always = self._allow_always_offered(request)
            answer = self._ask(offer_always)

            if answer == "a" and offer_always:
                self._always_allow.add(tool_name)
                if self._on_remember is not None:
                    # 落盘成一条 allow 规则;工具名级记忆 → 规则就是裸工具名。
                    self._on_remember(tool_name)
                scope = "并已写入配置(跨会话生效)" if self._on_remember else "本会话内"
                return self._allow(request, f"用户批准,记住总是允许 {tool_name}({scope})")
            if answer == "y":
                return self._allow(request, "用户批准本次执行")
            return self._deny(request, f"用户拒绝(输入 {answer!r})")

    # ── 终端呈现与读取 ────────────────────────────────────────────────────────

    def _render(self, request: PermissionRequest) -> None:
        flags = ", ".join(request.check.risk_flags) or "无"
        subject = _subject_line(request.arguments)
        self._output("")
        self._output("⚠️  需要权限确认")
        self._output(f"  工具: {request.tool.name}")
        if subject:
            self._output(f"  参数: {subject}")
        self._output(f"  风险: {flags}")
        self._output(f"  说明: {request.check.reason}")

    def _ask(self, offer_always: bool) -> str:
        choices = "[y]允许一次 / [n]拒绝"
        if offer_always:
            choices += " / [a]本会话总是允许该工具"
        try:
            return self._input(f"  允许执行? {choices}: ").strip().lower()
        except EOFError:
            # 没有真正的交互终端(如管道/CI)→ 当拒,绝不静默放行。
            return "n"

    @staticmethod
    def _allow_always_offered(request: PermissionRequest) -> bool:
        """高风险副作用不提供"总是允许":别让一次回车把整类危险操作永久放行。"""
        heavy = {"executes_shell", "deletes_files", "modifies_git_state"}
        return not (heavy & set(request.check.risk_flags))

    # ── 判定构造 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _allow(request: PermissionRequest, reason: str) -> PermissionCheckResult:
        return PermissionCheckResult(
            "allow", reason, request.check.risk_flags, source="user"
        )

    @staticmethod
    def _deny(request: PermissionRequest, reason: str) -> PermissionCheckResult:
        return PermissionCheckResult(
            "deny", reason, request.check.risk_flags, source="user"
        )


def _subject_line(arguments: dict) -> str:
    for key in ("command", "file", "directory", "url"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return f"{key}={value}"
    return ""
