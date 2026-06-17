"""
展示层：决定"事件如何呈现给人看"。

主循环只跟 Renderer 接口打交道，不关心具体怎么展示。
这样同一套 Agent 逻辑可以配不同的 Renderer：

    ConsoleRenderer  → 终端实时输出（当前默认）
    SilentRenderer   → 什么都不打（跑测试 / 批量任务）
    （未来）JSONRenderer / WebRenderer → 把事件推给前端

传输层负责"把实时内容放进事件"，展示层负责"自己去取并呈现"，
两端职责清晰、互不依赖。
"""

import json
import sys
from abc import ABC, abstractmethod
from typing import Any

from .tools.base import ToolCall, ToolResult

# 在 Windows 经典控制台里启用 ANSI 转义支持（Windows Terminal 默认已支持，
# 这段能让旧控制台也正确渲染颜色）。开启失败时静默忽略，不影响主流程。
if sys.platform == "win32":
    try:
        import ctypes

        _STD_OUTPUT_HANDLE = -11
        _ENABLE_VT_PROCESSING = 0x0007  # PROCESSED | WRAP | VIRTUAL_TERMINAL
        _kernel32 = ctypes.windll.kernel32
        _kernel32.SetConsoleMode(
            _kernel32.GetStdHandle(_STD_OUTPUT_HANDLE), _ENABLE_VT_PROCESSING
        )
    except Exception:
        pass


class _Style:
    """ANSI 样式常量。用语义命名而非颜色名，方便统一调整配色。"""

    DIM = "\033[2m"  # 暗淡：次要信息（参数、数据）
    GRAY = "\033[90m"  # 灰：思考过程（弱化，区别于正式回答）
    CYAN = "\033[36m"  # 青：回答标题
    GREEN = "\033[32m"  # 绿：成功 / 最终答案
    RED = "\033[31m"  # 红：失败
    YELLOW = "\033[33m"  # 黄：工具调用
    ORANGE = "\033[38;5;208m"  # 橙：用量 / 预算提示
    BOLD = "\033[1m"
    RESET = "\033[0m"


class Renderer(ABC):
    """展示层接口。主循环按 ReAct 的生命周期回调这些方法。"""

    @abstractmethod
    def on_reasoning_delta(self, piece: str) -> None: ...
    @abstractmethod
    def on_content_delta(self, piece: str) -> None: ...
    @abstractmethod
    def on_tool_call(self, tool_call: ToolCall | dict) -> None: ...
    @abstractmethod
    def on_tool_result(self, tool_result: "ToolResult | dict") -> None: ...
    @abstractmethod
    def on_final(self, answer: Any) -> None: ...

    def on_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        context_limit: int | None,
    ) -> None:
        """本轮 token 用量回调(服务端精确值)。默认不输出，子类按需覆盖。"""

    def on_context_compact(
        self,
        folded_count: int,
        prompt_tokens: int | None,
        context_limit: int | None,
        context_watermark: float,
    ) -> None:
        """上下文压缩回调。默认不输出，子类按需覆盖。"""

    def on_command_output(self, line: str) -> None:
        """命令流式输出回调。默认不输出，子类按需覆盖。"""


class ConsoleRenderer(Renderer):
    """终端渲染器：用颜色 + 图标对"思考 / 回答 / 工具 / 结论"做视觉分层。

    内部用 _phase 记住当前处于哪个流式阶段（reasoning / content / idle），
    只在阶段切换时打印小标题，避免逐 token 重复打标题。
    """

    def __init__(self) -> None:
        self._phase = "idle"  # "idle" | "reasoning" | "content"

    # ----- 流式阶段管理 -----

    def _start_phase(
        self, phase: str, title: str, title_style: str, body_style: str
    ) -> None:
        """进入一个流式阶段：若是新阶段，先收尾上一个，再打标题并开启正文配色。"""
        if self._phase == phase:
            return
        if self._phase in ("reasoning", "content"):
            print(_Style.RESET, end="")  # 关闭上一个阶段的正文配色
        print(f"\n{title_style}{title}{_Style.RESET}")
        print(body_style, end="", flush=True)  # 开启本阶段正文配色
        self._phase = phase

    def _end_stream(self) -> None:
        """结束流式阶段（工具调用 / 最终答案前调用），复位颜色与状态。"""
        if self._phase in ("reasoning", "content"):
            print(_Style.RESET, end="", flush=True)
        self._phase = "idle"

    # ----- Renderer 接口实现 -----

    def on_reasoning_delta(self, piece: str) -> None:
        self._start_phase(
            "reasoning", "💭 思考", _Style.GRAY + _Style.BOLD, _Style.GRAY
        )
        print(piece, end="", flush=True)

    def on_content_delta(self, piece: str) -> None:
        self._start_phase("content", "💬 回答", _Style.CYAN + _Style.BOLD, "")
        print(piece, end="", flush=True)

    def on_tool_call(self, tool_call) -> None:
        self._end_stream()
        name = tool_call.name
        print(f"\n{_Style.YELLOW}{_Style.BOLD}🔧 调用工具 › {name}{_Style.RESET}")
        arguments = tool_call.arguments
        if arguments:
            body = json.dumps(arguments, ensure_ascii=False, indent=2)
            print(f"{_Style.DIM}{body}{_Style.RESET}", flush=True)
        if name == "execute_command":
            print(f"{_Style.DIM}── 输出 ──{_Style.RESET}", flush=True)

    def on_command_output(self, line: str) -> None:
        print(f"{_Style.DIM}{line}{_Style.RESET}", end="", flush=True)

    def on_tool_result(self, tool_result) -> None:
        self._end_stream()
        if hasattr(tool_result, "to_dict"):
            tool_result = tool_result.to_dict()

        if tool_result.get("ok"):
            print(f"{_Style.GREEN}{_Style.BOLD}✅ 工具结果{_Style.RESET}")
            data = tool_result.get("data")
            body = json.dumps(data, ensure_ascii=False, indent=2)
            print(f"{_Style.DIM}{body}{_Style.RESET}\n", flush=True)
        else:
            print(f"{_Style.RED}{_Style.BOLD}❌ 工具失败{_Style.RESET}")
            print(f"{_Style.RED}{tool_result.get('err')}{_Style.RESET}\n", flush=True)

    def on_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        context_limit: int | None,
    ) -> None:
        self._end_stream()
        input_tokens = prompt_tokens if prompt_tokens is not None else "?"
        output_tokens = completion_tokens if completion_tokens is not None else "?"
        request_total = total_tokens if total_tokens is not None else "?"

        # 上下文水位 = P+C:模型回复(C)已入队 messages,下次一定是输入的一部分,
        # 所以当前上下文的精确大小 = 本轮输入(P) + 本轮输出(C),两者都是服务端真值。
        if prompt_tokens is not None and completion_tokens is not None and context_limit:
            context_size = prompt_tokens + completion_tokens
            context_usage = f"{context_size} / {context_limit}"
            context_percent = f" ({context_size / context_limit:.1%})"
        else:
            context_usage = f"{input_tokens} / ?"
            context_percent = ""

        print(
            f"\n{_Style.ORANGE}{_Style.BOLD}tokens{_Style.RESET} "
            f"{_Style.ORANGE}本轮输入 {input_tokens} / "
            f"本轮输出 {output_tokens} / "
            f"本轮总计 {request_total} / "
            f"\n{_Style.ORANGE}{_Style.BOLD}context{_Style.RESET} "
            f"{_Style.ORANGE}上下文水位 {context_usage}{context_percent}"
            f"{_Style.RESET}",
            flush=True,
        )

    def on_context_compact(
        self,
        folded_count: int,
        prompt_tokens: int | None,
        context_limit: int | None,
        context_watermark: float,
    ) -> None:
        self._end_stream()
        if prompt_tokens is not None and context_limit:
            context_usage = f"{prompt_tokens} / {context_limit}"
            context_percent = f" ({prompt_tokens / context_limit:.1%})"
        else:
            context_usage = "? / ?"
            context_percent = ""
        watermark = f"{context_watermark:.0%}"

        if folded_count > 0:
            message = f"已折叠 {folded_count} 条旧工具结果"
            style = _Style.ORANGE
        else:
            message = "上下文已超水位,但暂无可折叠旧工具结果"
            style = _Style.YELLOW

        print(
            f"\n{style}{_Style.BOLD}context compact{_Style.RESET} "
            f"{style}{message} / 水位 {context_usage}{context_percent} / "
            f"阈值 {watermark}{_Style.RESET}",
            flush=True,
        )

    def on_final(self, answer) -> None:
        self._end_stream()
        text = (
            answer
            if isinstance(answer, str)
            else json.dumps(answer, ensure_ascii=False, indent=2)
        )
        print(f"\n{_Style.GREEN}{_Style.BOLD}🎯 最终答案{_Style.RESET}")
        print(f"{_Style.GREEN}{text}{_Style.RESET}\n", flush=True)


class SilentRenderer(Renderer):
    """静默渲染器：什么都不输出。用于测试或批量任务。"""

    def on_reasoning_delta(self, piece: str) -> None: ...
    def on_content_delta(self, piece: str) -> None: ...
    def on_tool_call(self, tool_call) -> None: ...
    def on_tool_result(self, tool_result) -> None: ...
    def on_final(self, answer) -> None: ...
