from math import ceil
from .util import (
    llm_json_parser,
    route,
    parse_tool_calls,
    build_tool_results_message,
    TurnAbort,
)
from .prompt import SYSTEM_PROMPT
from .renderer import Renderer
from .events import ReasoningDelta, ContentDelta, ContentDone
from .llm import LLMClient
from .tools.base import Tool, ToolCall, ToolResult
from openai.types.chat import ChatCompletionMessageParam

import json
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        renderer: Renderer,
        tool_timeout: float = 30,
    ):
        self.llm = llm
        self.tools = tools
        self.renderer = renderer
        self.tool_timeout = tool_timeout

        self.messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools=json.dumps(
                        [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
                    )
                ),
            }
        ]
        self.tool_registry = {tool.name: tool.func for tool in tools}

    def _run_turn(self) -> str:
        """跑一轮 LLM 调用：实时渲染事件流，返回拼接好的完整 content。"""

        # 初始化空串:依赖"LLMClient 必以 ContentDone 收尾"的契约,
        # 但契约被破坏时不该炸出莫名其妙的 NameError
        content = ""
        for event in self.llm(self.messages):
            if isinstance(event, ReasoningDelta):
                self.renderer.on_reasoning_delta(event.piece)
            elif isinstance(event, ContentDelta):
                self.renderer.on_content_delta(event.piece)
            elif isinstance(event, ContentDone):
                content = event.content
        return content

    def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """查找并执行【单个】工具，返回标准化 tool_result。"""
        if tool_call.error:
            return ToolResult.fail(tool_call.error)

        tool_name = tool_call.name
        tool_arguments = tool_call.arguments

        tool_fn = self.tool_registry.get(tool_name)
        if tool_fn is None:
            return ToolResult.fail(err=f"Unknown tool: {tool_name}")

        # 内层超时必须 ≤ 外层线程预算:模型可以给工具传很大的 timeout,
        # 不钳制的话外层先掐,工具内部的超时机制(如 execute_command 转后台)永远轮不到登场
        if isinstance(tool_arguments.get("timeout"), (int, float)):
            tool_arguments["timeout"] = min(
                tool_arguments["timeout"], self.tool_timeout
            )

        try:
            if tool_name == "execute_command":
                tool_result = tool_fn(
                    **tool_arguments, on_output=self.renderer.on_command_output
                )
            else:
                tool_result = tool_fn(**tool_arguments)
        except Exception as e:
            tool_result = ToolResult.fail(f"{type(e).__name__}: {e}")

        return tool_result

    def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        on_call: Callable[[ToolCall], None] | None = None,
        on_result: Callable[[ToolResult], None] | None = None,
    ) -> list[tuple[ToolCall, ToolResult]]:
        """逐个执行工具,返回 (call, result) 列表。

        on_call / on_result 是可选回调,在每个工具执行前/后被喊一声。
        不传则不渲染进度(便于单测);传了就能实时渲染。
        循环的所有权始终在本函数,渲染只是从插槽注入。

        ### Explain
        执行循环想保持可测,但它跑的那个 for 循环又是渲染唯一能"实时插话"的地方。
        矛盾点在于:循环的所有权在执行函数手里,但渲染想在循环的每一步插一脚。
        回调就是执行函数对外开的两个"插槽"——"我每调一个工具前/后会喊一声,
        你想接就接,不接我照跑"。循环归执行函数独有(不重复),渲染从外部注入。
        """
        results: list[tuple[ToolCall, ToolResult]] = []
        for tool_call in tool_calls:
            if on_call:
                on_call(tool_call)
            result = self._execute_tool_call(tool_call)
            if on_result:
                on_result(result)
            results.append((tool_call, result))
        return results

    def execute_tool_calls_parallel(
        self,
        tool_calls: list[ToolCall],
        on_call: Callable[[ToolCall], None] | None = None,
        on_result: Callable[[ToolResult], None] | None = None,
        max_workers: int = 8,
    ) -> list[tuple[ToolCall, ToolResult]]:
        """并行执行工具,返回 (call, result) 列表,顺序与输入 tool_calls 一致。

        为什么用线程池而非进程池:工具(读写文件/跑命令/查网络)都是 I/O 密集,
        等待时释放 GIL,线程足够;且进程池要求参数能 pickle,得不偿失。

        并发但保序:每个 future 记住自己的原始下标,完工后按下标回填,
        所以无论谁先跑完,返回顺序恒等于输入顺序——这是 OpenAI/Anthropic 的做法,
        因为结果要按 tool_call.id 一一对应喂回给 LLM,顺序错模型就对不上号。

        on_call / on_result 都在主线程跑(提交前调 on_call;as_completed 里
        按完成顺序调 on_result)。例外是 execute_command 的 on_output——
        它从工具自己的 reader 线程触发,渲染器若做复杂状态更新需自行考虑并发。
        _execute_tool_call 已把异常吞成 ToolResult.fail,单个工具失败
        被隔离成一条错误结果,不会搞崩整轮;超时同理,超时的调用以
        fail 占位留在结果里,绝不"蒸发"(模型靠 id 对账,少一条都不行)。
        """
        if on_call:
            for tool_call in tool_calls:
                on_call(tool_call)

        # 预留与输入等长的槽位,完工后按原始下标回填 → 保序
        slots: list[tuple[ToolCall, ToolResult] | None] = [None] * len(tool_calls)
        budget = self.tool_timeout * ceil(len(tool_calls) / max_workers)

        pool = ThreadPoolExecutor(max_workers=max_workers)
        fut_to_idx = {
            pool.submit(self._execute_tool_call, tc): i
            for i, tc in enumerate(tool_calls)
        }

        try:
            # 谁先跑完谁先渲染(实时),但写回固定槽位(保序)
            for fut in as_completed(fut_to_idx, timeout=budget):
                idx = fut_to_idx[fut]
                result = fut.result()  # _execute_tool_call 不抛异常,恒拿到 ToolResult
                if on_result:
                    on_result(result)
                slots[idx] = (tool_calls[idx], result)

        except (TimeoutError, FuturesTimeoutError):
            for idx, slot in enumerate(slots):
                if slot is None:
                    result = ToolResult.fail(
                        f"timeout: 超过 {budget}s 未完成(工具可能仍在后台运行)"
                    )
                    if on_result:
                        on_result(result)
                    slots[idx] = (tool_calls[idx], result)

        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        return [slot for slot in slots if slot is not None]

    def run(self, prompt: str, max_steps: int = 25) -> str | None:
        """执行任务直到模型给出 final_answer(返回它)或步数耗尽(返回 None)。"""
        self.messages.append({"role": "user", "content": prompt})

        for _ in range(max_steps):
            # ----- 步骤 1：调用 LLM 推理 -----
            content = self._run_turn()
            self.messages.append({"role": "assistant", "content": content})

            # ----- 步骤 2：解析 + 路由 -----
            try:
                content_json = llm_json_parser(content)
                kind, payload = route(content_json)

                if kind == "final":
                    self.renderer.on_final(payload)
                    return payload

                tool_calls = parse_tool_calls(payload)

                results = self.execute_tool_calls_parallel(
                    tool_calls,
                    on_call=self.renderer.on_tool_call,
                    on_result=self.renderer.on_tool_result,
                )
                self.messages.append(build_tool_results_message(results))

            except TurnAbort as e:
                msg: ChatCompletionMessageParam = {
                    "role": "user",
                    "content": json.dumps(
                        {"error": f"LLM 输出无法解析或路由：{e}"},
                        ensure_ascii=False,
                    ),
                }
                self.messages.append(msg)
                continue

        else:
            self.renderer.on_final(
                f"已达到最大步数上限（{max_steps} 步），任务未完成。"
            )
            return None
