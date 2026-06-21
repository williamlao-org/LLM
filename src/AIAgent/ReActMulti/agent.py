import json
from collections.abc import Sequence

from openai.types.chat import ChatCompletionMessageParam

from .context import ContextCompactor
from .events import ContentDelta, ContentDone, ReasoningDelta, UsageEvent
from .executor import ToolExecutor
from .permission import PermissionResolver
from .llm import LLMClient
from .prompt import build_system_prompt
from .renderer import Renderer
from .session import SessionState, UsageRecord
from .protocol import TurnAbort, parse_turn
from .tools.base import Tool
from .tools.command_tools import get_cwd
from .util import build_tool_results_message


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        session_state: SessionState,
        renderer: Renderer,
        tool_timeout: float = 30,
        context_watermark: float = 0.75,
        keep_recent_tool_results: int = 3,
        max_consecutive_invalid: int = 3,
        permission_resolver: PermissionResolver | None = None,
    ):
        self.llm = llm
        self.session_state = session_state
        self.renderer = renderer
        # 权限裁决器可由装配层注入(承载规则/模式配置),并沿主→子 Agent 共用同一份;
        # 不传则 ToolExecutor 自建一个无 handler 的默认 resolver(ask 一律 fail-closed)。
        self._permission_resolver = permission_resolver
        # 连续 N 轮解析失败就止损:再喂回去也大概率是同样的废 JSON,
        # 与其烧光 max_steps,不如如实标 failed 退出。中间成功一次即清零。
        self.max_consecutive_invalid = max_consecutive_invalid

        # 上下文压缩独立成 collaborator:Agent 只负责在主循环里喊它一声 +
        # 折叠后从 running total 扣减省下的 token,折叠逻辑本身归 ContextCompactor。
        self.compactor = ContextCompactor(
            renderer,
            context_watermark=context_watermark,
            keep_recent_tool_results=keep_recent_tool_results,
        )

        if not self.session_state.message_records:
            msg: ChatCompletionMessageParam = {
                "role": "system",
                "content": build_system_prompt(
                    json.dumps(
                        [tool.to_dict() for tool in tools], ensure_ascii=False, indent=2
                    )
                ),
            }
            self.session_state.append_message(msg)

        # 工具调度执行独立成 collaborator:Agent 只在主循环里把这一轮的 tool_calls
        # 交给它,查表/钳超时/并发分流/异常兜底都归 ToolExecutor。
        # registry 存整个 Tool:执行要 call,调度要 concurrency 等元数据。
        self.executor = ToolExecutor(
            {tool.name: tool for tool in tools},
            tool_timeout=tool_timeout,
            on_command_output=renderer.on_command_output,
            permission_resolver=permission_resolver,
            workspace_dir=session_state.workspace_dir,
            cwd_provider=get_cwd,
        )

    @property
    def context_limit(self) -> int | None:
        return self.llm.context_limit

    @property
    def messages(self) -> Sequence[ChatCompletionMessageParam]:
        return self.session_state.messages

    def _compact_context_if_needed(self) -> int:
        """喊 compactor 折叠旧工具结果;折叠后从 running total 扣减省下的 token。

        不再作废锚点:running total 被增量调整(减去折叠省下的),
        下次 usage 回来时自然会精确校准。
        """
        folded_count, token_savings = self.compactor.compact_if_needed(
            self.session_state.message_records,
            self.session_state.context_tokens,
            self.context_limit,
        )
        if token_savings:
            self.session_state.context_tokens -= token_savings
        return folded_count

    def _run_turn(self) -> tuple[str, UsageRecord | None]:
        """跑一轮 LLM 调用：实时渲染事件流，返回拼接好的完整 content。"""

        # 初始化空串:依赖"LLMClient 必以 ContentDone 收尾"的契约,
        # 但契约被破坏时不该炸出莫名其妙的 NameError
        content = ""
        usage_record: UsageRecord | None = None

        for event in self.llm(self.session_state.wire_messages()):
            if isinstance(event, ReasoningDelta):
                self.renderer.on_reasoning_delta(event.piece)
            elif isinstance(event, ContentDelta):
                self.renderer.on_content_delta(event.piece)
            elif isinstance(event, ContentDone):
                content = event.content
            elif isinstance(event, UsageEvent):
                usage_record = UsageRecord.from_usage(event.usage)

                self.renderer.on_usage(
                    usage_record.prompt_tokens,
                    usage_record.completion_tokens,
                    usage_record.total_tokens,
                    self.context_limit,
                )

        return content, usage_record

    def run(self, prompt: str, max_steps: int | None = None) -> str | None:
        """执行任务直到模型给出 final_answer(返回它)或步数耗尽(返回 None)。"""
        max_steps = self.session_state.max_steps if max_steps is None else max_steps
        self.session_state.max_steps = max_steps
        self.session_state.append_message({"role": "user", "content": prompt})

        consecutive_invalid = 0

        for _ in range(max_steps):
            self._compact_context_if_needed()

            # ----- 步骤 1：调用 LLM 推理 -----
            content, usage_record = self._run_turn()
            # assistant 原文不再在这里手动入队:改由 session 的 record_* 方法
            # 在记账的同时落进 wire,wire 与 turn 原子产生、靠稳定 id 关联。

            # ----- 步骤 2：解析 + 校验(协议层) -----
            # parse_turn 把"解析 JSON + 校验形状 + 二选一路由 + 解析 tool_calls"
            # 一次性收口在 protocol 层;形状级错误统一抛 TurnAbort,主循环只管分流。
            try:
                turn = parse_turn(content)
                consecutive_invalid = 0  # 解析成功,连击清零

                if turn.kind == "final":
                    self.renderer.on_final(turn.final_answer)
                    # 更新会话，添加成功回合记录
                    turn_record = self.session_state.record_assistant_turn(
                        assistant_raw=content,
                        parsed=turn.parsed,
                        route="final",
                    )
                    if usage_record is not None:
                        self.session_state.record_usage_for_turn(
                            turn_record, usage_record
                        )

                    self.session_state.mark_completed()
                    return turn.final_answer

                if turn.kind == "tool_calls":
                    # 更新会话，添加成功回合记录
                    turn_record = self.session_state.record_assistant_turn(
                        assistant_raw=content,
                        parsed=turn.parsed,
                        route="tool_calls",
                        tool_calls=turn.tool_calls,
                    )
                    if usage_record is not None:
                        self.session_state.record_usage_for_turn(
                            turn_record, usage_record
                        )

                    outcomes = self.executor.execute(
                        turn.tool_calls,
                        on_call=self.renderer.on_tool_call,
                        on_result=self.renderer.on_tool_result,
                    )

                    for outcome in outcomes:
                        self.session_state.record_tool_execution(
                            call_id=outcome.call.id,
                            result=outcome.result,
                            status=outcome.status,
                        )

                    self.session_state.append_message(
                        build_tool_results_message(
                            [
                                (outcome.call, outcome.result)
                                for outcome in outcomes
                            ]
                        )
                    )

            except TurnAbort as e:
                consecutive_invalid += 1

                # 更新会话，添加失败回合记录
                turn_record = self.session_state.record_invalid_turn(
                    content,
                    f"LLM 输出无法解析或路由: {e}",
                )
                if usage_record is not None:
                    self.session_state.record_usage_for_turn(turn_record, usage_record)

                # 连续失败到阈值就止损:再喂回去多半还是同样的废 JSON。
                if consecutive_invalid >= self.max_consecutive_invalid:
                    self.renderer.on_final(
                        f"连续 {consecutive_invalid} 轮输出无法解析，任务终止。"
                    )
                    self.session_state.mark_failed()
                    return None

                # 没到阈值:把错误喂回模型,给它一次改正的机会
                self.session_state.append_message({
                    "role": "user",
                    "content": json.dumps(
                        {"error": f"LLM 输出无法解析或路由：{e}"},
                        ensure_ascii=False,
                    ),
                })
                continue

        else:
            self.renderer.on_final(
                f"已达到最大步数上限（{max_steps} 步），任务未完成。"
            )
            self.session_state.mark_max_steps()
            return None
