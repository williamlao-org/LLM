"""可靠性路径的回归测试。

超时/失败路径平时跑不到——不写测试它就等于不存在,第一次线上工具挂死时才爆。
LLMClient 用假参数构造(不发起任何网络请求),整套测试离线可跑。

运行(项目根目录下):
    python -m src.AIAgent.ReActMulti.test
"""

import time

from .agent import Agent
from .llm import LLMClient
from .renderer import SilentRenderer
from .tools.base import Tool, ToolCall, ToolResult


def _make_agent(tools: list[Tool], tool_timeout: float) -> Agent:
    llm = LLMClient(base_url="http://x", api_key="sk-x", model="m")
    return Agent(llm, tools, SilentRenderer(), tool_timeout=tool_timeout)


def test_parallel_timeout_fills_fail():
    """超时的调用必须以 fail 占位留在结果里,不能蒸发,也不能炸穿。"""

    def fast():
        return ToolResult.success("fast done")

    def slow():
        time.sleep(3)
        return ToolResult.success("slow done")

    agent = _make_agent(
        [Tool("fast", "", {}, fast), Tool("slow", "", {}, slow)],
        tool_timeout=0.5,
    )
    calls = [ToolCall("fast", {}, "c1"), ToolCall("slow", {}, "c2")]

    t0 = time.time()
    results = agent.execute_tool_calls_parallel(calls)
    elapsed = time.time() - t0

    assert len(results) == len(calls), "模型靠 id 对账,结果少一条都不行"
    assert results[0][1].ok and results[0][1].data == "fast done"
    assert not results[1][1].ok and "timeout" in results[1][1].err
    assert elapsed < 2, f"应在预算 0.5s 附近返回,实际等了 {elapsed:.1f}s"


def test_inner_timeout_clamped_to_budget():
    """模型传的内层 timeout 必须被钳到外层预算内,否则外层先掐。"""
    captured = {}

    def spy(timeout: int = 20):
        captured["timeout"] = timeout
        return ToolResult.success(None)

    agent = _make_agent([Tool("spy", "", {}, spy)], tool_timeout=30)
    agent.execute_tool_calls([ToolCall("spy", {"timeout": 300}, "c1")])

    assert captured["timeout"] == 30


def test_tool_exception_becomes_fail():
    """工具抛异常是数据(fail),不是事故,整轮照常。"""

    def boom():
        raise RuntimeError("炸了")

    agent = _make_agent([Tool("boom", "", {}, boom)], tool_timeout=5)
    results = agent.execute_tool_calls([ToolCall("boom", {}, "c1")])

    assert not results[0][1].ok
    assert "RuntimeError" in results[0][1].err


if __name__ == "__main__":
    test_parallel_timeout_fills_fail()
    test_inner_timeout_clamped_to_budget()
    test_tool_exception_becomes_fail()
    print("all tests passed")
