"""
基于 GraphFlow 重构的 Multi-Agent 系统（真实开发流水线）

流转过程：
planner → coder → reviewer → tester → arbiter ──→ 验收通过？
                                                 ├─ 是 → END
                                                 └─ 否 → coder (回炉重造)
"""
import json
import os
import subprocess
import time
from typing import Any, TypedDict

from openai import OpenAI

# 引用我们之前写的图引擎
from StateMachine_GraphFlow.rewrite_graph import GraphFlow, END

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-")
MODEL = "zai-org/GLM-4.6"


# ============================================================
# 1. 定义 Multi-Agent 的全局共享状态 (State)
# ============================================================
# 为了兼容 rewrite_graph.State 的类型定义，我们包含基础字段，并加入 MA 需要的专有字段
class MultiAgentState(TypedDict):
    # Base requirements from GraphFlow
    user_input: str
    messages: list
    tool_calls: list
    answer: str
    trace: list

    # Multi-Agent 的业务字段
    task: str
    plan: dict
    draft_code: str
    review: dict
    test_report: dict
    verdict: dict
    history: list  # 保存历史轮次的反馈，避免反复踩坑


# ============================================================
# 2. 角色提示词 (System Prompts)
# ============================================================
PLANNER_PROMPT = """你是一名资深架构师。请把任务拆解成执行计划。
只返回有效JSON，包含：
- objective: string
- steps: string[]"""

CODER_PROMPT = """你是一名高级Python开发工程师。结合计划和历史审查反馈，编写/修改代码。
只返回有效JSON，包含：
- summary: string
- code: string"""

REVIEWER_PROMPT = """你是一名严苛的代码审查员。
检查代码的问题并给出修改建议。只返回有效JSON，包含：
- approved: boolean
- issues: string[]"""

TESTER_PROMPT = """你是一名测试工程师。
根据代码，决定是否需要执行测试以及运行什么命令（如 `python -c "..."`）。
由于是教学演示，你可以写一段包含断言 assert 的验证代码，用 `python -c` 运行。
只返回有效JSON，包含：
- run_tests: boolean
- command: string"""

ARBITER_PROMPT = """你是一名发版仲裁者。
综合代码、审查意见和测试结果，决定是否验收通过。
只返回有效JSON，包含：
- accept: boolean
- reason: string"""


# ============================================================
# 3. 工具类：LLM 交互与安全命令执行
# ============================================================
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def ask_json(prompt: str, payload: dict, temperature: float = 0.1) -> dict:
    """包装一下，让大模型只返回 JSON"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"请返回 JSON 格式结果。输入信息：\n{json.dumps(payload, ensure_ascii=False)}"}
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"} # 强制返回 JSON
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


class SafeCommandRunner:
    """安全的命令执行器，由于是本地运行，我们做个简单的拦截"""
    def run(self, command: str) -> dict:
        if not command or "rm " in command or "del " in command:
            return {"ok": False, "reason": "不允许执行危险命令或命令为空", "stdout": ""}
        try:
            completed = subprocess.run(command, shell=True, timeout=5, capture_output=True, text=True)
            return {
                "ok": completed.returncode == 0,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "command": command
            }
        except Exception as e:
            return {"ok": False, "reason": str(e), "stdout": ""}

runner = SafeCommandRunner()


# ============================================================
# 4. 节点定义 (Nodes)
# ============================================================

def trace_print(state: MultiAgentState, msg: str):
    """记录到 trace 并实时打印，让你能在控制台看到进度"""
    print(msg)
    state["trace"].append(msg)


def node_planner(state: MultiAgentState):
    trace_print(state, "\n[架构师] 正在制定计划...")
    plan = ask_json(PLANNER_PROMPT, {"task": state["task"]})
    state["plan"] = plan
    trace_print(state, f"└─ 计划完成：{plan}")


def node_coder(state: MultiAgentState):
    trace_print(state, "\n[程序员] 正在编写代码...")
    # 把之前失败的教训告诉程序员
    feedback = [
        {"review": h["review"], "verdict": h["verdict"]} 
        for h in state["history"]
    ]
    result = ask_json(CODER_PROMPT, {
        "task": state["task"],
        "plan": state["plan"],
        "previous_feedback": feedback
    }, temperature=0.3)
    
    state["draft_code"] = result.get("code", "# 没写出代码")
    trace_print(state, f"└─ 提交代码：\n{state['draft_code'][:100]}...\n")


def node_reviewer(state: MultiAgentState):
    trace_print(state, "[审查员] 正在 Review...")
    review = ask_json(REVIEWER_PROMPT, {
        "plan": state["plan"],
        "candidate_code": state["draft_code"]
    })
    state["review"] = review
    trace_print(state, f"└─ 审查结果：通过={review.get('approved', False)}，发现 {len(review.get('issues', []))} 个问题")


def node_tester(state: MultiAgentState):
    trace_print(state, "[测试员] 正在构思测试...")
    tester = ask_json(TESTER_PROMPT, {"candidate_code": state["draft_code"]})
    
    run_cmd = tester.get("run_tests", False)
    cmd = tester.get("command", "")
    
    if run_cmd and cmd:
        trace_print(state, f"└─ 执行测试: {cmd}")
        report = runner.run(cmd)
        trace_print(state, f"   => 成功={report['ok']}")
    else:
        report = {"ok": True, "reason": "未执行任何测试"}
        trace_print(state, "└─ 跳过本地测试执行")
        
    state["test_report"] = report


def node_arbiter(state: MultiAgentState):
    trace_print(state, "\n[仲裁者] 正在最终验收...")
    verdict = ask_json(ARBITER_PROMPT, {
        "candidate_code": state["draft_code"],
        "review": state["review"],
        "test_report": state["test_report"]
    })
    state["verdict"] = verdict
    trace_print(state, f"└─ 仲裁结果：验收={'通过' if verdict.get('accept') else '打回'}, 理由: {verdict.get('reason')}")


# ============================================================
# 5. 路由函数 (Conditional Routers)
# ============================================================

def route_after_arbiter(state: MultiAgentState) -> str:
    """根据仲裁结果，决定是结束还是打回给程序员重写"""
    if state["verdict"].get("accept", False):
        return END
    else:
        # 如果不通过，把这轮的结论存入历史记录，以便下一轮交给 Coder 吸收教训
        state["history"].append({
            "review": state["review"],
            "verdict": state["verdict"]
        })
        return "coder"


# ============================================================
# 6. 组装 GraphFlow
# ============================================================

def build_multi_agent() -> GraphFlow:
    g = GraphFlow()

    # 注册节点
    g.add_node("planner", node_planner)
    g.add_node("coder", node_coder)
    g.add_node("reviewer", node_reviewer)
    g.add_node("tester", node_tester)
    g.add_node("arbiter", node_arbiter)

    # 连边排流水线
    g.add_edge("planner", "coder")
    g.add_edge("coder", "reviewer")
    g.add_edge("reviewer", "tester")
    g.add_edge("tester", "arbiter")

    # 唯一的控制回环：根据仲裁结果路由
    g.add_conditional_edges("arbiter", route_after_arbiter, {
        "coder": "coder", # 打回重造
        END: END          # 验收通过结束
    })

    g.set_start("planner")
    return g


# ============================================================
# 测试运行
# ============================================================
if __name__ == "__main__":
    task_desc = "写一个 Python 函数快速排序 quick_sort()。必须处理边界情况（如空数组）。"
    
    agent_graph = build_multi_agent()
    
    # 初始化状态板
    initial_state: MultiAgentState = {
        "user_input": "", "messages": [], "tool_calls": [], "answer": "", "trace": [],
        "task": task_desc,
        "plan": {},
        "draft_code": "",
        "review": {},
        "test_report": {},
        "verdict": {},
        "history": []
    }
    
    print(f"=== 任务: {task_desc} ===")
    
    # 限制最多走 15 步 (防止死循环无限花钱)
    final_state = agent_graph.run(initial_state, max_steps=15)
    
    print("\n\n=== 最终生成的代码 ===")
    print(final_state["draft_code"])
