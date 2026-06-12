import os

from openai import OpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.64.0.4:8080/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "1234567890")
MODEL = os.getenv("OPENAI_MODEL", "Qwen3.5-27B-Opus4.6-Q4_K_M.gguf")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ==========================================
# 1. 定义两个截然不同的 Agent 人设 (System Prompts)
# ==========================================
CODER_PROMPT = """
你是一个顶级的 Python 程序员。
你的任务是根据用户的需求或者 Reviewer 的反馈，编写或修改代码。
请只输出代码，不要多余的废话。
"""

REVIEWER_PROMPT = """
你是一个严苛的代码审查员 (Code Reviewer)。
你的任务是检查 Coder 提交的 Python 代码。
寻找潜在的 Bug、性能问题或不优雅的写法。
如果你发现了问题，请指出并要求修改；
如果代码完美无缺，请直接回复大写的 "APPROVED"。
"""


# ==========================================
# 2. 基础 LLM 调用函数
# ==========================================
def ask_agent(role_prompt: str, chat_history: list) -> str | None:
    messages = [{"role": "system", "content": role_prompt}] + chat_history
    response = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.2  # 编程任务保持低随机性
    )
    return response.choices[0].message.content


# ==========================================
# 3. 核心引擎：多智能体辩论循环 (The Debate Loop)
# ==========================================
def run_multi_agent_debate(task: str, max_turns: int = 3):
    print(f"🎯 初始任务: {task}\n" + "=" * 50)

    # 初始化两套独立的记忆（上下文）
    coder_history = [{"role": "user", "content": task}]

    for turn in range(max_turns):
        print(f"\n🔄 [第 {turn + 1} 轮辩论开始]")

        # 1. Coder 行动
        print("🧑‍💻 程序员 Agent 正在编写代码...")
        draft_code = ask_agent(CODER_PROMPT, coder_history)
        print(f"--- 程序员提交的代码 ---\n{draft_code}\n------------------------")

        # 将自己写的代码记入 Coder 的记忆
        coder_history.append({"role": "assistant", "content": draft_code})

        # 2. Reviewer 审查 (把 Coder 的输出作为 Reviewer 的输入)
        print("🕵️‍♂️ 审查员 Agent 正在 Review...")
        reviewer_history = [
            {"role": "user", "content": f"请审查以下代码：\n{draft_code}"}
        ]
        review_feedback = ask_agent(REVIEWER_PROMPT, reviewer_history)
        print(f"--- 审查员反馈 ---\n{review_feedback}\n------------------")

        # 3. 仲裁：是否通过？
        if "APPROVED" in review_feedback:
            print("\n✅ 审查通过！多智能体协作完成。")
            break

        # 4. 如果没通过，把 Reviewer 的反馈塞给 Coder，进入下一轮
        print("\n❌ 审查未通过，打回给程序员修改...")
        coder_history.append(
            {
                "role": "user",
                "content": f"Reviewer的反馈：{review_feedback}。请据此修改代码。",
            }
        )

    else:
        print("\n⚠️ 达到最大辩论轮数，强制终止。")


# 运行测试：故意给一个容易写错或写得不够优雅的数学任务
run_multi_agent_debate("写一个 Python 函数，计算斐波那契数列的第 N 项。")
