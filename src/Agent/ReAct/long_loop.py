from openai import OpenAI
import re

# client = OpenAI(base_url="http://100.64.0.4:8080/v1", api_key="1234567890")
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="1234567890")

MODEL = "Qwen3.5-27B-Opus4.6-Q4_K_M.gguf"
DEFAULT_MAX_STEPS = 10
MIN_TOOL_STEPS_BEFORE_FINAL = 4

REACT_PROMPT = """
尽可能地回答用户的问题。你可以使用以下工具：
1. get_weather: 获取指定城市的天气。输入参数为城市名，例如：北京
2. calculate_length: 计算一段文本的字符长度。输入参数为任意字符串。

请严格遵循以下格式进行回答：

Question: 你需要回答的输入问题
Thought: 你应该始终思考接下来要做什么
Action: 你要采取的行动，必须是 [get_weather, calculate_length] 之一
Action Input: 采取该行动的输入参数
Observation: 动作的结果
... (Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 针对原始问题的最终答案

额外要求：
1. 在给出 Final Answer 之前，至少完成 4 轮完整的 Thought/Action/Action Input/Observation 循环。
2. 如果信息已经足够，也不要提前结束；继续做补充验证、补充计算或交叉检查。
3. 只要还没达到最少工具轮数，就不要输出 Final Answer。

---开始---
"""


def get_weather(city: str):
    """获取指定城市的天气"""
    return f"{city}天气晴朗，气温25°C"


def calculate_length(text: str):
    """计算一段文本的字符长度"""
    return len(text)


available_tools = {"get_weather": get_weather, "calculate_length": calculate_length}


def run_agent(
    question: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    min_tool_steps: int = MIN_TOOL_STEPS_BEFORE_FINAL,
):
    print(f"User:{question}\n" + "=" * 40)

    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": f"Question:{question}\nThought:"},
    ]

    tool_steps_executed = 0

    for step in range(1, max_steps + 1):
        print(f"\n----- Step {step}/{max_steps} -----")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            stop=["Observation:"],
        )

        reply = response.choices[0].message.content or ""
        print(reply)
        messages.append({"role": "assistant", "content": reply})

        if "Final Answer:" in reply and tool_steps_executed < min_tool_steps:
            print(
                f"\n⚠️ 工具循环仅完成 {tool_steps_executed} 轮，"
                f"未达到最少 {min_tool_steps} 轮，继续思考"
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: 你过早给出了 Final Answer。"
                        f"当前仅完成 {tool_steps_executed} 轮工具调用，"
                        f"至少需要 {min_tool_steps} 轮。请继续输出 Thought/Action/Action Input。"
                    ),
                }
            )
            continue

        if "Final Answer:" in reply:
            print("\n✅ 任务完成！")
            print('=' * 40+'模型完整回复'+'='*40)
            print(response.model_dump_json(indent=2))
            return

        action_match = re.search(r"Action:\s*(\w+)", reply)
        action_input_match = re.search(r"Action Input:\s*(.+)", reply)

        if not (action_match and action_input_match):
            print("\n⚠️ 无法解析 Action/Action Input，停止循环")
            return

        action_name = action_match.group(1).strip()
        action_input = action_input_match.group(1).strip()

        if action_name not in available_tools:
            print(f"\n⚠️ 未知工具: {action_name}")
            messages.append(
                {
                    "role": "user",
                    "content": f"Observation: 未知工具 {action_name}\nThought:",
                }
            )
            continue

        print(f'\n[系统拦截并执行工具] -> {action_name}("{action_input}")')
        observation = available_tools[action_name](action_input)
        tool_steps_executed += 1
        print(f"\n🔧 工具执行结果: {observation}")
        messages.append({"role": "user", "content": f"Observation:{observation}\nThought:"})

    print("\n⚠️ 达到最大步数，任务未完成")


if __name__ == "__main__":
    run_agent("请先查询北京天气，再计算“北京天气晴朗，气温25°C”这句话的字符长度，然后再做两次补充检查，最后统一总结。")

