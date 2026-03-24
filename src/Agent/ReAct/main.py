from openai import OpenAI
import re

client = OpenAI(base_url="http://100.64.0.4:8080/v1", api_key="1234567890")

MODEL = "Qwen3.5-27B-Opus4.6-Q4_K_M.gguf"

REACT_PROMPT = """
尽可能地回答用户的问题。你可以使用以下工具：
1. get_weather: 获取指定城市的天气。输入参数为城市名，例如：北京
2. calculate_length: 计算一段文本的字符长度。输入参数为任意字符串。

请严格遵循以下格式进行回答：

Question: 你需要回答的输入问题
Thought: 你你应该始终思考接下来要做什么
Action: 你要采取的行动，必须是 [get_weather, calculate_length] 之一
Action Input: 采取该行动的输入参数
Observation: 动作的结果
... (Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 针对原始问题的最终答案

---开始---
"""


def get_weather(city: str):
    """获取指定城市的天气"""
    return f"{city}天气晴朗，气温25°C"


def calculate_length(text: str):
    """计算一段文本的字符长度"""
    return len(text)


available_tools = {"get_weather": get_weather, "calculate_length": calculate_length}


def run_agent(question: str, max_steps: int = 5):
    print(f"User:{question}\n" + "=" * 40)

    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": f"Question:{question}\nThought:"},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            stop=["Observation:"],  # 遇到这个词立刻停止生成，把控制权交还给 Python！
        )

        reply = response.choices[0].message.content
        print(f"{reply}")

        # 将模型的半截思考过程记录到上下文中
        messages.append({"role": "assistant", "content": reply})

        if "Final Answer:" in reply:
            print("\n✅ 任务完成！")
            break

        # 提取 Action 和 Action Input
        action_match = re.search(r"Action:\s*(\w+)", reply)
        action_input_match = re.search(r"Action Input:\s*(.+)", reply)

        if action_match and action_input_match:
            action_name = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()

            if action_name in available_tools:
                print(f'\n[系统拦截并执行工具] -> {action_name}("{action_input}")')

                tool_func = available_tools[action_name]
                observation = tool_func(action_input)
                print(f"\n🔧 工具执行结果: {observation}")

                # 将工具结果反馈给模型
                messages.append(
                    {"role": "user", "content": f"Observation:{observation}\nThought:"}
                )
            else:
                print(f"\n⚠️ 未知工具: {action_name}")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Observation: 未知工具 {action_name}\nThought:",
                    }
                )
        else:
            print("\n⚠️ 无法解析 Action/Action Input，停止循环")
            break

    else:
        print("\n⚠️ 达到最大步数，任务未完成")


if __name__ == "__main__":
    run_agent("北京天气怎么样？")
