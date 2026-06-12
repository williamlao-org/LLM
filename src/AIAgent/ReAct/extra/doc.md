抛开 LangChain、Dify 这些高度封装的框架，直接用 `openai` 官方库手搓一个原生的 ReAct 循环，是你深入理解 Agent 本质的最好方式。很多开发者调用了半年 API，都以为 Agent 是什么高深莫测的黑魔法，但当你扒开它的底层，你会发现**它就是一个 `While` 循环加上正则表达式**。

为了自底向上地看清全貌，我们需要先掌握原生 ReAct 运作的两个“核心机密”。

### 核心机密一：经典的纯文本 Prompt 模板

在 2022 年普林斯顿大学姚顺雨（Shunyu Yao）等人发表 ReAct 论文时，大模型还不懂什么是 JSON。研究人员是通过一段极其严格的**纯英文系统提示词**，强迫模型按照特定的格式输出。

下面这段 Prompt 是我刚刚检索了当前高质量英文开源社区（如 LangChain Hub 的 `hwchase17/react` 模板）提取出来的原版架构，至今被无数框架在底层悄悄使用：

```text
Answer the following questions as best you can. You have access to the following tools:
[这里注入工具的名称和描述]

Use the following format strictly:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [tool_names]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
```

### 核心机密二：控制权交接的魔法 (`stop` 参数)

这是高层框架绝对不会告诉你的秘密。

如果把上面那段 Prompt 发给模型，模型会顺着往下写 `Thought` -> `Action` -> `Action Input`。**然后它会企图自己编造一个 `Observation`（幻觉）！** 因为它是文本接龙机器，它不知道停下。

为了阻止它产生幻觉，我们需要在调用 OpenAI API 时，传入一个强制截断参数：`stop=["Observation:"]`。
当模型生成到这个词时，API 会强行打断它，把控制权交还给我们的 Python 代码。我们拿到模型输出的 `Action Input`，在本地运行完真实的函数，再由我们把结果以 `Observation: 真实结果` 的形式追加到对话历史中，让模型继续推导。

---

### 底层全貌实战：纯 Python + OpenAI 实现原生 ReAct

下面是一段完全解耦的底层代码（已根据 2026 年最新的 OpenAI V1 语法编写）。你可以直接把它复制到你的 IDE 里运行。

```python
import re
from openai import OpenAI

# 初始化最新版 OpenAI 客户端
client = OpenAI(api_key="sk-你的真实API_KEY")

# 1. 定义底层系统提示词
REACT_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

get_weather: 获取指定城市的天气。输入参数为城市名，例如：北京
calculate_length: 计算一段文本的字符长度。输入参数为任意字符串。

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [get_weather, calculate_length]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

# 2. 准备本地工具函数
def get_weather(location: str) -> str:
    # 真实场景下这里是 requests.get("天气API")
    return f"{location}今天是晴天，气温25度"

def calculate_length(text: str) -> str:
    return f"长度为 {len(text)}"

available_tools = {
    "get_weather": get_weather,
    "calculate_length": calculate_length
}

# 3. 核心 Agent 大循环
def run_agent(question: str, max_steps: int = 5):
    print(f"User: {question}\n" + "="*40)
    
    # 初始化对话上下文
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": f"Question: {question}\nThought:"}
    ]
    
    for step in range(max_steps):
        # 向 OpenAI 发起请求，核心秘密：设置 stop word！
        response = client.chat.completions.create(
            model="gpt-4o", # 或 gpt-3.5-turbo
            messages=messages,
            temperature=0.1, # 逻辑推理需要极低的随机性
            stop=["Observation:"] # 遇到这个词立刻停止生成，把控制权交还给 Python！
        )
        
        reply = response.choices[0].message.content
        print(f"{reply}") # 打印模型的内部思考和动作
        
        # 将模型的半截思考过程记录到上下文中
        messages.append({"role": "assistant", "content": reply})
        
        # 判断是否得出最终答案
        if "Final Answer:" in reply:
            print("\n✅ 任务完成！")
            break
            
        # 4. 解析需要调用的工具和参数 (传统的文本解析法)
        action_match = re.search(r"Action: (.*)", reply)
        action_input_match = re.search(r"Action Input: (.*)", reply)
        
        if action_match and action_input_match:
            action_name = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            # 执行本地工具
            if action_name in available_tools:
                print(f"\n[系统拦截并执行工具] -> {action_name}('{action_input}')")
                tool_result = available_tools[action_name](action_input)
            else:
                tool_result = f"Error: Tool {action_name} not found."
                
            print(f"[工具返回结果] -> {tool_result}\n" + "-"*40)
            
            # 5. 将真实的观察结果 (Observation) 塞回给模型，继续下一轮循环
            messages.append({"role": "user", "content": f"Observation: {tool_result}\nThought:"})
        else:
            print("❌ 解析失败，模型没有按照格式输出")
            break

# 运行测试：这是一个需要多次调用不同工具的复杂任务
run_agent("北京的天气怎么样？然后告诉我这段天气描述文字的长度是多少。")
```

---

### 历史发展脉络：从正则表达式到 Function Calling

当你运行上面这段代码时，你会看到一个非常清晰的推理、行动、再推理的过程。这就是 2022 年底到 2023 年中旬，全球所有 AI 开发者手搓 Agent 的方式。

但如果你仔细看代码的第 `4` 步，你会发现这种原生方案存在一个**致命的工程缺陷**：它是用正则表达式（Regex）去提取文本的。
如果某天 OpenAI 的模型抽风了，少输出一个空格，写成了 `ActionInput:` 或者输出了中文 `动作输入：`，你的正则表达式就会立刻崩溃（著名的 Parsing Error）。

**为了解决这个痛点，OpenAI 官方在后来推出了原生的 Function Calling（即 `tools` 参数）：**
OpenAI 在底层模型预训练时，专门针对工具调用注入了海量的数据，让模型不再输出非结构化的纯文本，而是强制输出标准的 JSON 对象。

换句话说，现在的 OpenAI `tools` API，只是**替代了上面代码中的 Prompt 格式约束和正则表达式提取部分**，保证了输出100%是代码可读的 JSON，但外层控制整个系统运转的 `For/While` 循环逻辑（Thought $\rightarrow$ Action $\rightarrow$ Observation），依然与上面的代码一模一样。

你可以先在本地跑通上面这个最原始的纯文本 ReAct 循环，感受一下模型是如何被 `stop` 参数逼着“交出”控制权的。如果你测试没问题了，我们下一步就可以把中间那段脆弱的正则表达式，替换成 OpenAI 现代、健壮的 `tools` (Function Calling) 语法，也就是现代 Agent 的标准形态。你想试着跑一下这段代码，还是直接看现代化改造？