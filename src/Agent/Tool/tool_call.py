"""
Tool Calling (工具调用) 底层原理学习

核心问题: LLM 只能生成文本，怎么让它调用函数/工具？

答案: 本质上就是一个 "约定格式" 的文本生成过程：
    1. 你告诉 LLM "你有这些工具可以用"（通过 tools 参数描述工具的 JSON Schema）
    2. LLM 判断是否需要调用工具，如果需要，它生成一段特殊格式的 JSON（而不是普通文本）
    3. 你的代码解析这个 JSON，执行对应的真实函数
    4. 把函数执行结果喂回给 LLM
    5. LLM 根据结果生成最终回答

整个流程:
    用户提问
        ↓
    LLM 分析 → "我需要调用 get_weather 工具"
        ↓
    LLM 输出: {"name": "get_weather", "arguments": {"city": "北京"}}   ← 这不是普通回答，是工具调用指令
        ↓
    你的代码: 好的，我来执行 get_weather("北京")
        ↓
    函数返回: "北京今天晴，25°C"
        ↓
    把结果发回 LLM
        ↓
    LLM: "北京今天天气晴朗，气温25度，适合出门~"   ← 这才是最终回答
"""

import json
from openai import OpenAI

client = OpenAI(
    base_url='http://100.64.0.4:8080/v1',
    api_key='1234567890'
)

MODEL = 'Qwen3.5-27B-Opus4.6-Q4_K_M.gguf'


# ============================================================
# 第一步：定义真实的工具函数
# ============================================================
# 这些就是普通的 Python 函数，LLM 并不会直接调用它们
# 是你的代码负责调用的

def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟）"""
    # 真实场景会调用天气 API，这里用假数据演示流程
    fake_weather = {
        "北京": "晴天，25°C，微风",
        "上海": "多云，22°C，东南风3级",
        "深圳": "阵雨，28°C，湿度85%",
    }
    return fake_weather.get(city, f"{city}：暂无天气数据")


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        # eval 在生产环境不安全，这里纯学习用
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """搜索知识库（模拟）"""
    fake_kb = {
        "Python": "Python 是一种解释型、高级编程语言，由 Guido van Rossum 于 1991 年发布。",
        "Transformer": "Transformer 是 Google 于 2017 年提出的深度学习架构，核心是 Self-Attention 机制。",
    }
    for key, value in fake_kb.items():
        if key.lower() in query.lower():
            return value
    return f"未找到关于 '{query}' 的信息"


# ============================================================
# 第二步：把工具函数映射到一个字典
# ============================================================
# 这样当 LLM 返回 "调用 get_weather" 时，我们能找到对应的函数

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}


# ============================================================
# 第三步：定义工具描述（JSON Schema 格式）
# ============================================================
# 这是发给 LLM 的 "工具说明书"
# LLM 根据这些描述来决定什么时候用什么工具、传什么参数
#
# 格式要求（OpenAI 兼容格式）：
#   type: "function"
#   function:
#     name:        工具名（要和 TOOL_MAP 的 key 对应）
#     description: 工具的功能描述（LLM 靠这个判断什么时候该用这个工具）
#     parameters:  参数的 JSON Schema（告诉 LLM 需要传什么参数）

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息。当用户询问某个城市的天气时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、深圳"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。当用户需要进行数学计算时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：2+3、100/4、2**10"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库获取信息。当用户询问某个概念或技术的定义时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
]


# ============================================================
# 第四步：执行工具调用（核心逻辑）
# ============================================================

def execute_tool_call(tool_call):
    """
    执行单个工具调用。

    tool_call 是 LLM 返回的对象，结构：
        tool_call.id            → 调用 ID（用于后续匹配结果）
        tool_call.function.name → 函数名
        tool_call.function.arguments → 参数 JSON 字符串
    """
    func_name = tool_call.function.name
    # LLM 返回的 arguments 是 JSON 字符串，需要解析
    func_args = json.loads(tool_call.function.arguments)

    print(f"  🔧 调用工具: {func_name}({func_args})")

    # 从 TOOL_MAP 找到真实函数并执行
    if func_name in TOOL_MAP:
        result = TOOL_MAP[func_name](**func_args)
    else:
        result = f"未知工具: {func_name}"

    print(f"  📋 工具返回: {result}")
    return result


# ============================================================
# 第五步：完整的 Tool Calling 流程
# ============================================================

def chat_with_tools(user_message: str):
    """
    完整的工具调用流程演示。

    流程图:
        用户消息
            ↓
        [第1次调用 LLM] 带 tools 参数
            ↓
        LLM 返回 tool_calls?
            ├── 否 → 直接返回回答（不需要工具）
            └── 是 → 执行工具 → 把结果放回 messages
                        ↓
                  [第2次调用 LLM] 带工具结果
                        ↓
                  LLM 生成最终回答
    """
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_message}")
    print(f"{'='*60}")

    # 构造消息列表
    messages = [
        {"role": "user", "content": user_message}
    ]

    # ---- 第1次调用 LLM ----
    # 关键：传入 tools 参数，告诉 LLM 有哪些工具可用
    print("\n📡 第1次调用 LLM（带 tools 描述）...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,           # ← 关键！把工具描述发给 LLM
        tool_choice="auto",    # ← "auto" = 让 LLM 自己决定要不要用工具
    )

    assistant_message = response.choices[0].message

    # ---- 检查 LLM 是否要调用工具 ----
    if assistant_message.tool_calls is None:
        # LLM 认为不需要工具，直接回答了
        print("\n💬 LLM 直接回答（未使用工具）:")
        print(f"  {assistant_message.content}")
        return assistant_message.content

    # ---- LLM 要求调用工具 ----
    print(f"\n🔄 LLM 要求调用 {len(assistant_message.tool_calls)} 个工具:")

    # 把 LLM 的回复（包含 tool_calls）加入消息列表
    # 这一步很重要！LLM 需要看到自己之前说了什么
    messages.append(assistant_message)

    # 依次执行每个工具调用
    for tool_call in assistant_message.tool_calls:
        result = execute_tool_call(tool_call)

        # 把工具执行结果加入消息列表
        # role="tool" 表示这是工具的返回结果
        # tool_call_id 用于匹配是哪个调用的结果
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    # ---- 第2次调用 LLM ----
    # 现在消息列表包含了: 用户问题 + LLM的工具调用请求 + 工具执行结果
    # LLM 会根据工具结果生成最终的自然语言回答
    print("\n📡 第2次调用 LLM（带工具执行结果）...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        # 注意：第2次不需要再传 tools 了（也可以传，取决于是否允许连续调用）
    )

    final_answer = response.choices[0].message.content
    print(f"\n💬 最终回答:")
    print(f"  {final_answer}")
    return final_answer


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    print("🎓 Tool Calling 工具调用 学习示例")
    print("=" * 60)

    # 测试 1：需要调用天气工具
    chat_with_tools("北京今天天气怎么样？")

    # 测试 2：需要调用计算工具
    chat_with_tools("帮我算一下 1024 * 768 等于多少")

    # 测试 3：需要调用知识搜索工具
    chat_with_tools("什么是 Transformer？")

    # 测试 4：不需要工具的普通对话
    chat_with_tools("你好，你是谁？")

    print("\n" + "=" * 60)
    print("📚 关键总结:")
    print("  1. LLM 不会真的执行函数，它只是生成 '我要调用xxx' 的 JSON 指令")
    print("  2. 你的代码负责解析指令、执行函数、把结果喂回 LLM")
    print("  3. tools 参数里的 description 非常重要，LLM 靠它决定用哪个工具")
    print("  4. 整个过程至少需要调用 LLM 两次（判断+总结）")
    print("  5. tool_choice='auto' 让 LLM 自己决定是否需要工具")
