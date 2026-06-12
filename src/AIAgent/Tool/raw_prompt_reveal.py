"""
揭秘：模型真正看到的是什么？

答案：模型是一个 next-token predictor，它只认识一样东西 —— 一段连续的 token 序列（文本）。

所谓的 messages=[{"role":"user", "content":"..."}] 只是 API 层的抽象。
在真正喂入模型之前，会经过一个叫 "Chat Template" 的模板，
把 messages 列表拼接成一段纯文本。

这个过程：
    messages (结构化数据)
        ↓  Chat Template 渲染
    一段纯文本 prompt
        ↓  Tokenizer
    token IDs [整数序列]
        ↓  喂入模型
    模型生成下一个 token
"""

# ============================================================
# 第一层真相：Chat Template 把 messages 变成纯文本
# ============================================================

# 以 Qwen2/ChatML 格式为例（你的 Qwen 模型用的就是这个格式）
# 不同模型用不同模板，但本质都一样 —— 把 messages 拼成一段字符串

def apply_chatml_template(messages: list[dict]) -> str:
    """
    手动模拟 ChatML 模板的渲染过程。

    ChatML 是 OpenAI/Qwen 等模型使用的 chat 格式。
    它用特殊标记 <|im_start|> 和 <|im_end|> 来分隔不同角色的消息。

    这就是模型真正看到的东西！
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # 最后加上 assistant 的开头，让模型接着生成
    prompt += "<|im_start|>assistant\n"
    return prompt


# ============================================================
# 演示 1：普通对话，模型实际看到了什么
# ============================================================

print("=" * 70)
print("演示 1：普通对话")
print("=" * 70)

messages_simple = [
    {"role": "user", "content": "你好"}
]

print("\n📦 API 层看到的（结构化 messages）:")
for m in messages_simple:
    print(f"  {m}")

print("\n📜 模型实际看到的（纯文本，经过 Chat Template 渲染）:")
raw = apply_chatml_template(messages_simple)
print(raw)
print("  ↑ 模型从这里开始续写，逐 token 生成回答")


# ============================================================
# 演示 2：带 system prompt 的对话
# ============================================================

print("\n" + "=" * 70)
print("演示 2：带 system prompt")
print("=" * 70)

messages_system = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "1+1等于几？"},
]

print("\n📦 API 层:")
for m in messages_system:
    print(f"  {m}")

print("\n📜 模型实际看到的:")
print(apply_chatml_template(messages_system))


# ============================================================
# 演示 3：Tool Calling 模型实际看到了什么（核心！）
# ============================================================

print("\n" + "=" * 70)
print("演示 3：Tool Calling 完整流程，模型真正看到的文本")
print("=" * 70)

# 这里完全手动展示 tool calling 在模型眼里的样子
# 不同模型的 tool calling 格式不同，以下是 Qwen 的典型格式

def apply_chatml_template_with_tools(messages, tools=None):
    """
    带 Tool 描述的 ChatML 模板渲染。

    关键发现：tools 的描述也是以纯文本形式塞进 system prompt 的！
    """
    prompt = ""

    # 如果有 tools，把工具说明塞进 system 消息
    if tools:
        tool_desc = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        tool_desc += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        for tool in tools:
            f = tool["function"]
            tool_desc += json.dumps(f, ensure_ascii=False) + "\n"
        tool_desc += "</tools>\n\n"
        tool_desc += 'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
        tool_desc += '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'

        prompt += f"<|im_start|>system\n{tool_desc}<|im_end|>\n"

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "assistant" and "tool_calls" in msg:
            # assistant 的 tool_call 消息：模型生成的实际是这样的文本
            tc = msg["tool_calls"][0]
            tool_call_text = f'<tool_call>\n{{"name": "{tc["function"]["name"]}", "arguments": {tc["function"]["arguments"]}}}\n</tool_call>'
            prompt += f"<|im_start|>assistant\n{tool_call_text}<|im_end|>\n"
        elif role == "tool":
            # 工具结果：也是以特殊标记包裹的纯文本
            prompt += f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
        else:
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt


import json

# 工具描述
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

print("\n--- 第 1 次调用 LLM：模型看到的文本 ---\n")

messages_step1 = [
    {"role": "user", "content": "北京天气怎么样？"},
]
raw1 = apply_chatml_template_with_tools(messages_step1, tools=tools)
print(raw1)
print("  ↑ 模型在这里续写。因为 system 里提到了 <tool_call> 格式，")
print("    模型会生成: <tool_call>{\"name\":\"get_weather\",...}</tool_call>")

print("\n--- 第 2 次调用 LLM：带工具结果，模型看到的文本 ---\n")

messages_step2 = [
    {"role": "user", "content": "北京天气怎么样？"},
    # 模型第1次的输出（tool call 请求）
    {"role": "assistant", "tool_calls": [{
        "id": "call_001",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}
    }]},
    # 工具执行结果
    {"role": "tool", "tool_call_id": "call_001", "content": "晴天，25°C，微风"},
]
raw2 = apply_chatml_template_with_tools(messages_step2, tools=tools)
print(raw2)
print("  ↑ 模型在这里续写最终回答: '北京今天晴天，气温25度...'")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 70)
print("📚 总结：真相的三层洋葱")
print("=" * 70)
print("""
  你写的代码（最外层）:
  ┌─────────────────────────────────────────────────┐
  │ messages = [                                    │
  │   {"role": "user", "content": "北京天气？"},     │
  │ ]                                               │
  │ client.chat.completions.create(                 │
  │   messages=messages, tools=tools                │
  │ )                                               │
  └─────────────────────────────────────────────────┘
                      ↓  openai 库 / llama.cpp server 做的事
  API / 推理服务器（中间层）:
  ┌─────────────────────────────────────────────────┐
  │ 1. 用 Chat Template 把 messages 拼成纯文本       │
  │ 2. 用 Tokenizer 把文本转换成 token IDs           │
  │ 3. 喂入模型，逐 token 生成                       │
  │ 4. 检测到 <tool_call> 标记 → 解析成结构化数据     │
  │ 5. 返回给你: tool_calls=[...]                    │
  └─────────────────────────────────────────────────┘
                      ↓
  模型（最底层）:
  ┌─────────────────────────────────────────────────┐
  │ 只看到一串 token IDs:                            │
  │ [151644, 8948, 198, ..., 151645, 198, ...]      │
  │                                                 │
  │ 对应的文本:                                      │
  │ <|im_start|>system                              │
  │ # Tools                                         │
  │ You may call one or more functions...            │
  │ <|im_end|>                                      │
  │ <|im_start|>user                                │
  │ 北京天气怎么样？<|im_end|>                        │
  │ <|im_start|>assistant                           │
  │ ← 从这里开始逐个token生成                        │
  └─────────────────────────────────────────────────┘

  关键认知：
  • role="user/assistant/tool" 是 API 的抽象概念
  • 到模型层面，全部变成一段连续文本，用特殊标记分隔
  • <|im_start|>、<|im_end|> 是特殊 token，训练时学会的
  • tools 的描述也是纯文本，塞在 system prompt 里
  • 模型"调用工具"其实只是生成了 <tool_call>...</tool_call> 这段文本
  • 是外层程序（llama.cpp/vLLM）检测到这个标记后，解析成结构化数据返回给你
""")
