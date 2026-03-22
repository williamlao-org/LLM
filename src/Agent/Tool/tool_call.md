# Tool Calling 工具调用 学习笔记

## 一、Tool Calling 的本质

LLM 只能生成文本，它不会真的执行函数。所谓"工具调用"的底层原理：

1. 你告诉 LLM "你有这些工具可以用"（通过 `tools` 参数，以 JSON Schema 描述工具）
2. LLM 判断是否需要调用工具，如果需要，它生成一段特殊格式的 JSON（而不是普通文本回答）
3. **你的代码**解析这个 JSON，执行对应的真实 Python 函数
4. 把函数执行结果喂回给 LLM
5. LLM 根据结果生成最终的自然语言回答

**整个过程至少 2 次 LLM 调用**：第1次让它决定要不要用工具，第2次让它根据工具结果回答。

---

## 二、模型真正看到的是什么？

`messages=[{"role":"user", "content":"..."}]` 这种结构化数据是 **API 层的抽象**。

模型本身是一个 next-token predictor，它只认一样东西——**一段连续的 token 序列（文本）**。

在真正喂入模型之前，会经过一个叫 **Chat Template** 的模板，把 messages 列表拼接成一段纯文本。

### 普通对话

API 层看到的：
```json
[{"role": "user", "content": "你好"}]
```

模型实际看到的（纯文本，经过 Chat Template 渲染）：
```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```
↑ 模型从这里开始续写，逐 token 生成回答

### 带 system prompt

API 层：
```json
[
  {"role": "system", "content": "你是一个有用的助手。"},
  {"role": "user", "content": "1+1等于几？"}
]
```

模型实际看到的：
```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
1+1等于几？<|im_end|>
<|im_start|>assistant
```

---

## 三、Tool Calling 完整流程，模型真正看到的文本

### 第 1 次调用 LLM

```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "get_weather", "description": "获取城市天气", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "城市名"}}, "required": ["city"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
```

↑ 模型在这里续写。因为 system 里提到了 `<tool_call>` 格式，模型会生成: `<tool_call>{"name":"get_weather",...}</tool_call>`

### 第 2 次调用 LLM（带工具结果）

```
<|im_start|>system
# Tools
...（同上，工具描述）...<|im_end|>
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "北京"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
晴天，25°C，微风
</tool_response><|im_end|>
<|im_start|>assistant
```

↑ 模型在这里续写最终回答: "北京今天晴天，气温25度..."

---

## 四、理解总结（三层洋葱）

```
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
│ ← 从这里开始逐个 token 生成                      │
└─────────────────────────────────────────────────┘
```

### 关键认知

- `role="user/assistant/tool"` 是 API 的抽象概念
- 到模型层面，全部变成一段连续文本，用特殊标记 `<|im_start|>` `<|im_end|>` 分隔
- `<|im_start|>` `<|im_end|>` 是特殊 token，训练时学会的，不会被 BPE 拆分
- tools 的描述也是纯文本，塞在 system prompt 里用 `<tools>` XML 标签包裹
- 模型"调用工具"其实只是生成了 `<tool_call>...</tool_call>` 这段文本
- 是外层程序（llama.cpp/vLLM）检测到这个标记后，解析成结构化数据返回给你
- API 会在末尾接上 `<|im_start|>assistant\n` 来引导 LLM 输出 assistant 段

### `<|im_start|>` 和 `<|im_end|>` 是特殊 token

```python
# Qwen 的词表里大概是这样的
token_to_id = {
    "<|im_start|>": 151644,  # 一个整数 ID
    "<|im_end|>":   151645,
}
```

它们不会被 BPE 拆分，而是作为一整个 token 直接映射成 ID。模型在训练时学会了：看到 `151644`（`<|im_start|>`）后面跟 `assistant`，就知道该"扮演助手角色"开始输出了。

---

## 五、每轮的结构规律

原来是无数个 `<|im_start|>` `<|im_end|>` 包装起来的，`<|im_start|>` 后面会立即接一个主体词（user/system/assistant）。

- 每一次都会带着前面很多个 `<im>` 里的内容
- 新的输出不是接入到之前的 `<im>` 段里，而是**再新开一个 `<im>` 段**
- 工具调用时 assistant 主体里放的不是结果，而是模型的 JSON 调用输出，用 `<tool_call>` 包装
- 工具信息放在了 system 字段里，用 `<tools>` 包装
- 回复工具结果时，你的代码执行工具并在 user 主体里用 `<tool_response>` 包装了工具输出
- 用户输入就是 user，模型输出就是 assistant，system 是模型定位加工具信息等内容
- 每次对话都是新加 `<im_start>` user/system/assistant `<im_end>` 段
- API 还会接上 `<|im_start|>assistant\n` 来引导 LLM 输出 assistant 段

---

## 六、Think 模式（思考模式）是怎么回事？

Think 模式**不是** API 在输入加了 `<|im_thinking|>` 标记，而是**模型在训练时就学会了在输出里生成 `<think>` 标记**。

### 模型实际生成的文本

```
<|im_start|>assistant
<think>
嗯，用户问的是北京天气...
我需要考虑当前季节...
北京3月份一般还挺冷的...
</think>

北京今天天气晴朗，气温约15°C，建议穿外套。<|im_end|>
```

### 各层做了什么

| 层面 | 做了什么 |
|------|---------|
| **训练阶段** | 模型在训练数据里见过大量 `<think>...</think>` 格式的文本，学会了"先想后答" |
| **模型输出** | 模型自己生成 `<think>思考过程</think>`，然后接着生成正式回答 |
| **API 层** | 解析输出，把 `<think>` 里的内容提取出来放到 `reasoning_content` 字段，`</think>` 后面的放到 `content` 字段 |

### 开/关对比

```
不开 think:
    模型生成 → "北京今天晴天15度"
    API 返回 → content: "北京今天晴天15度"

开 think:
    模型生成 → "<think>用户问天气...我来分析...</think>\n\n北京今天晴天15度"
    API 返回 → reasoning_content: "用户问天气...我来分析..."
              content: "北京今天晴天15度"
```

### Think 模式怎么"关"的？

```
开 think（默认）:
    ...
    <|im_start|>assistant
    ← 模型自己决定先输出 <think>...</think> 再给答案

关 think:
    ...
    <|im_start|>assistant
    <think>
    </think>          ← API 预填了一个空的 think 块，模型就跳过思考直接回答
```

关闭时，API 层会在 `<|im_start|>assistant\n` 后面**预填** `<think>\n</think>\n\n`，模型看到思考块已经"结束了"，就直接输出正式回答。

### 本质

**Think 模式 = 训练 + 特殊 token，跟 tool calling 原理完全一样：**

- Tool calling：模型学会生成 `<tool_call>...</tool_call>`，API 层解析
- Think 模式：模型学会生成 `<think>...</think>`，API 层解析

都是模型在训练时学会了"在特定标记内输出特定格式的内容"，然后外层程序负责解析和展示。

### 发展

> [!QUESTION]
> 开始是由Zero shot CoT或Few shot cot这种来引入思考过程,think模式的出现是不是相当于它们的延伸,在训练阶段就引入了思考,让用户不需要在提示词里引导,模型也会自动来一步步推演?所以是不是现在的zero shot或few shot是不是没太多必要了?

你的理解非常准确！这正是这个领域的演进脉络。让我帮你理清楚：

#### 演进关系

```
2022: Zero-Shot CoT / Few-Shot CoT
      ↓  "在提示词里手动引导推理"（推理时的技巧）
      
2023-2024: o1 / DeepSeek-R1 / QwQ 等 Thinking 模型
      ↓  "在训练阶段就让模型学会推理"（训练时内化了）

本质: 把 Prompt 层的技巧 → 内化到模型权重里
```

||CoT (Prompt 技巧)|Think 模式 (训练内化)|
|---|---|---|
|谁在引导推理|用户在 prompt 里引导|模型自己就会|
|实现层面|推理时加提示词|训练时用 RL + <think> 数据|
|推理质量|取决于你写的提示词和示例|模型自己学会了更深层的推理|
|代价|用户需要精心设计 prompt|训练成本高，推理时多消耗 token|

##### 但 CoT 并没有完全没用

1. **Think 模式 ≈ 内化了 Zero-Shot CoT**

没错，对于大多数场景，开了 think 的模型已经不需要你手动写 "Let's think step by step" 了——模型自己就会这样做。**所以 Zero-Shot CoT 确实基本被取代了。**

2. **Few-Shot CoT 仍有价值**

Few-Shot 的核心不只是"让模型思考"，而是**控制推理的方向和格式**：

```
场景：你要模型用特定的财务分析框架来评估公司
  - Think 模式：模型会思考，但按它自己的方式
  - Few-Shot CoT：你给示例，模型会模仿你指定的分析框架和输出格式
```

Few-Shot 的价值从"触发推理"变成了 **"引导推理风格和格式"** 。

3. **小模型仍然需要 CoT**

Think 模式需要大规模 RL 训练，小模型（7B 以下）的 think 能力很弱。对这些模型来说，Few-Shot CoT 仍然是提升推理能力的有效手段。

##### 一句话总结

> Think 模式是 CoT 的"毕业形态"——把推理时的 prompt 技巧，通过训练固化到了模型参数里。
> 
> Zero-Shot CoT 基本被取代，Few-Shot CoT 从"触发推理"变成了"控制推理格式"的工具。

