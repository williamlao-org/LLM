# Chain-of-Thought (CoT) 思维链 教程

## 一、什么是 CoT？

**Chain-of-Thought（思维链）** 是一种强大的 Prompt Engineering 技术，核心思想是：

> **让 LLM 在给出最终答案前，先"展示"它的推理过程** —— 就像你在数学考试中写出每一步解题步骤，而不是直接交出答案。

### 为什么需要 CoT？

LLM 本质上是 **next-token predictor**（下一个 token 的预测器）。当遇到复杂推理问题时：

| 方式 | 做法 | 效果 |
|------|------|------|
| **直接回答 (Standard)** | 问题 → 答案 | ❌ 容易出错，尤其是多步推理 |
| **CoT** | 问题 → 推理步骤1 → 步骤2 → ... → 答案 | ✅ 准确率大幅提升 |

**原理直觉**：每生成一步推理，就相当于给模型**增加了中间上下文**，让后续的 token 预测更准确。

---

## 二、CoT 的三种主要形式

### 1. Few-Shot CoT（少样本思维链）
最经典的方式 —— 在 prompt 中提供 **带推理过程的示例**：

```
Q: 小明有5个苹果，给了小红2个，又买了3个，现在有几个？
A: 让我们一步步思考。
   小明开始有 5 个苹果。
   给了小红 2 个，剩 5 - 2 = 3 个。
   又买了 3 个，变成 3 + 3 = 6 个。
   所以答案是 6。

Q: 一个农场有23只鸡，又买了2箱，每箱12只，总共多少只？
A: 让我们一步步思考。
   ...（模型会模仿示例格式来推理）
```

### 2. Zero-Shot CoT（零样本思维链）
最简单 —— 只需在 prompt 末尾加一句魔法咒语：

> **"Let's think step by step"**（让我们一步步思考）

```
Q: 一个农场有23只鸡，又买了2箱，每箱12只，总共多少只？
A: Let's think step by step.
```

这一句简单的话就能触发模型的推理能力！（来自 Google 2022 年的论文 "Large Language Models are Zero-Shot Reasoners"）

### 3. Auto-CoT（自动思维链）
自动生成 few-shot 示例中的推理步骤，减少人工标注的工作。

---

## 三、代码实现架构

我们的实现将包含以下文件：

```
CoT/
├── __init__.py
├── base.py          # 基础 CoT 类（定义接口）
├── few_shot_cot.py  # Few-Shot CoT 实现
├── zero_shot_cot.py # Zero-Shot CoT 实现
├── prompts.py       # Prompt 模板管理
└── examples.py      # 运行示例
```

### 核心流程

```mermaid
flowchart LR
    A[用户问题] --> B[Prompt 构造]
    B --> C{CoT 类型}
    C -->|Few-Shot| D[拼接示例 + 问题]
    C -->|Zero-Shot| E[问题 + 'Let us think step by step']
    D --> F[调用 LLM]
    E --> F
    F --> G[解析推理过程]
    G --> H[提取最终答案]
```

### 关键设计理念

1. **Prompt 构造** —— CoT 的核心就是如何构造 prompt，让模型输出推理链
2. **答案提取** —— 从模型的推理过程中提取最终答案
3. **模板可复用** —— 同一套模板可以应用于不同类型的问题

---

## 四、代码详解要点

### Few-Shot CoT 的关键

```python
# 核心就是把 "示例推理" 和 "新问题" 拼在一起
prompt = ""
for example in examples:
    prompt += f"Q: {example['question']}\n"
    prompt += f"A: {example['reasoning']}\n"  # ← 关键：包含推理过程
    prompt += f"Therefore, the answer is {example['answer']}.\n\n"

prompt += f"Q: {new_question}\n"
prompt += f"A: Let's think step by step.\n"
```

### Zero-Shot CoT 的关键

```python
# 就这么简单！
prompt = f"Q: {question}\nA: Let's think step by step.\n"
```

> [!IMPORTANT]
> CoT 不是一种模型架构或训练技术，它是一种 **Prompt 工程技术**。我们写的代码本质上是在做 **prompt 的构造与管理**，然后调用现有的 LLM API。

---

## 五、适用场景与局限

### ✅ CoT 擅长的
- 数学推理（算术、代数）
- 逻辑推理（演绎、归纳）
- 常识推理（日常推理、因果关系）
- 多步骤问题

### ❌ CoT 的局限
- 对于简单的事实查询无效（甚至可能更慢）
- 模型规模太小时效果不明显（通常需要 >100B 参数的模型才能明显获益）
- 推理链可能是"看起来正确"但实际逻辑有误的

---

## 六、论文参考

| 论文 | 年份 | 贡献 |
|------|------|------|
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | 2022 | 提出 Few-Shot CoT |
| [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | 2022 | 提出 Zero-Shot CoT |
| [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493) | 2022 | 提出 Auto-CoT |
