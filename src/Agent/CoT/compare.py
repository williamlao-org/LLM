from openai import OpenAI

client=OpenAI(
    base_url='http://100.64.0.4:8080/v1',
    api_key='1234567890'
)

MODEL='Qwen3.5-27B-Opus4.6-Q4_K_M.gguf'

prompt0='我想学习 prompt engineering, 请你教我。'
prompt1='''
我想系统学习 prompt engineering。请你用中文给我做一个入门教学，假设我是零基础。

要求：
1. 先解释什么是 prompt engineering；
2. 再讲它为什么重要；
3. 再讲最常见的 5 个技巧；
4. 每个技巧都给一个直观示例；
5. 最后给我 3 个练习题。

输出要求：
- 先讲结论，再讲原因；
- 尽量少用术语；
- 适合初学者理解。
'''

prompt2='''
我想学习 prompt engineering。请你作为教练，而不是只做知识介绍。

背景：
- 我是初学者；
- 我希望学完后能自己写出更清晰、更稳定的 prompt；
- 我的目标不是研究理论，而是实际能用。

请按下面方式教我：
1. 先用 200 字以内解释 prompt engineering 是什么；
2. 再告诉我一个好 prompt 的核心组成部分；
3. 然后从最基础的场景开始，给我 5 个由浅入深的例子；
4. 每个例子都要包含：
   - 一个模糊 prompt
   - 一个改进后的 prompt
   - 为什么这样改会更好
5. 最后给我 3 个练习，让我自己试着改写 prompt。

要求：
- 用中文；
- 先给结论，再解释；
- 不要讲空泛概念；
- 每个例子都要足够直观。
'''

prompt3='''
我想学习 prompt engineering，请你把我当成零基础学生，给我上一节 30 分钟的入门课。

教学目标：
学完后，我应该能做到这 3 件事：
1. 看出一个 prompt 为什么差；
2. 把模糊 prompt 改写得更清晰；
3. 针对写作、总结、分析这 3 类任务，写出基本可用的 prompt。

教学方式：
请按“概念最少、例子最多”的方式进行，结构如下：

第 1 部分：用最简单的话解释什么是 prompt engineering
- 控制在 150 字以内
- 不要堆术语

第 2 部分：告诉我一个好 prompt 最关键的 4 个要素
- 例如：目标、上下文、约束、输出格式
- 每个要素用一句话解释

第 3 部分：给我 6 个直观示例
每个示例都必须包含：
1. 原始模糊 prompt
2. 改进版 prompt
3. 改进思路
4. 改进前后效果差异

示例场景至少覆盖：
- 写作
- 总结
- 信息提取
- 分析判断
- 改写
- 学习辅导

第 4 部分：给我一个通用模板
要求这个模板能让我以后自己套用。

第 5 部分：给我 5 道练习题
要求：
- 从简单到难
- 不要直接给答案
- 只告诉我每题要优化的重点

表达要求：
- 用中文
- 先给结论，再给理由
- 多给反例和边界条件
- 不要写成宣传文
- 尽量让我看完就能立刻上手
'''

prompt4='''
我想学习 prompt engineering，请你用互动教学的方式教我，假设我是零基础。

目标：
让我通过练习学会，而不是只看概念介绍。

教学规则：
1. 每次只教一个小点，不要一次讲太多；
2. 每个小点都要先给一个直观例子；
3. 讲完后立刻给我一道小练习；
4. 等我回答后，你再指出问题并继续下一步；
5. 重点训练我把模糊 prompt 改成清晰 prompt 的能力。

课程内容至少包括：
- 什么是 prompt engineering
- 一个好 prompt 的核心组成
- 常见错误有哪些
- 如何补充约束
- 如何指定输出格式
- 如何给示例
- 如何拆解复杂任务

要求：
- 用中文
- 先说结论，再解释
- 少讲空理论
- 多指出我 prompt 里的漏洞
- 有信息不足时直接说明，不要替我脑补
'''

res0=client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':prompt0}
    ]
)

res1=client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':prompt1}
    ]
)

res2=client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':prompt2}
    ]
)

res3=client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':prompt3}
    ]
)

res4=client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':prompt4}
    ]
)

with open('result.md','w',encoding='utf-8') as f:
    f.write(f"# Prompt Engineering 对比\n\n")
    f.write(f"# 原始问题\n\n{prompt0}\n\n")
    f.write(f"# 简单 Prompt\n\n{prompt1}\n\n")
    f.write(f"# 结构化 Prompt\n\n{prompt2}\n\n")
    f.write(f"# 详细 Prompt\n\n{prompt3}\n\n")
    f.write(f"# 互动 Prompt\n\n{prompt4}\n\n")
    f.write(f"# 原始回答\n\n{res0.choices[0].message.content}\n\n")
    f.write(f"# 简单回答\n\n{res1.choices[0].message.content}\n\n")
    f.write(f"# 结构化回答\n\n{res2.choices[0].message.content}\n\n")
    f.write(f"# 详细回答\n\n{res3.choices[0].message.content}\n\n")
    f.write(f"# 互动回答\n\n{res4.choices[0].message.content}\n\n")