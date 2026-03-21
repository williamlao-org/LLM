from openai import OpenAI

client=OpenAI(
    base_url='http://100.64.0.4:8080/v1',
    api_key='1234567890'
)

response=client.chat.completions.create(
    model='Qwen3.5-27B-Opus4.6-Q4_K_M.ggufB',
    messages=[
        {'role':'user','content':'Hello'}
    ]
)

print(response.choices[0].message.content)

# 设计 Few-Shot CoT Prompt：提供 [问题 -> 思考过程 -> 答案] 的范本
cot_prompt = """
请解决以下逻辑数学问题。

示例 1：
问题：餐厅原本有 50 个苹果。厨师用了 20 个做派，然后又买了 10 个。现在有几个？
思考过程：
1. 初始苹果数量：50
2. 用掉 20 个，剩下：50 - 20 = 30
3. 又买 10 个，总计：30 + 10 = 40
答案：40

示例 2：
问题：Roger 有 5 个网球。他又买了两罐网球，每罐有 3 个。他现在有几个网球？
思考过程：
1. 初始网球数量：5
2. 买了两罐，每罐 3 个，所以新增数量：2 * 3 = 6
3. 最终总计：5 + 6 = 11
答案：11

真实问题：
问题：一家书店原本有 120 本科幻小说。上午卖出了总数的四分之一，下午又进货了 45 本。接着，店员将剩下的科幻小说平均分到 3 个书架上。请问每个书架上有几本？
思考过程：
"""

response = client.chat.completions.create(
    model='Qwen3.5-27B-Opus4.6-Q4_K_M.ggufB',
    messages=[
        {'role':'user','content':cot_prompt}
    ]
)

print(response.choices[0].message.content)