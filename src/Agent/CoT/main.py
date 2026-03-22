from openai import OpenAI

client=OpenAI(
    base_url='http://100.64.0.4:8080/v1',
    api_key='1234567890'
)

MODEL='Qwen3.5-27B-Opus4.6-Q4_K_M.gguf'

def print_response(type,prompt,answer):
    print("="*30+type+'='*30)
    print("问题：\n",prompt)
    print("\n")
    print("回答：\n",answer)
    print("="*60)

# 简单的对话
response=client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':'Hello'}
    ]
)


print_response('简单的对话', 'Hello', response.choices[0].message.content)

# 设计 Few-Shot CoT Prompt：提供 [问题 -> 思考过程 -> 答案] 的范本

from .prompts import few_shot_prompt
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':few_shot_prompt}
    ]
)

print_response('Few-Shot CoT', few_shot_prompt, response.choices[0].message.content)

# 设计 Zero-Shot CoT Prompt：提供 [问题 -> 思考过程 -> 答案] 的范本

from .prompts import zero_shot_prompt
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {'role':'user','content':zero_shot_prompt}
    ]
)

print_response('Zero-Shot CoT', zero_shot_prompt, response.choices[0].message.content)