import logging, httpx
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

client = OpenAI(
    api_key=os.getenv("OPENAI_APIKEY"), base_url="https://api.siliconflow.cn/v1"
)
response = client.chat.completions.create(
    model="zai-org/GLM-4.6",
    messages=[
        {"role": "system", "content": "你是数据分析专家"},
        {"role": "user", "content": "你好"},
    ],
    temperature=0.7,
    max_tokens=4096,
    stream=True,
)

for event in response:
    delta = event.choices[0].delta

    # 推理内容（如果有）
    if getattr(delta, "reasoning_content", None):
        print(delta.reasoning_content, end="", flush=True)

    # 可见内容
    if getattr(delta, "content", None):
        print(delta.content, end="", flush=True)