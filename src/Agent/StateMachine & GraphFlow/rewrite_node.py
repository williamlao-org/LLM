from .rewrite_graph import GraphFlow,NodeFunc,State

from openai import OpenAI

client=OpenAI(base_url="http://100.64.0.4:8080/v1",api_key="1234567890")

MODEL="Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

def node_llm(state:State):
    