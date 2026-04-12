from .rewrite_graph import GraphFlow,NodeFunc,State

from openai import OpenAI

client=OpenAI(base_url="http://100.64.0.4:8080/v1",api_key="1234567890")

MODEL="Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"


def node_entry(state:State):
    """初始化message"""
    state["messages"]=[
        {'role':'system','content':'你是一个智能助手,可以回答用户问题'},
        {'role':'user','content':state['user_input']}
    ]

# 意图识别节点
def node_intent(state:State):
    """意图识别节点,负责识别用户意图"""
    prompt="""你是一个意图分类器,
    
    """


def intent_embedding(state:State):
    intent_vec=
    