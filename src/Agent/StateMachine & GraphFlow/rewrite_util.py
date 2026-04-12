from .rewrite_node import client

def cosine_similarity(vec1, vec2):
    dot=sum(a*b for a,b in zip(vec1,vec2))
    norm_a=sum(a**2 for a in vec1)**0.5
    norm_b=sum(b**2 for b in vec2)**0.5
    return dot/(norm_a*norm_b)

def get_embedding(text:str):
    response=client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding