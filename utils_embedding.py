# utils_embedding.py
import numpy as np, requests, json
from tqdm import tqdm
from openai import OpenAI



import os

client = OpenAI(
    # 密钥仅从环境变量读取；请先 export DASHSCOPE_API_KEY=sk-...
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    base_url=os.environ.get(
        "EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
)


# def embed_sentences(sentences: list[str]) -> np.ndarray:
#     embeddings = []
#     for sent in tqdm(sentences, desc="生成嵌入向量"):
#         '''使用阿里嵌入'''
#         completion = client.embeddings.create(
#             model="text-embedding-v4",
#             input=sent,
#             dimensions=768, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#             encoding_format="float"
#         )
        
#         # print(completion.data[0].embedding)
#         emb = completion.data[0].embedding
#         embeddings.append(np.array(emb, dtype=np.float32))
#     return np.vstack(embeddings)

def embed_sentences(sentences: list[str]) -> np.ndarray:
    """
    分批嵌入，每批 ≤ 10 句，返回 768 维向量。
    """
    embeddings = []
    for i in tqdm(range(0, len(sentences), 10), desc="分批嵌入"):
        batch = sentences[i:i + 10]   # 每批 ≤ 10 句
        print(batch)
        resp = client.embeddings.create(
            model="text-embedding-v4",
            input=batch,
            dimensions=768,
            encoding_format="float"
        )
        embeddings.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
    return np.vstack(embeddings)
