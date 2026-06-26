# Phase 1: 经典 RAG —— 从零理解检索增强生成

## 1. RAG 为什么存在？

LLM 有三个根本性的局限：

| 问题 | 表现 | 例子 |
|:---|:---|:---|
| **知识截止** | 训练数据有截止日期，不知道最新信息 | "2026年6月的新闻是什么？" |
| **幻觉** | 不知道的东西也会编 | 编造不存在的论文、代码库 |
| **缺乏私域知识** | 不知道你公司的内部文档 | "我们的部署流程是什么？" |

RAG 的思路非常朴素：**既然 LLM 自己知识不够，那就在回答之前先帮它查资料。**

```
没有 RAG:
  用户提问 ──→ LLM（靠自己的知识回答）──→ 可能瞎编

有 RAG:
  用户提问 ──→ 去知识库检索相关资料 ──→ 把资料 + 问题一起给 LLM ──→ 基于资料回答
```

本质上，RAG = **给 LLM 开卷考试**。

---

## 2. RAG 完整 Pipeline

```
┌─────────────────── 离线阶段（索引构建）──────────────────┐
│                                                          │
│  📄 文档集合                                              │
│      │                                                   │
│      ▼                                                   │
│  🔪 分块 (Chunking)                                      │
│      │  把长文档切成小段                                   │
│      ▼                                                   │
│  🧮 Embedding                                            │
│      │  把文本转成向量（数字表示）                          │
│      ▼                                                   │
│  💾 存入向量数据库                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────── 在线阶段（查询回答）──────────────────┐
│                                                          │
│  ❓ 用户提问                                              │
│      │                                                   │
│      ▼                                                   │
│  🧮 Query Embedding                                     │
│      │  把问题也转成向量                                  │
│      ▼                                                   │
│  🔍 相似度检索                                            │
│      │  在向量库中找最相似的 chunks                        │
│      ▼                                                   │
│  📝 构造 Prompt                                          │
│      │  把检索到的内容 + 用户问题拼成 Prompt               │
│      ▼                                                   │
│  🤖 LLM 生成回答                                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

下面逐个拆解每个环节。

---

## 3. 文档加载 (Document Loading)

这一步比较直观：把各种格式的文件读进来变成纯文本。

```python
# 最简单的情况：读 txt/md 文件
with open("doc.md", "r") as f:
    text = f.read()

# PDF 需要专门的库
# 常用：PyPDF2, pdfplumber, PyMuPDF(fitz)
import pdfplumber
with pdfplumber.open("doc.pdf") as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)
```

> [!NOTE]
> 文档加载看起来简单，但在生产环境中是坑最多的地方。PDF 的表格、图片、多栏排版都会导致解析错乱。不过 Phase 1 我们先用简单的 txt/md 文件，不纠结这些。

---

## 4. 文本分块 (Chunking) —— 最被低估的环节

### 为什么要分块？

两个原因：
1. **LLM 上下文窗口有限**：不能把整个知识库塞进 Prompt
2. **检索粒度**：整篇文档太大，检索命中但大部分内容无关；需要切成小块，精确命中相关段落

### 分块策略

#### 策略 1：固定大小分块 (Fixed Size)
```
chunk_size = 500  # 500个字符一块
overlap = 50      # 相邻块之间重叠50个字符

文档: [AAAAAA|BBBBBBB|CCCCCCC|DDDDDDD]
            ↓
Chunk 1: [AAAAAA + BB]   ← 500字符 + 50字符重叠
Chunk 2: [BB + BBBBBBB]  ← 重叠部分保证上下文不断裂
Chunk 3: [BB + CCCCCCC]
...
```

**优点**：简单  
**缺点**：可能在句子中间切断，破坏语义

#### 策略 2：递归字符分割 (Recursive Character Splitting)
```
按优先级依次尝试：
1. 先按 "\n\n"（段落）分割
2. 如果块太大，再按 "\n"（换行）分割
3. 还是太大，按 ". "（句子）分割
4. 最后按空格分割
```

**优点**：尽量保持语义完整性  
**缺点**：块大小不均匀

#### 策略 3：句子分割 (Sentence Splitting)
```
用 NLP 工具（如 spaCy、NLTK）按句子边界切分
然后把相邻的几个句子合并成一个 chunk
```

### 分块的关键参数

```
chunk_size:  每块的大小（字符数或 token 数）
             太小 → 上下文不足，碎片化
             太大 → 检索精度下降，噪声多
             通常 200-1000 tokens

overlap:     相邻块的重叠量
             目的是防止信息在块边界处丢失
             通常 chunk_size 的 10-20%
```

> [!TIP]
> 经验法则：先用 `chunk_size=500, overlap=50` 开始，后续根据评估结果调。没有"最佳"参数，取决于你的文档类型和查询方式。

---

## 5. Embedding（文本向量化）—— 核心中的核心

### 什么是 Embedding？

把文本转成一个**高维向量（一组浮点数）**，使得语义相似的文本在向量空间中距离相近。

```
"今天天气真好"  →  [0.12, -0.34, 0.78, ..., 0.56]   (1536维)
"天气不错"      →  [0.11, -0.32, 0.80, ..., 0.55]   ← 语义相似，向量接近！
"量子力学导论"  →  [0.89, 0.45, -0.23, ..., -0.67]  ← 语义不同，向量远离
```

### 为什么能用向量表示语义？

Embedding 模型（如 text-embedding-3-small）是在海量文本对上训练的：
- 训练目标：让"意思相近"的文本对应的向量靠近
- 训练数据：(问题, 答案)、(标题, 正文)、(查询, 相关文档) 等配对
- 结果：模型学会了把语义编码到向量空间中

### 常用 Embedding 模型

| 模型 | 维度 | 特点 |
|:---|:---|:---|
| OpenAI `text-embedding-3-small` | 1536 | 性价比高，API 调用 |
| OpenAI `text-embedding-3-large` | 3072 | 精度更高，更贵 |
| BGE (BAAI) | 768/1024 | 开源，中文效果好 |
| Jina Embeddings | 768 | 开源，支持长文本 |

### 代码示例

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    """将文本转成向量"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# 使用
vec = get_embedding("什么是机器学习？")
print(f"向量维度: {len(vec)}")  # 1536
print(f"前5个值: {vec[:5]}")    # [0.012, -0.034, ...]
```

---

## 6. 向量数据库 (Vector Store)

### 为什么需要向量数据库？

普通数据库做精确匹配（SQL WHERE）。但向量搜索需要做的是：
- 在百万个向量中，找到和查询向量**最相似**的 Top-K 个

暴力遍历太慢（O(n)），需要专门的索引结构来加速近似最近邻搜索 (ANN)。

### 常用选择

| 工具 | 类型 | 特点 | 适合场景 |
|:---|:---|:---|:---|
| **Chroma** | 嵌入式 | 轻量，像 SQLite 一样简单 | 学习、原型 |
| **FAISS** | 库 | Meta 开源，极高性能 | 大规模、注重性能 |
| **Milvus** | 服务端 | 分布式，云原生 | 生产部署 |
| **Pinecone** | 云服务 | 全托管 | 不想自己运维 |

Phase 1 我们用 **Chroma**，最简单。

### 相似度度量

最常用的是**余弦相似度 (Cosine Similarity)**：

```
         A · B          向量 A 和 B 的点积
cos(θ) = ──────  =  ─────────────────────────
         |A|·|B|       两个向量的模的乘积

范围：[-1, 1]
  1  → 完全相同方向（最相似）
  0  → 完全正交（无关）
 -1  → 完全相反方向
```

> [!NOTE]
> 其他度量方式还有欧氏距离 (L2)、点积 (Inner Product)。对于归一化后的向量，余弦相似度和点积等价。Chroma 默认使用 L2 距离。

---

## 7. 检索与生成 (Retrieve & Generate)

### 检索过程

```python
# 1. 用户问题 → Embedding
query_vec = get_embedding("什么是反向传播？")

# 2. 在向量库中搜索最相似的 Top-K 个 chunk
results = vector_db.query(query_vec, top_k=3)

# 3. 拿到的 results 就是最相关的文档片段
# [
#   {"text": "反向传播是神经网络训练的核心算法...", "score": 0.92},
#   {"text": "通过链式法则计算梯度...", "score": 0.87},
#   {"text": "误差从输出层反向传播到输入层...", "score": 0.85},
# ]
```

### 构造 Prompt

```python
context = "\n\n".join([r["text"] for r in results])

prompt = f"""基于以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请说"我没有找到相关信息"。
不要编造参考资料中没有的内容。

参考资料：
{context}

用户问题：{query}

回答："""
```

> [!IMPORTANT]
> Prompt 设计的关键点：
> 1. 明确告诉 LLM "基于参考资料回答"，减少幻觉
> 2. 给 LLM 一个"不知道就说不知道"的退路
> 3. 把 context 放在 question 前面（LLM 对开头和结尾的注意力更强）

---

## 8. 动手项目架构

我们要搭建的项目结构：

```
RAG/
├── main.py              # 主程序：交互式问答
├── document_loader.py   # 文档加载
├── chunker.py           # 文本分块
├── embedder.py          # Embedding 封装
├── vector_store.py      # 向量库操作
├── rag_chain.py         # RAG 主流程（检索 + 生成）
├── config.py            # 配置（API key、参数等）
└── docs/                # 你的知识库文档
    ├── doc1.md
    ├── doc2.txt
    └── ...
```

### 不用框架，手写每个模块

为什么不用 LangChain？因为：
1. 手写一遍你才真正理解每个环节在做什么
2. LangChain 封装太重，出了问题不知道怎么 debug
3. 理解原理后再看框架，会有"原来如此"的感觉

---

## 接下来

概念讲完了，准备开始写代码。我们会按照这个顺序逐步实现：

1. `config.py` — 基础配置
2. `document_loader.py` — 加载文档
3. `chunker.py` — 分块策略
4. `embedder.py` — Embedding 封装
5. `vector_store.py` — 向量存储与检索
6. `rag_chain.py` — 串联完整流程
7. `main.py` — 交互式问答入口
