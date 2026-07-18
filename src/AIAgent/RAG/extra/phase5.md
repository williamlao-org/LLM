# Phase 5：GraphRAG — 用知识图谱让 RAG 学会"关系推理"

> 本文是学习笔记 + 「新对话快速启动器」。
> 新开会话学 Phase 5 时，把这份文件喂给我即可立刻接上下文，无需重新铺垫。

---

## 0. 我现在在哪（进度锚点）

```
Phase 1 ✅ 经典 RAG（document_loader / chunker / embedder / vector_store / rag_chain / main）
Phase 2 ✅ 进阶 RAG（Hybrid Search / Reranking / Query Rewriting / RAGAS 评估）
Phase 3 ✅ Agentic RAG（Self-RAG / CRAG / 多知识库路由 / 多跳检索）
Phase 4 ✅ 记忆系统（Working / Token / Summary / Structured / Episodic / Semantic Memory）
Phase 5 ✅ GraphRAG
   ├─ 第一块 ✅ 知识图谱基础与 LLM 抽取
   ├─ 第二块 ✅ 图索引构建（社区检测 + 社区摘要）
   ├─ 第三块 ✅ Local Search（实体中心检索）
   ├─ 第四块 ✅ Global Search（社区摘要 Map-Reduce）
   └─ 第五块 ✅ Hybrid：向量检索 + 图检索融合
Phase 6  ⬜ 综合系统
```

学习方式偏好：**不上来就用框架**，先手写理解原理；小步推进；对比驱动（A/B）。

---

## 1. 一句话与动机

**GraphRAG = 知识图谱 + RAG。用图结构捕捉实体之间的关系，解决向量检索"只懂相似、不懂关联"的死穴。**

### 1.1 向量 RAG 到底在哪翻车？

你 Phase 1–3 搭的向量 RAG，本质做的一件事：**找到和问题"语义最像"的文本片段**。

这在大多数场景够用，但有一类问题它系统性地搞不定：

| 问题类型           | 示例                                                 | 向量 RAG 为什么翻车                                  |
| :----------------- | :--------------------------------------------------- | :--------------------------------------------------- |
| **关系查询** | "张三的导师是谁？他导师又在哪家公司？"               | 答案分散在两个不同 chunk，余弦相似度不会把它们串起来 |
| **聚合查询** | "这篇论文里提到的所有数据集有哪些？"                 | 向量检索 top-k 只返回"最像"的几段，不保证覆盖全      |
| **全局主题** | "这 100 篇文档的核心主题是什么？"                    | 向量检索是局部的，它根本不看全局                     |
| **多跳推理** | "A 投资了 B，B 收购了 C，那 A 和 C 有什么间接关系？" | 每一跳的信息在不同 chunk，向量检索无法沿路径追溯     |

用一句话总结翻车原因：

> **向量检索是"点对点"的相似度匹配，它不理解实体之间的"边"（关系）。知识图谱天生就是用来存"边"的。**

### 1.2 用人类类比

```
向量 RAG ≈ 在图书馆里找"和你问题最像的那几页"
GraphRAG ≈ 先画一张人物关系图 / 概念地图，然后沿着关系线去追溯

你问"红楼梦里贾宝玉和薛宝钗什么关系？"
  向量 RAG：找到几段提到他俩的文字，拼在一起碰运气
  GraphRAG：直接在关系图上走 贾宝玉 —[婚姻]→ 薛宝钗，一步到位
```

---

## 2. 知识图谱基础：60 秒搞懂

### 2.1 核心数据结构：三元组

知识图谱的最小单位是**三元组 (Triple)**：

```
(实体A) —[关系]→ (实体B)
(Entity) —[Relation]→ (Entity)

例子：
(贾宝玉) —[父亲]→ (贾政)
(贾宝玉) —[婚姻]→ (薛宝钗)
(DeepSeek) —[开发了]→ (DeepSeek-V3)
(DeepSeek-V3) —[属于]→ (大语言模型)
```

多个三元组连起来就形成了**图**：

```
                    ┌──[父亲]──→ 贾政 ──[妻子]──→ 王夫人
                    │
  薛宝钗 ←─[婚姻]─ 贾宝玉
                    │
                    └──[表亲]──→ 林黛玉 ──[父亲]──→ 林如海
```

### 2.2 和向量库的本质区别

| 维度     | 向量库 (Vector Store)       | 知识图谱 (Knowledge Graph) |
| :------- | :-------------------------- | :------------------------- |
| 存储单位 | 文本 chunk + embedding 向量 | 实体节点 + 关系边          |
| 查询方式 | 余弦相似度 top-k            | 图遍历（沿边走）+ 子图匹配 |
| 擅长     | "找到最像的内容"            | "找到有关联的内容"         |
| 不擅长   | 关系推理、全局聚合          | 模糊语义匹配               |
| 数据结构 | 平坦的（所有 chunk 平级）   | 层次化的（社区、子图）     |

**关键认知：两者不是替代关系，而是互补关系。最终目标是 Hybrid。**

### 2.3 实体 (Entity) vs 关系 (Relation) vs 属性 (Attribute)

```
实体 (Entity)：现实世界中的"东西"
  → 人、组织、地点、概念、事件、产品...
  → 图中的"节点"

关系 (Relation)：两个实体之间的"联系"
  → 工作于、发明了、位于、属于、父亲...
  → 图中的"边"

属性 (Attribute)：实体或关系自身的附加信息
  → 实体属性：DeepSeek-V3.参数量 = "671B"
  → 关系属性：(张三)—[工作于 {since: 2020}]→(Google)
```

---

## 3. Microsoft GraphRAG：当前最重要的架构

2024 年微软发布了 GraphRAG 论文和开源项目，定义了这个领域最主流的架构。核心论文：*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*。

### 3.1 整体流程一图看懂

```
                        ┌─────────────────────────────┐
                        │       原始文档集合            │
                        └──────────┬──────────────────┘
                                   │
                    ═══════════════╪══════════════════
                    ║   索引阶段（离线，构建一次）      ║
                    ═══════════════╪══════════════════
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Step 1: 文本分块              │
                    │  和你 Phase 1 做的一样         │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Step 2: LLM 实体/关系抽取     │
                    │  每个 chunk → 三元组列表        │
                    │  "从这段文字中提取所有实体和关系" │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Step 3: 构建知识图谱           │
                    │  所有三元组合并 → 一张大图       │
                    │  实体去重、关系合并              │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Step 4: 社区检测               │
                    │  Leiden 算法把图分成"社区"       │
                    │  = 紧密相连的实体群组           │
                    └──────────┬───────────────────┘
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  Step 5: 社区摘要               │
                    │  LLM 为每个社区生成一段摘要     │
                    │  "这群实体讲的是什么主题？"      │
                    └──────────┬───────────────────┘
                               │
                    ═══════════╪════════════════════
                    ║   查询阶段（在线，每次请求）    ║
                    ═══════════╪════════════════════
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │  用户提问                      │
                    │  路由到 Local 或 Global Search  │
                    └──────────┬───────────────────┘
                        ┌──────┴──────┐
                        ▼             ▼
                   Local Search   Global Search
                   (实体追溯)     (社区摘要聚合)
```

### 3.2 两种检索模式的直觉

这是 GraphRAG 最核心的设计——**一套索引支撑两种完全不同的查询模式**：

```
Local Search（局部搜索）
  问题类型：具体的、有明确实体的问题
  例："DeepSeek-V3 用了什么训练策略？"
  做法：找到实体 "DeepSeek-V3" → 沿关系边收集相关信息 → 生成回答

Global Search（全局搜索）
  问题类型：宏观的、需要全局视角的问题
  例："这些论文的研究趋势是什么？"
  做法：读取所有社区摘要 → Map-Reduce → 综合回答
```

| 维度     | Local Search                 | Global Search                |
| :------- | :--------------------------- | :--------------------------- |
| 问题类型 | 具体、聚焦、有明确实体       | 宏观、全局、主题性           |
| 入口     | 从匹配的实体节点出发         | 从所有社区摘要出发           |
| 数据范围 | 目标实体 + 邻居 + 相关 chunk | 全部社区摘要                 |
| 类比     | 查字典：查"贾宝玉"词条       | 读目录：看整本书讲了哪些主题 |
| 优势     | 精准、快速、上下文紧凑       | 能回答"整体是什么"           |
| 劣势     | 看不到全局                   | 依赖摘要质量、token 开销大   |

---

## 4. 索引构建：手把手拆解每一步

### 4.1 Step 1 — 文本分块（你已经会了）

和 Phase 1 的分块一模一样，通常 chunk_size 稍大一些（比如 1200 token），因为 LLM 抽取实体时需要足够的上下文。

### 4.2 Step 2 — LLM 实体/关系抽取（核心步骤）

这是 GraphRAG 和传统知识图谱最大的区别：**不用 NER + 规则，直接让 LLM 干**。

#### 抽取 Prompt 的设计思路

```python
EXTRACT_PROMPT = """
-Goal-
给定一段可能与该活动相关的文本文档和一个实体类型列表，
从文本中识别出这些类型的所有实体以及已识别实体之间的所有关系。

-Steps-
1. 识别所有实体。对于每个已识别的实体，提取以下信息：
   - entity_name: 实体名称（大写）
   - entity_type: 实体类型，从以下列表中选择：[{entity_types}]
   - entity_description: 对该实体的属性和活动的全面描述

2. 从步骤 1 中识别的实体中，识别出所有*明确相关*的 (source_entity, target_entity) 对。
   对于每对相关实体，提取以下信息：
   - source_entity: 源实体名称
   - target_entity: 目标实体名称
   - relationship_description: 解释为什么认为源实体和目标实体相关
   - relationship_strength: 1-10 的数值，表示关系强度

3. 以 JSON 列表形式输出所有实体和关系。

-Real Data-
Entity_types: {entity_types}
Text: {input_text}

Output:
"""
```

#### 抽取结果示例

输入文本：

```
DeepSeek 公司在 2024 年发布了 DeepSeek-V3 模型，采用了 Mixture of Experts (MoE) 架构。
该模型拥有 671B 总参数，其中每个 token 激活 37B 参数。训练使用了 14.8T token 的数据。
```

LLM 抽取输出：

```json
{
  "entities": [
    {"name": "DEEPSEEK", "type": "ORGANIZATION", "description": "一家 AI 公司，开发大语言模型"},
    {"name": "DEEPSEEK-V3", "type": "MODEL", "description": "DeepSeek 公司在 2024 年发布的大语言模型，拥有 671B 总参数"},
    {"name": "MIXTURE OF EXPERTS", "type": "TECHNOLOGY", "description": "一种模型架构，允许每个 token 只激活部分参数"}
  ],
  "relationships": [
    {"source": "DEEPSEEK", "target": "DEEPSEEK-V3", "description": "DeepSeek 公司开发并发布了 DeepSeek-V3", "strength": 9},
    {"source": "DEEPSEEK-V3", "target": "MIXTURE OF EXPERTS", "description": "DeepSeek-V3 采用了 MoE 架构", "strength": 8}
  ]
}
```

#### 关键工程细节

```
⚠️ LLM 抽取不是一次就够的！

问题：一次抽取经常遗漏实体（尤其是 chunk 末尾的、隐含的）
解法：Gleaning（二次收割）

做法：抽取完后再问一次 LLM：
  "上一轮你可能遗漏了一些实体，请仔细检查原文，提取遗漏的实体和关系。"
  → 通常做 1-2 轮 gleaning，再多收益递减
```

### 4.3 Step 3 — 构建知识图谱（合并去重）

多个 chunk 会抽取出重复或近似的实体，需要合并：

```
Chunk 1 抽到: (DEEPSEEK, ORGANIZATION, "一家AI公司")
Chunk 5 抽到: (DEEPSEEK, ORGANIZATION, "深度求索，中国AI公司，成立于2023年")
Chunk 9 抽到: (DEEP SEEK, ORGANIZATION, "AI公司")

合并策略：
  1. 名称归一化：大写 + 去空格 → DEEPSEEK
  2. 描述合并：让 LLM 把多段描述综合成一段
  3. 关系去重：相同 (source, target) 的关系合并描述

合并后：
  DEEPSEEK (ORGANIZATION): "深度求索，中国 AI 公司，成立于 2023 年，开发大语言模型"
```

用 Python 数据结构表示：

```python
from dataclasses import dataclass, field

@dataclass
class Entity:
    name: str                          # 归一化名称
    entity_type: str                   # PERSON / ORG / TECH / ...
    description: str                   # LLM 生成的综合描述
    source_chunk_ids: list[str]        # 来自哪些 chunk（溯源用）

@dataclass
class Relationship:
    source: str                        # 源实体名称
    target: str                        # 目标实体名称
    description: str                   # 关系描述
    strength: float                    # 关系强度
    source_chunk_ids: list[str]

@dataclass
class KnowledgeGraph:
    entities: dict[str, Entity]        # name → Entity
    relationships: list[Relationship]
```

### 4.4 Step 4 — 社区检测（Leiden 算法）

这是 GraphRAG 最精妙的设计之一。直觉：

```
一张大图里，有些实体彼此联系很紧密，形成"小圈子"。
社区检测就是自动找出这些小圈子。

       社区 A                社区 B
    ┌───────────┐        ┌───────────┐
    │ 贾宝玉     │        │ 刘姥姥     │
    │  ↕    ↕   │ ─弱─→ │  ↕    ↕   │
    │ 林黛玉  薛宝钗│      │ 王熙凤  贾母 │
    └───────────┘        └───────────┘
    主题：宝黛钗三角恋      主题：贾府社交与管理
```

#### 为什么用 Leiden 算法？

| 算法             | 特点             | 为什么选/不选                        |
| :--------------- | :--------------- | :----------------------------------- |
| Louvain          | 经典社区检测     | 可能产生碎片化的"坏"社区             |
| **Leiden** | Louvain 的改进版 | **保证社区内部连通**，质量更好 |
| Spectral         | 基于矩阵分解     | 需要预设社区数，不适合               |

#### 层次化社区

Leiden 支持**多层级**社区检测：

```
Level 0（最细粒度）：每 3-5 个紧密实体为一组
Level 1（中粒度）：多个 Level 0 社区合并
Level 2（最粗粒度）：全局大主题

Level 2:  ┌── 贾府家族 ──────────────────────┐
          │                                   │
Level 1:  ├── 核心人物关系 ──┤ ├── 家族管理 ──┤
          │                  │ │              │
Level 0:  ├ 宝黛钗 ┤ ├ 元妃省亲 ┤ ├ 管家体系 ┤

Local Search 用细粒度层（Level 0-1）
Global Search 用粗粒度层（Level 1-2）
```

#### 代码层面怎么做

```python
import networkx as nx

# 手写版不需要安装 igraph，用 networkx + graspologic
# pip install graspologic  (微软出的图分析库，包含 Leiden)

from graspologic.partition import hierarchical_leiden

# 1. 构建 networkx 图
G = nx.Graph()
for entity in knowledge_graph.entities.values():
    G.add_node(entity.name, type=entity.entity_type, desc=entity.description)

for rel in knowledge_graph.relationships:
    G.add_edge(rel.source, rel.target, weight=rel.strength, desc=rel.description)

# 2. Leiden 社区检测
community_mapping = hierarchical_leiden(G, max_cluster_size=10)
# 返回 [(node, community_id, level), ...]
```

### 4.5 Step 5 — 社区摘要

对每个社区，把其中所有实体、关系、相关 chunk 文本喂给 LLM，让它生成一段摘要：

```python
COMMUNITY_SUMMARY_PROMPT = """
你是一个数据分析师，负责对知识图谱中的一个社区进行摘要。

社区中包含以下实体和关系：

实体：
{entities_text}

关系：
{relationships_text}

相关原文片段：
{source_chunks_text}

请生成一段全面的摘要，说明这个社区的核心主题、关键发现和重要关系。
摘要应该能帮助回答关于该社区主题的宏观问题。
"""
```

输出示例：

```
社区摘要 #7 — DeepSeek 模型家族
此社区围绕 DeepSeek 公司及其系列模型。DeepSeek 是一家中国 AI 公司，
先后发布了 DeepSeek-V2、DeepSeek-V3 和 DeepSeek-R1 等模型。
核心技术路线包括 Mixture of Experts (MoE) 架构和强化学习训练。
DeepSeek-V3 采用 671B 参数的 MoE 架构，训练成本显著低于同等规模模型...
```

---

## 5. 查询阶段：Local Search 详解

### 5.1 流程

```
用户问题："DeepSeek-V3 的训练用了什么技术？"
           │
           ▼
  ① 实体识别：从问题中提取关键实体
     → ["DEEPSEEK-V3"]
           │
           ▼
  ② 实体匹配：在图中找到对应节点
     → 精确匹配 or embedding 相似度匹配
           │
           ▼
  ③ 子图提取：从匹配实体出发，沿关系边走 1-2 跳
     → DEEPSEEK-V3 —[采用]→ MOE
     → DEEPSEEK-V3 —[使用]→ FP8_TRAINING
     → DEEPSEEK-V3 —[开发者]→ DEEPSEEK
     → DEEPSEEK-V3 所属社区的摘要
           │
           ▼
  ④ 上下文组装：实体描述 + 关系描述 + 社区摘要 + 源 chunk
           │
           ▼
  ⑤ LLM 生成：基于图上下文生成回答
```

### 5.2 关键代码逻辑

```python
def local_search(query: str, graph: KnowledgeGraph, top_k: int = 10):
    # 1. 从问题中提取实体（可以用 LLM 或简单的名称匹配）
    query_entities = extract_entities_from_query(query)
  
    # 2. 在图中找到匹配的实体节点
    matched_nodes = []
    for qe in query_entities:
        # 精确匹配
        if qe.upper() in graph.entities:
            matched_nodes.append(graph.entities[qe.upper()])
        else:
            # 退路：用 embedding 相似度匹配最近的实体
            matched_nodes.extend(embedding_match(qe, graph.entities, top_k=3))
  
    # 3. 沿边扩展 1-2 跳，收集相关实体和关系
    context_entities = set()
    context_relations = []
    for node in matched_nodes:
        context_entities.add(node)
        for rel in graph.relationships:
            if rel.source == node.name or rel.target == node.name:
                context_relations.append(rel)
                # 把关系另一端的实体也加进来（1 跳）
                other = rel.target if rel.source == node.name else rel.source
                if other in graph.entities:
                    context_entities.add(graph.entities[other])
  
    # 4. 收集相关社区摘要
    community_summaries = get_community_summaries_for_entities(context_entities)
  
    # 5. 收集源 chunk 文本（溯源）
    source_chunks = get_source_chunks(context_entities, context_relations)
  
    # 6. 组装上下文
    context = format_graph_context(
        entities=context_entities,
        relations=context_relations,
        community_summaries=community_summaries,
        source_chunks=source_chunks
    )
  
    return context
```

### 5.3 Local Search 上下文的优先级排序

上下文窗口有限，不能什么都塞进去。排序策略：

```
优先级从高到低：
  1. 匹配实体自身的描述（最直接相关）
  2. 直接关系的描述（一跳关系）
  3. 匹配实体所在社区的摘要（提供背景）
  4. 关系另一端实体的描述（补充信息）
  5. 源 chunk 原文（兜底溯源）

每一级别内部按 relationship_strength 或 相关度排序
总 token 超过预算时从低优先级开始截断
```

---

## 6. 查询阶段：Global Search 详解

### 6.1 为什么需要 Global Search？

```
问题："这些文档的核心研究主题有哪些？"

Local Search 的困境：
  → 问题里没有具体实体，找不到入口节点
  → 即使硬找，top-k 个局部子图也覆盖不了全局

Global Search 的做法：
  → 不从实体出发，从【所有社区摘要】出发
  → 每个社区摘要都是一个"主题概括"
  → Map-Reduce：先让每个社区独立回答，再汇总
```

### 6.2 Map-Reduce 流程

```
               社区摘要1  社区摘要2  社区摘要3  ...  社区摘要N
                  │          │          │              │
                  ▼          ▼          ▼              ▼
  Map 阶段:    LLM 对每个社区摘要独立生成「部分回答 + 相关度评分」
                  │          │          │              │
                  ▼          ▼          ▼              ▼
              分数=85     分数=20     分数=92        分数=45
                  │                     │
                  ▼                     ▼
  Filter:    只保留相关度 > 阈值的部分回答
                  │                     │
                  └──────┬──────────────┘
                         ▼
  Reduce:    LLM 把保留的部分回答合成最终答案
                         │
                         ▼
                     最终回答
```

### 6.3 代码逻辑

```python
def global_search(query: str, communities: list[Community], threshold: float = 30):
    # Map 阶段：每个社区独立评估
    partial_answers = []
    for community in communities:
        result = llm_call(
            prompt=f"""
            根据以下社区摘要，回答用户的问题。
            如果这个社区与问题无关，返回相关度 0。
          
            社区摘要：{community.summary}
            用户问题：{query}
          
            请返回 JSON：
            {{"relevance_score": 0-100, "partial_answer": "..."}}
            """
        )
        if result.relevance_score > threshold:
            partial_answers.append(result)
  
    # 按相关度排序
    partial_answers.sort(key=lambda x: x.relevance_score, reverse=True)
  
    # Reduce 阶段：合成最终答案
    combined_context = "\n\n".join(
        f"[相关度 {pa.relevance_score}] {pa.partial_answer}" 
        for pa in partial_answers
    )
  
    final_answer = llm_call(
        prompt=f"""
        以下是从不同主题社区收集的部分回答，请综合这些信息，
        生成一个全面、连贯的最终回答。
      
        用户问题：{query}
      
        部分回答：
        {combined_context}
        """
    )
    return final_answer
```

### 6.4 Global Search 的代价

> ⚠️ **Global Search 很贵**。每次查询要对所有社区摘要调用一次 LLM（Map），社区数量多时 token 开销巨大。这是 GraphRAG 最大的成本问题。

优化策略：

- 只用粗粒度层（Level 2）的社区摘要，数量少
- 先用 embedding 过滤明显不相关的社区
- 异步并发调用 Map 阶段

---

## 7. 向量检索 vs GraphRAG 对比（A/B 思维）

| 问题                            | 纯向量 RAG 表现 | GraphRAG Local             | GraphRAG Global |
| :------------------------------ | :-------------- | :------------------------- | :-------------- |
| "DeepSeek-V3 有多少参数？"      | ✅ 容易命中     | ✅ 实体直达                | ❌ 杀鸡用牛刀   |
| "DeepSeek-V3 和 GPT-4 的关系？" | ⚠️ 碰运气     | ✅ 关系边直达              | ⚠️ 可能相关   |
| "所有用了 MoE 的模型有哪些？"   | ❌ top-k 不全   | ✅ 从 MoE 节点出发找所有边 | ⚠️ 看社区覆盖 |
| "这些论文的主要趋势？"          | ❌ 完全无能     | ❌ 没入口                  | ✅ 专门干这个   |
| "如何配置 BGE-M3？"             | ✅ 语义匹配强   | ⚠️ 实体抽取质量依赖      | ❌ 不适合       |

**结论：没有银弹，Hybrid 是终局。**

---

## 8. Hybrid 架构：向量 + 图的融合

### 8.1 设计思路

```
用户问题
    │
    ▼
┌────────────────────────────┐
│  Router（路由器）            │
│  判断问题类型：              │
│    - 具体事实 → Vector RAG  │
│    - 关系查询 → Local Search │
│    - 全局主题 → Global Search│
│    - 不确定 → 两路都查       │
└──────┬──────┬──────┬───────┘
       │      │      │
       ▼      ▼      ▼
   Vector   Local  Global
   Search   Search  Search
       │      │      │
       └──────┼──────┘
              ▼
     上下文合并 + 去重
              │
              ▼
         LLM 生成回答
```

### 8.2 路由策略

可以复用你 Phase 3 的 Router 思路：

```python
ROUTE_PROMPT = """
分析以下用户问题，判断最佳检索策略：

1. "vector" — 事实性问题，找最相关的文本片段即可
   例：如何配置某个参数？某个概念是什么意思？
   
2. "local" — 关系性问题，需要追溯实体之间的联系
   例：A 和 B 有什么关系？谁发明了 X？X 用了什么技术？
   
3. "global" — 全局性问题，需要纵览全局
   例：主要趋势是什么？有哪些共同特点？总结一下所有...
   
4. "hybrid" — 不确定，或问题同时涉及具体和关系
   例：比较 A 和 B 的优劣（需要各自事实 + 相互关系）

用户问题：{query}

只返回一个词：vector / local / global / hybrid
"""
```

---

## 9. 成本与取舍：工程现实

### 9.1 GraphRAG 的代价

| 维度                 | 向量 RAG                  | GraphRAG                                    |
| :------------------- | :------------------------ | :------------------------------------------ |
| **索引成本**   | 便宜（embedding API）     | **贵！每个 chunk 调一次 LLM 做抽取**  |
| **索引时间**   | 快（分钟级）              | 慢（大文档集可能数小时）                    |
| **查询成本**   | 一次 embedding + 一次 LLM | Local: 类似；Global:**N 倍 LLM 调用** |
| **维护**       | 简单（重新 embed）        | 增量更新图结构较复杂                        |
| **实现复杂度** | 低                        | 高（图构建 + 社区检测 + 多种检索）          |

### 9.2 什么时候值得用 GraphRAG？

```
✅ 值得用：
  - 文档集中充满实体和关系（人物传记、公司报告、学术论文）
  - 用户经常问关系类、聚合类问题
  - 需要回答全局性问题
  - 文档集相对稳定，不频繁更新

❌ 不值得用（向量 RAG 就够了）：
  - 文档是操作手册/FAQ，问题都是"怎么做 X"
  - 文档集很小（几十个 chunk，向量 top-k 就能覆盖）
  - 文档频繁变化（图的重建成本太高）
  - 预算有限（索引阶段 LLM 调用费用）
```

---

## 10. 和你已有系统的连接点

```
你已有的 Phase 1-4 代码              GraphRAG 新增
─────────────────────────           ────────────────
phase1_chunker.py            →      复用分块逻辑
phase1_embedder.py           →      实体 embedding 匹配
phase1_vector_store.py       →      Hybrid 中的向量路
phase2_01_hybrid_retriever   →      最终 Hybrid 融合参考
phase3_router.py             →      路由到 vector/local/global
phase3_query_decomposer.py   →      复杂图查询也可以拆子问题
phase4_semantic_memory.py    →      语义记忆的事实可以建图！
                                     (user.city = 纽约) 就是三元组
```

> **你 Phase 4 的 SemanticEntry（key-value 事实）本质上已经是简化的三元组了！**
> `(USER) —[city]→ (纽约)` 就是一个知识图谱关系。
> GraphRAG 只是把这个思路推广到整个文档集合。

---

## 11. 动手项目路线

```
📦 Phase 5 项目：GraphRAG 系统
│
├── 第一步：LLM 实体/关系抽取器
│   ├── 设计 extraction prompt
│   ├── 对已有 docs/ 里的文档做抽取
│   ├── 实现 gleaning（二次收割）
│   └── 输出：entities.json + relationships.json
│
├── 第二步：知识图谱构建
│   ├── 实体名称归一化 + 去重
│   ├── 关系合并
│   ├── 用 networkx 构建图 + 可视化
│   └── 输出：knowledge_graph.json
│
├── 第三步：社区检测 + 摘要
│   ├── Leiden 算法做社区检测
│   ├── LLM 生成社区摘要
│   └── 输出：communities.json (带层级)
│
├── 第四步：Local Search
│   ├── 实体匹配（精确 + embedding fallback）
│   ├── 子图提取（1-2 跳）
│   ├── 上下文组装 + 优先级排序
│   └── 对比向量 RAG 在关系问题上的表现
│
├── 第五步：Global Search
│   ├── Map-Reduce 流程
│   ├── 社区过滤优化
│   └── 对比向量 RAG 在全局问题上的表现
│
└── 第六步：Hybrid 融合
    ├── 实现 Router（复用 phase3_router 思路）
    ├── 向量检索 + 图检索上下文合并
    └── 端到端评估对比
```

---

## 12. 关键依赖（无框架方案）

```
# 你已有的
openai          # LLM 调用（DeepSeek）
numpy           # 向量计算

# 新增（都很轻量）
networkx        # 图数据结构 + 基础操作
graspologic     # 微软出的图分析库，包含 Leiden 算法

# 可选
pyvis           # 图可视化（生成交互式 HTML）
```

不需要 Neo4j。学习阶段用 networkx + JSON 持久化就够了，和你 Phase 1 用 `simple_index.json` 而不是 Chroma 是一个思路。

---

## 13. 推荐阅读顺序

1. **论文**：[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) — 只需读 Section 1-3，理解 Local/Global 设计
2. **微软博客**：[GraphRAG: Unlocking LLM discovery on narrative private datasets](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-datasets/) — 图文并茂，适合入门
3. **微软开源代码**：[microsoft/graphrag](https://github.com/microsoft/graphrag) — 学完原理后翻看源码，对照你的实现

---

以上就是 GraphRAG 的完整概念框架。要点回顾：

- **知识图谱 = 三元组（实体 → 关系 → 实体）**，向量库存文本，图存关系
- **索引阶段五步**：分块 → LLM 抽取 → 建图 → 社区检测 → 社区摘要
- **两种检索**：Local（实体追溯）解决关系问题，Global（Map-Reduce 社区摘要）解决全局问题
- **终局是 Hybrid**：Router 分流到向量检索/Local/Global

准备好开始动手写代码的时候告诉我，我们可以从第一步「LLM 实体关系抽取器」开始，复用你已有的 `config.py` 和 DeepSeek API 配置。
