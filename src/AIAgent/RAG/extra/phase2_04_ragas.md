# Phase 2 · 第四块：评估（RAGAS）

> 本文是学习笔记 + 「新对话快速启动器」。
> 新开会话学 Phase 2 第四块时，把这份文件喂给我即可立刻接上下文，无需重新铺垫。

---

## 0. 我现在在哪（进度锚点）

```
Phase 1 ✅ 经典 RAG（document_loader / chunker / embedder / vector_store / rag_chain / main）
Phase 2 🚧 进阶 RAG
   ├─ 第一块 ✅ 混合检索 Hybrid Search（BM25 + RRF）
   ├─ 第二块 ✅ Reranking（Cross-encoder 二次精排）
   ├─ 第三块 ✅ Query Rewriting / HyDE / Multi-Query
   └─ 第四块 ✅ 评估（RAGAS）                         ← 本文，已落地、已跑通
Phase 3+ ⬜ Agentic RAG / 记忆系统 / GraphRAG（见 learning_roadmap.md）
```

---

## 1. 为什么需要 RAGAS

前三块优化（Hybrid / Reranker / Query Rewriting）做完之后，有一个核心问题：

> **「它真的变好了吗？好了多少？」**

光靠肉眼看几个例子是不够的——可能刚好挑了有利的 case。
RAGAS 给 RAG 系统提供了**定量的、自动化的、无需人工标注答案就能运行的**评估框架。

## 什么是 RAGAS？
RAGAS（Retrieval Augmented Generation Assessment）是专门为 RAG 系统设计的评估框架，核心目标是：不需要人工标注的"黄金答案"，也能定量评估 RAG 各环节的质量。

---

## 2. 指标体系：核心 2 + 2，以及两个常见扩展

RAGAS 把 RAG 拆成两大部分分别评估：

> 指标没有“检索必须三个、生成必须三个”的固定标准。不同版本、教程和项目会选择不同组合。
> 本项目实现经典四指标最小集合（检索 2 个 + 生成 2 个），先覆盖 RAG 两个阶段最关键的问题；
> 另外两个指标列为后续扩展。代码和报告统一按 **Retrieval → Generation** 排列。

### 检索质量（Retrieval）

| 指标 | 评估什么 | 需要 ground_truth？ | 本项目 |
|:---|:---|:---|:---|
| **Context Precision** | 相关 chunk 是否排在召回结果前面 | ✅ 当前实现用 reference | ✅ 已实现 |
| **Context Recall** | 参考答案所需事实有多少被召回 | ✅ 需要 | ✅ 已实现 |
| **Context Relevancy** | 召回内容与问题的相关程度 | ❌ 通常不需要 | ⬜ 扩展；与 Precision 部分重叠 |

> 直觉：Precision 看的是"召回的有没有废料"，Recall 看的是"有没有漏掉关键信息"。


### 生成质量（Generation）

| 指标 | 评估什么 | 需要 ground_truth？ | 本项目 |
|:---|:---|:---|:---|
| **Faithfulness** | LLM 回答是否受到 contexts 支持 | ❌ 不需要 | ✅ 已实现 |
| **Answer Relevancy** | 生成回答是否切题 | ❌ 不需要 | ✅ 已实现 |
| **Answer Correctness** | answer 与参考答案在事实上是否一致 | ✅ 需要 | ⬜ 扩展 |


> **最重要的是 Faithfulness**：它回答"LLM 有没有瞎编"。

---

## 3. RAGAS 的运作原理（LLM-as-Judge）

RAGAS 的精髓是用**另一个 LLM 来评判**。整体流程：

```
输入                       自定义指标                输出
─────                      ────────────────         ──────
question                   ┌─ LLM-as-Judge ─┐
+ contexts      ──────────►│                │──►   Context Precision: 0.85
+ answer                   │  (DeepSeek)    │      Context Recall: 0.79
[+ ground_truth]           └────────────────┘      Faithfulness: 0.92
                                                    Answer Relevancy: 0.88
```

### Faithfulness 的计算逻辑（最典型）

1. 把 LLM 的回答拆成若干**原子声明**（atomic claims）
   - 例："ReAct 由 Thought、Action、Observation 三步组成"
2. 对每个声明，让评判 LLM 判断：这条声明能否从召回的 context 中推出？
3. `Faithfulness = 能被支撑的声明数 / 总声明数`

### Answer Relevancy 的计算逻辑

1. 让评判 LLM 根据答案**反向生成若干问题**
2. 计算生成的问题与原始问题的语义相似度（需要 Embedding）
3. 平均相似度 = Answer Relevancy

---

## 4. 代码落地（文件 / 接口 / 跑法）

### 新增文件

| 文件 | 内容 |
|:---|:---|
| `phase2_04_models.py` | EvaluationSample、RAGOutput、MetricScore、EvaluationReport |
| `phase2_04_eval_dataset.py` | 10 条强类型评估样本（覆盖 Transformer / RAG / Agent） |
| `phase2_04_metric_common.py` | 指标协议、名称和展示顺序 |
| `phase2_04_official_ragas_prompts.py` | 固定的 RAGAS 0.4.3 官方 Prompt / few-shot / Schema 快照 |
| `phase2_04_prompt_profiles.py` | `official` / `custom_zh` Profile 与输入字段适配 |
| `phase2_04_retrieval_metrics.py` | Context Precision / Context Recall 完整实现 |
| `phase2_04_generation_metrics.py` | Faithfulness / Answer Relevancy 完整实现 |
| `phase2_04_metrics.py` | 按 Retrieval → Generation 组装四个指标 |
| `phase2_04_evaluator.py` | 收集、调度、Trace 和 Baseline vs Full 报告 |

### 修改文件

| 文件 | 改动 |
|:---|:---|
| `config.py` | 新增 `ragas_llm_*` 和 `ragas_embedding_*` 配置（复用 DeepSeek + BGE-M3） |

### 跑法

```bash
# 冒烟测试（只跑 2 条，验证流程通）
uv run python phase2_04_evaluator.py --samples 2

# 使用项目自有中文 Prompt
uv run python phase2_04_evaluator.py --samples 2 --prompt-profile custom_zh

# 查看实际发送的 Profile、完整 Prompt、结构化输出和公式
uv run python phase2_04_evaluator.py --samples 1 --prompt-profile official --trace

# 单组评估（Full 配置，10 条）
uv run python phase2_04_evaluator.py

# 单组评估（Baseline 配置）
uv run python phase2_04_evaluator.py --config baseline

# 对比模式（Baseline vs Full，消耗 token 较多）
uv run python phase2_04_evaluator.py --compare
```

### 关键接口

```python
# phase2_04_evaluator.py 内部流程
rag = RAGChain(retriever_type="hybrid", use_reranker=True, query_rewrite="multi_query")
rag.load_index()

# 1. 收集 RAG 输出
collected = collect_rag_outputs(rag, EVAL_SAMPLES)
# collected[i] 是 RAGOutput，不再使用松散字典

# 2. 运行自定义指标；--trace 可查看完整 Prompt 和 Judge 输出
metrics = build_metrics(llm, embeddings, prompt_profile="official")
report = run_metrics(collected, metrics, trace=True)
# report.average_scores() 得到四项聚合分数
```

---

## 5. 评估配置设计

| 配置名 | retriever | reranker | query_rewrite | 目的 |
|:---|:---|:---|:---|:---|
| **Baseline** | dense | ❌ | none | 最朴素的 RAG，作为对照组 |
| **Full** | hybrid | ✅ | multi_query | 启用前三块全部优化 |

对比 Baseline vs Full 的各项指标，就能定量看到每块优化的实际收益。

---

## 6. RAGAS 配置要点

### 为什么用 DeepSeek，而 RAGAS 还保留着？

项目已有 DeepSeek API Key，因此通过 OpenAI 兼容接口接入。RAGAS 只用于创建统一的
LLM/Embedding 客户端；Prompt、Judge 输出结构和指标公式都已抽到项目源码，不再调用
RAGAS 的黑盒指标：

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(
    api_key="...",
    base_url="https://api.deepseek.com",
)
llm = llm_factory(
    "deepseek-chat",
    client=client,
    temperature=0,
    max_tokens=4096,
)
```

### Official 与 custom_zh Prompt Profile

`official` 是默认配置，完整固定 RAGAS 0.4.3 的英文 instruction、全部 few-shot、
Pydantic 输出 Schema 描述和 Prompt 拼接格式。它是稳定的对照基线，不会因为未来升级
RAGAS 依赖而自动变化。

`custom_zh` 是项目实验配置，保留中文 instruction、中文示例和提示词注入防护。
两套 Profile 使用完全相同的样本、Judge 模型、Embedding、指标公式与执行顺序，只有
Prompt 不同，因此可以分别运行后比较结果：

```bash
uv run python phase2_04_evaluator.py --prompt-profile official
uv run python phase2_04_evaluator.py --prompt-profile custom_zh
```

命令行不会自动双跑，避免在不知情时把 Judge 调用成本翻倍。报告和 `--trace` 都会打印
Profile 名称，保存结果时应连同 Profile 一起记录。

### 为什么 Answer Relevancy 还需要 Embedding？

该指标的计算方式是：让 LLM 根据答案反向生成问题，再用 Embedding 计算与原始问题的语义相似度。所以需要同时传入 `llm` 和 `embeddings`。

### 项目自己的数据模型

```python
output = RAGOutput(
    question=question,
    answer=answer,
    contexts=contexts,
    ground_truth=ground_truth,
)
```

---

## 7. 实测观察（待填充）

> 运行 `uv run python phase2_04_evaluator.py --compare` 后，把结果贴在这里。

### Full 配置（单组，10 条样本）

| 指标 | 分数 |
|:---|:---|
| Context Precision | — |
| Context Recall | — |
| Faithfulness | — |
| Answer Relevancy | — |

### 对比：Baseline vs Full

| 指标 | Baseline | Full | 提升 |
|:---|:---|:---|:---|
| Context Precision | — | — | — |
| Context Recall | — | — | — |
| Faithfulness | — | — | — |
| Answer Relevancy | — | — | — |

---

## 8. 核心收获

1. **RAGAS = LLM-as-Judge**：不需要人工标注大量答案，用另一个 LLM 当裁判，大幅降低评估成本
2. **无需 ground_truth 也能跑**：Faithfulness + Answer Relevancy 不依赖参考答案，可以立刻评估新问题
3. **评估本身有 token 消耗**：每条样本的评估约消耗 5~10 次 LLM 调用，10 条 × 4 指标约 200~400 次
4. **分数是相对的，对比才有意义**：A/B 对比（加了 Reranker 前后）比绝对分数更有参考价值
5. **评判 LLM 的质量影响结果**：DeepSeek 便宜好用，但如果评判 LLM 本身有偏差，指标也会有偏差

---

## 9. 待消化 / 可自己玩

- 把 `--samples 2` 改成全部 10 条，看完整评估结果
- 跑 `--compare` 对比 Baseline vs Full，量化前三块优化的收益
- 把实测分数填入上面的"实测观察"表格
- 尝试添加更多评估样本，覆盖更多 edge case（如知识库中没有答案的问题）
- 用 `result.to_pandas()` 把详细结果导出成 DataFrame 分析

---

## 10. 下一步：Phase 3 Agentic RAG

Phase 2 全部四块已完成。下一步是 Phase 3：

- **Agentic RAG**：把 RAG 变成 Agent 的一个工具，让 Agent 自主决定何时检索、检索什么
- **记忆系统**：长期记忆 + 短期记忆的工程实现
- **GraphRAG**：把知识库组织成知识图谱，支持更复杂的关系推理

> 新对话开场建议说：「Phase 2 全部四块已完成（hybrid / reranking / query_rewriting / ragas），现在要做 Phase 3 Agentic RAG，参考 learning_roadmap.md」。
