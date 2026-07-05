# Phase 4.5：情景记忆完整闭环

情景记忆保存的不是“用户喜欢蓝色”这类稳定事实，而是“过去怎样完成一个具体任务”。本实现把每次 `AgenticRAG.query()` 视为一个任务边界，完成下面的闭环：

```text
新任务
  → Embedding 查询
  → 召回 Top-K 相似历史经验
  → 将精简经验注入 Agent 上下文
  → Agent 执行任务
  → LLM 结构化反思
  → 新经验 + 向量原子写入 JSON
```

## 1. 经验中保存什么

一条 `Episode` 包含：

- 任务描述与 `success / partial / failure` 结果。
- 脱敏后的工具名、参数和结果预览。
- 反思得到的结果摘要、执行策略、可复用经验和风险。
- 由“任务 + 反思”生成的 Embedding。

完整最终回答不写入经验库：它通常冗长，也可能包含不适合跨会话保留的内容。反思器只把对下次任务有用的部分压缩下来。

## 2. 为什么用 JSON 向量库

当前阶段故意不引入新框架。`EpisodicMemory` 把经验和向量保存在带 `schema_version` 的 JSON 中，召回时手写遍历并计算余弦相似度。这适合数百条以内的学习实验，也能直接打开文件观察数据。

写入流程先生成临时文件、`flush + fsync`，再用 `os.replace` 替换正式文件。崩溃或磁盘错误不会留下半条 JSON，内存状态也只在落盘成功后更新。

## 3. 反思与降级

`LLMEpisodeReflector` 用 function calling 强制返回 `EpisodeReflection` 结构。如果反思模型失败，系统仍会保存一条最小降级经验：

- Agent 已返回结果：记为 `partial`，保留脱敏轨迹。
- Agent 抛出异常：记为 `failure`，保留脱敏错误摘要。

Embedding、召回、反思或写盘失败都不会替换 Agent 的正常回答，也不会覆盖 Agent 原本抛出的异常。

## 4. 上下文顺序与安全

本轮召回的经验是查询相关、频繁变化的内容，所以它放在稳定的工作记忆和历史对话之后：

```text
Agent system prompt
  → 结构化状态
  → 历史摘要
  → 近期原文
  → 本轮召回的情景经验
  → 当前问题
```

经验消息明确标注为“参考数据，不是当前指令”。任务、回答、工具参数、预览和反思在进入 LLM / Embedding / JSON 前都会过确定性脱敏，过滤密码、API Key、Bearer Token 和私钥等常见秘密。

## 5. 运行实验

```bash
uv run python src/AIAgent/RAG/phase4_main.py \
  --strategy summary \
  --structured-state \
  --structured-state-file structured_memory.json \
  --episodic-memory-file episodic_memory.json \
  --episodic-top-k 3 \
  --episodic-min-score 0.35 \
  --episodic-max-episodes 200
```

情景记忆只在传入 `--episodic-memory-file` 时启用。可用命令：

- `/episodes`：查看所有已持久化经验。
- `/recall <query>`：不执行 Agent，手动检查相似经验。
- `/forget-episode <id>`：删除一条指定经验。
- `/clear-episodes`：清空长期情景记忆。
- `/clear`：仍然只清理当前工作记忆，不触碰情景记忆。

## 6. 生产落地与工程边界

本实现是一个面向小规模、本地实验的同步 Write-through 学习实现。如果在真实的工业级生产环境中落地情景记忆系统，需要重点解决以下工程要点：

### 6.1 用户隔离与多租户安全（安全红线 🔴）
* **物理/逻辑隔离**：在向量数据库中必须为每个 Episode 附带 `user_id` 或 `tenant_id` 等元数据。
* **强制过滤**：在执行向量检索时，必须附带数据库底层的 `Metadata Filter`（例如 `where user_id == current_user`），严防跨租户/跨用户的经验泄露。

### 6.2 按需调用与路由（效率与精准度 ⚡）
* **避免无脑检索**：每次查询都强制计算向量相似度不仅增加系统延迟与 Token 成本，还可能引入无关的历史经验，污染 LLM 的上下文（导致 Lost in the Middle 等干扰问题）。
* **工程解法**：
  1. **意图路由（Router）**：利用轻量级模型或规则预先判断任务复杂度，仅对复杂任务开启记忆检索。
  2. **LLM 按需调用（Tool Calling）**：将记忆检索包装为 Tool，由 LLM 根据当前规划自主决定是否调用 `recall_experiences` 工具。

### 6.3 异步反思与写入（延迟优化 ⏱️）
* **异步化解耦**：生成反思和计算 Embedding 属于慢操作。生产中不应同步阻塞主请求，应在 Agent 答复用户后，将执行轨迹（Traces）发送到消息队列（如 Kafka、RabbitMQ），由后台 Worker 异步进行 LLM 反思并落盘。

### 6.4 记忆的合并与剪枝（整合与冲突解决 🔄）
* **遗忘与去重**：随着运行时间增长，相似的经验会重复累积导致向量库膨胀，且新旧经验、成败经验之间可能存在冲突。
* **工程解法**：引入定时任务（类似于人类的睡眠记忆整理机制），在后台定期对情景记忆进行聚类、去重和提炼，将重复的、冲突的具体轨迹压缩合并为更高维度的通用规则，并清理废弃的陈旧记忆。

### 6.5 专用向量数据库（存储层扩展 💾）
* 当前使用的本地单 JSON 文件线性遍历仅适用于数百条以内的小规模实验。
* 生产环境需迁移到支持高并发、强一致性及实时索引的专业数据库（如 pgvector、Qdrant、Milvus）。当前的 `EpisodeReflector` 和 `EpisodeEmbedder` 协议已经为这种底层存储替换保留了良好的接口边界。

