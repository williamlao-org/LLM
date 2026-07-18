# Phase 4.6：核心常驻 + 事实按需召回的语义记忆

语义记忆保存的是从具体对话中抽取出来、未来仍有用的稳定事实。它和工作记忆的关键区别不是“有没有写文件”，而是**作用域与激活方式**：工作记忆服务当前线程；长期语义记忆跨线程保存，只在与当前任务相关时进入 Prompt。

本阶段实现工程中常见的两层混合架构：少量核心记忆始终可见，大量长期事实存储在上下文之外，通过自动检索或工具调用按需激活。

> [!IMPORTANT]
> 本文同时讲两条线：标注为“当前实现”的内容描述这个本地学习项目；“真实生产工程”描述线上系统通常还需要的组件和约束。当前 JSON 实现用于看清原理，不应原样复制到多用户生产环境。

## 1. 先区分三个容易混淆的概念

### 1.1 工作记忆不等于非持久化

工作记忆描述的是“当前线程正在使用的状态”。它可以写入 checkpoint，以便进程重启后恢复；这仍然不意味着它变成了长期语义记忆。

### 1.2 持久化对话不等于语义记忆

把聊天原文整体保存下来，得到的是会话记录或情景资料。语义记忆还需要经过抽取和去上下文化，例如：

```text
原始对话：我下个月搬去纽约，以后推荐餐厅按纽约来。
语义事实：user.city = 纽约
```

### 1.3 存储状态不等于激活状态

长期事实写入文件，不代表必须立即加入 Prompt：

```text
SemanticMemory          # 所有持久化事实
        ↓ 按当前问题召回
Active Context          # 本轮真正需要的少量事实
        ↓
LLM Prompt
```

更新记忆文件本身不会影响 Prompt Cache；只有活动上下文发生变化，才会影响对应位置之后的缓存。

## 2. 本项目的分层模型

```text
identity / preference / constraint
  → StructuredWorkingMemory 核心状态，始终可见

decision / pending_task
  → 当前线程工作状态

fact
  → SemanticMemory 长期存储，按相关性召回
```

这三层分别解决不同问题：

| 类型     | 典型内容                             | 是否常驻 Prompt | 是否写入语义库 |
| -------- | ------------------------------------ | --------------: | -------------: |
| 核心状态 | 姓名、语言、稳定偏好、全局约束       |              是 |             否 |
| 线程状态 | 当前决定、未完成任务                 |      当前线程内 |             否 |
| 长期事实 | 居住地、项目背景、过去明确提供的事实 |              否 |             是 |

`StructuredWorkingMemory` 仍然只调用一次抽取模型。抽取出的 operations 按类别路由：`fact` 交给 `SemanticMemory`，不写入活动 `_entries`；因此正常的长期事实更新不会改变 `state_version`，也不会重建结构化 Prompt。

现有 structured-state JSON 被视为**线程 checkpoint**。旧文件会原样加载，不自动迁移；其中已有的 `fact` 只有在后续被更新或删除时，才退出活动状态并按新规则处理。

## 3. 什么时候写入记忆

工程中一般不会把每轮短期上下文整体复制进长期库，而是先进行记忆固化（memory consolidation）。常见策略有：

- 用户明确说“请记住”时立即处理。
- 用户明确忘记或更正旧信息时立即处理。
- 普通对话在自然停顿、累计一定 Token、任务结束或会话结束时批量抽取。
- 原始对话日志单独保存为情景资料，不直接当作语义事实。

从执行位置看，又可以分成：

- **Hot path**：在当前请求中抽取并写入，记忆立即可用，但会增加延迟。
- **Background consolidation**：回答后或会话之间异步整理，不阻塞主请求，但新记忆稍晚可见。

LangGraph 的记忆设计也明确区分这两类写入方式：[Memory writing strategies](https://docs.langchain.com/oss/python/concepts/memory)。

### 3.1 当前实现的策略

```text
Agent 成功回答
  → 完整问答进入 pending
  → 检查显式记忆信号 / Token 门槛 / 自然停顿
  → LLM 一次性生成结构化 operations
  → 按 category 路由到活动状态或长期语义库
```

- “请记住、忘记、更正、更新偏好”等中英文信号会绕过 Token 门槛，在本轮成功回答后立即抽取。
- 普通内容继续使用 Phase 4.4 的 Token 增量 + 自然停顿门控，减少模型调用和无意义写入。
- 失败的 Agent 回合不会进入 pending，也不会写记忆。
- 当前 CLI 仍同步执行抽取和写入；它借鉴了后台 consolidation 的批处理触发方式，但还没有真正使用异步 worker。

## 4. 如何存储、更新与保护事实

语义文件带有 `schema_version`。每个 `SemanticEntry` 包含：

- `category + key + value`
- `created_at + updated_at`
- 用于语义检索的 embedding

相同 `category + key` 使用最新明确值。更新时保留 `created_at`，刷新 `updated_at` 和 embedding；超过容量后淘汰最久未更新的事实。

抽取器不需要读取整个长期库。每次 consolidation 只检索一小组与 pending 对话相关的旧事实，提供给抽取模型复用稳定 key、识别更正和生成 delete，避免长期库变大后挤占抽取 Prompt。

写入采用：

```text
构建候选状态
  → 写临时文件
  → flush + fsync
  → os.replace 原子替换
  → 成功后才更新内存状态
```

Embedding 或落盘失败不会污染旧状态，也不会覆盖 Agent 的正常回答。损坏或版本不兼容的文件按空库降级，并保留原文件供排查。

密码、API Key、访问令牌、私钥和银行卡等敏感信息会在进入 embedding 或磁盘前被确定性拦截；只依赖抽取 Prompt 中的“不要保存秘密”是不够的。

## 5. 什么时候检索

工程中常见三种读取路径：

1. **会话开始时加载核心画像**姓名、语言、稳定偏好和全局约束数量少，直接放入活动状态。
2. **回答前由应用自动检索**使用当前问题搜索 top-k，延迟低、行为确定，适合大部分对话产品。
3. **Agent 推理时调用记忆工具**
   模型判断自动结果不足时，换查询角度再次检索；能力灵活，但多一次模型往返，而且 Agent 可能不调用。

本项目采用混合方式：

```text
核心画像       → 自动加载
普通长期事实   → 每轮自动检索，达到相似度门槛才注入
复杂或二次查询 → search_semantic_memory 工具
```

Letta 也采用类似的 core memory + archival memory 思路；LangGraph 则用跨线程 Store 保存长期记忆：[Letta Context Hierarchy](https://docs.letta.com/guides/core-concepts/memory/context-hierarchy)、[LangGraph Memory](https://docs.langchain.com/oss/python/concepts/memory)。

## 6. 检索结果放在哪里

Prompt Cache 通常依赖相同前缀。动态内容越靠前，失效范围越大：

| 放置方式           | 缓存影响                     | 适用场景                  |
| ------------------ | ---------------------------- | ------------------------- |
| System 后、历史前  | 更新时会使后续历史缓存失效   | 少量、低频变化的核心记忆  |
| 历史后、当前问题前 | 保留前面的稳定前缀           | 每轮动态召回的 top-k 事实 |
| Tool result        | 以追加消息形式出现，前缀友好 | Agent 自主二次检索        |

OpenAI 和 Anthropic 的 Prompt Cache 都强调稳定前缀：[OpenAI Prompt Caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Claude Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)。

因此当前上下文顺序固定为：

```text
Agent system / tools                     # 最稳定
  → 结构化核心状态                        # 低频更新
  → 历史摘要                              # 偶尔更新
  → 近期原始对话                          # 追加增长
  → 本轮自动召回的语义事实                 # 每轮变化
  → 本轮召回的情景经验                     # 每轮变化
  → 当前问题                              # 每轮变化
```

长期事实写入时不会自动改动核心状态。当前对话本身已经包含用户刚说的信息，无需在同一轮重复注入；后续只有相关问题才会召回该事实。

例如：

```text
长期存储：user.city = 纽约

用户问 Python → 检索结果低于门槛，不注入 city
用户问附近餐厅 → 召回 city，追加到历史之后
```

如果自动召回不足，工具调用形成以下消息序列：

```text
历史消息
user: 当前问题
assistant: search_semantic_memory(...)
tool: 检索出的事实
assistant: 最终回答
```

工具路径更灵活、也保持追加式结构，但需要额外一次模型调用；所以简单问题优先使用自动预检索，工具用于二次查询。

## 7. 完整数据流

```text
                              ┌──────────────────────────────┐
对话成功结束                  │ StructuredWorkingMemory      │
  → 门控批量抽取 operations ──┤ identity/preference/...      │
                              │ → 活动核心/线程状态           │
                              └──────────────────────────────┘
                                         │
                                         │ fact
                                         ▼
                              ┌──────────────────────────────┐
                              │ SemanticMemory               │
                              │ JSON + embedding             │
                              └──────────────────────────────┘
                                         │
                    ┌────────────────────┴───────────────────┐
                    ▼                                        ▼
          当前问题自动 top-k                      search_semantic_memory
                    │                                        │
                    └────────────────────┬───────────────────┘
                                         ▼
                              动态事实追加到 Prompt 尾部
```

这里最重要的解耦是：

```text
semantic_memory.apply_operations(...)  # 只改变长期存储
semantic_memory.recall(...)             # 决定本轮激活什么
```

“保存”和“放进上下文”不再是同一个动作。

## 8. 运行实验

```bash
uv run python src/AIAgent/RAG/phase4_main.py \
  --strategy summary \
  --structured-state-file structured_memory.json \
  --semantic-memory-file semantic_memory.json \
  --episodic-memory-file episodic_memory.json \
  --semantic-top-k 3 \
  --semantic-min-score 0.35 \
```

指定 `--semantic-memory-file` 会自动启用结构化抽取，不再强制要求 `--structured-state`。核心状态是否跨进程恢复，取决于是否同时提供 `--structured-state-file`。

可用命令：

- `/state`：查看当前核心与线程状态。
- `/extract`：立即处理 pending 对话。
- `/semantic`：查看全部长期事实。
- `/recall-semantic <query>`：手动验证召回结果。
- `/forget-semantic <key>`：精确删除一个长期事实。
- `/clear-semantic`：清空长期语义库。
- `/clear`：只清理当前工作/核心上下文，不删除语义库或情景库。

### 遗忘机制

语义事实与情景经验使用同一套可解释的遗忘评分。系统计算距离最近一次
成功召回的空闲时间（从未召回时使用语义事实的更新时间或情景经验的创建
时间），再除以保留窗口：

```
基础保留天数 × (1 + importance) × (1 + min(recall_count, 5) / 5)
```

评分达到 `1` 后会物理删除。默认语义事实保留 90 天、情景经验保留 30 天；
普通记录的 `importance` 为 `0.5`，更正信息通常为 `0.7`，用户明确要求
“不要忘记/长期记住”时为 `0.9`，但仍不是永久保留。每次写入后会自动清理，
也可以使用 `/prune` 立即清理两类已启用的长期记忆。可通过
`--semantic-retention-days` 和 `--episodic-retention-days` 调整基础窗口。

建议依次验证：

1. 输入“请记住我住在上海”，用 `/semantic` 确认它立即进入长期库。
2. 询问无关的 Python 问题，观察该事实不会被注入。
3. 询问“给我推荐附近的餐厅”，观察 `user.city` 被自动召回。
4. 输入“请更正我的地址：我搬到纽约了”，确认相同 key 被立即更新，而不是新增重复事实。
5. 使用 `/forget-semantic user.city`，确认事实被精确删除。

## 9. 真实生产工程：具体怎么落地

生产系统没有唯一标准答案，但成熟实现通常都会把**在线回答链路**和**后台记忆整理链路**分开。LangGraph 也将线程 checkpointer 与跨线程 Store 分开，并建议生产使用数据库后端：[LangGraph Memory](https://docs.langchain.com/oss/python/langgraph/add-memory)。

### 9.1 参考架构

```text
在线读取链路
────────────
Client
  → API Gateway / Auth
  → Conversation Service：加载 thread checkpoint
  → Core Profile Store：读取低频核心画像
  → Memory Retrieval：namespace 过滤 + 检索 + 重排
  → Prompt Builder：按稳定性排序并控制 Token 预算
  → Agent / LLM
  → 返回答案，同时写 conversation event

后台写入链路
────────────
conversation event
  → Transactional Outbox / Queue
  → Consolidation Worker
  → 候选事实抽取
  → 安全、重要性、冲突与权限校验
  → 事务性 upsert / tombstone
  → 生成或更新 embedding
  → 更新检索索引、审计记录与缓存版本
```

`Memory Service` 不一定一开始就要拆成微服务。小团队可以先做成应用内部模块，但接口上仍应把 `thread state`、`core profile`、`semantic store` 和 `retriever` 分开，避免以后无法替换存储或增加租户隔离。

### 9.2 各类数据实际存在哪里

| 数据           | 生产中的常见载体                           | 一致性要求                 | 读取方式                           |
| -------------- | ------------------------------------------ | -------------------------- | ---------------------------------- |
| 对话与工作状态 | PostgreSQL/数据库 checkpointer             | 同一线程内强一致或顺序一致 | 按`thread_id` 精确读取           |
| 核心用户画像   | PostgreSQL JSONB、关系表或 KV；可加 Redis  | 更新后下一轮应可见         | 按`tenant_id + user_id` 精确读取 |
| 长期语义事实   | PostgreSQL + pgvector，或 Qdrant/Milvus 等 | 通常最终一致               | 元数据过滤后做向量/混合检索        |
| 记忆事件与审计 | 追加式事件表或日志系统                     | 不可悄悄丢失               | 按 memory/user/source 追溯         |
| Embedding 索引 | pgvector 或独立向量库                      | 可重建                     | 只负责候选召回，不代替权限系统     |

规模不大、强事务要求高时，PostgreSQL + pgvector 往往最省复杂度；数据量、吞吐或检索能力增长后，再考虑独立向量数据库。需要严格审计或删除合规时，常见做法是保留一个规范化事实表作为 source of truth，把向量当作可重建索引；也可以使用同时保存 payload 和 vector 的数据库，但仍要有版本、来源和删除流程。

一条生产记忆通常不只包含 `key/value`：

```json
{
  "memory_id": "mem_...",
  "tenant_id": "tenant_123",
  "user_id": "user_456",
  "scope": "user|project|organization",
  "type": "identity|preference|constraint|fact",
  "key": "user.city",
  "value": "纽约",
  "source_message_ids": ["msg_..."],
  "confidence": 0.98,
  "status": "active|superseded|deleted",
  "valid_from": "...",
  "valid_to": null,
  "version": 3,
  "embedding_model": "...",
  "created_at": "...",
  "updated_at": "..."
}
```

字段不必一次全部实现，但 `tenant/user/scope`、来源、状态、版本和时间通常不能长期缺席。

### 9.3 在线读取链路

真实请求一般按以下顺序处理：

1. **鉴权并确定作用域**：从可信身份系统得到 `tenant_id/user_id/project_id`，不能让模型或用户文本自行提供过滤条件。
2. **加载线程状态**：按 `thread_id` 恢复摘要、近期消息、pending task 和工具状态。
3. **精确读取核心画像**：姓名、语言、明确约束等用 KV/SQL 获取，不必绕一圈向量检索。
4. **判断是否需要长期召回**：规则、轻量分类器或 Agent 规划均可；高召回要求的产品也会每轮低成本预检索。
5. **构造检索查询**：通常组合当前问题、任务摘要和必要实体，而不是把整段历史直接 embedding。
6. **先做硬过滤**：`tenant/user/scope/status/validity/ACL` 必须进入数据库查询条件。
7. **召回候选**：可组合 dense、BM25/关键词和精确 key 查询，再进行融合或 rerank。
8. **后处理**：去重、剔除过期/矛盾事实、限制每类数量，并按 Token 预算截断。
9. **组装 Prompt**：核心状态靠前；动态事实放在历史后、当前问题前。
10. **必要时工具二次检索**：第一次结果不足时换查询，而不是无上限地把更多记忆塞进上下文。

向量相似度不是权限控制。以 Qdrant 为例，官方建议把租户和业务字段放在 payload 中并建立过滤索引；多租户查询必须带相应 filter：[Qdrant Filtering](https://qdrant.tech/documentation/search/filtering/)、[Qdrant Multitenancy](https://qdrant.tech/documentation/tutorials/multiple-partitions/)。

```text
错误：vector_search(query) 后在应用层“希望”结果都属于当前用户

正确：vector_search(
        query,
        filter = tenant_id AND user_id AND scope AND status=active
      )
```

### 9.4 在线与后台写入如何选择

生产中通常混用两条写路径：

#### 显式记忆操作：走 hot path

用户说“请记住”“忘记”“把地址改成……”时，需要明确反馈成功或失败：

```text
请求
  → 校验用户权限与内容安全
  → 生成确定性 operation
  → 事务 upsert/delete
  → 写审计事件
  → 更新当前会话的 read-your-writes overlay
  → 返回确认
```

如果 embedding 异步生成，精确 key 读取仍应立即看到新值；向量召回可以稍后最终一致。这样用户刚更正地址后，下一句话不会继续读到旧地址。

#### 隐式候选事实：走后台 consolidation

模型从普通对话中推断“这可能值得记住”时，一般不阻塞主回答：

```text
主请求提交 conversation event
  → outbox 与会话事务一起提交
  → worker 消费事件
  → LLM 抽取候选事实
  → 确定性策略过滤
  → 查询现有事实并解决冲突
  → 幂等写入 + embedding + 审计
```

这里常用 transactional outbox，是为了避免“回答已经保存，但发送队列消息失败”导致永久漏记。Worker 必须支持重试、死信队列和幂等键，例如 `source_message_id + memory_key + extractor_version`。

主回答模型不应拿到一个可以绕过校验、任意写数据库的底层工具。更安全的做法是：模型只提交候选 operation，应用层负责 schema 校验、敏感信息过滤、权限、冲突策略和最终提交。

### 9.5 冲突、并发与最终一致性

简单 demo 可以“同 key 后写覆盖前写”，生产环境还要处理：

- **同时更新**：两个会话并发修改同一画像时，用版本号/乐观锁 compare-and-swap，冲突后重新读取和合并。
- **乱序事件**：后台任务可能后提交先完成；根据 source event 时间和版本判断，不能让旧消息覆盖新更正。
- **来源优先级**：用户明确更正通常高于模型推断；可信业务系统可能高于闲聊文本。
- **时间事实**：`曾经住在上海` 与 `现在住在纽约` 可以同时成立，需要 `valid_from/valid_to`，而不只是覆盖字符串。
- **语义重复**：`user.city=纽约` 与“用户目前居住在 NYC”应合并，而不是长期积累近义副本。
- **删除传播**：使用 tombstone/删除事件同步清理规范表、向量索引、缓存和派生画像；备份按合规策略到期清除。
- **Embedding 升级**：保存 `embedding_model/version`，通过后台重建或双索引切换，不能把不同维度向量混在同一索引里。

跨线程长期记忆通常是最终一致的，但同一用户刚执行的显式写入应提供 read-your-writes。常见办法是在当前会话保留短期 overlay，等后台索引完成后再由正式存储接管。

### 9.6 Prompt 与缓存的生产处理

生产 Prompt 通常按变化频率排列：

```text
公共 system / 稳定 tools
  → 租户策略
  → 版本化核心画像快照
  → 历史摘要与近期消息
  → 本轮动态召回事实
  → 当前问题
```

具体原则：

- 只更新长期数据库不会使 Prompt Cache 失效；更新了注入 Prompt 的核心画像才会从该位置产生新前缀。
- 核心画像用版本化快照，内容没变就不要重新序列化或改变字段顺序。
- 动态召回放在尾部；工具定义保持稳定，不要按每轮结果动态增删工具 schema。
- 检索结果通常不写回规范 conversation history，避免下一轮重复、过期和无限膨胀；它可以保存在 trace 中用于审计和调试。
- Tool result 属于当前 Agent 执行轨迹；长会话中应按上下文策略裁剪或摘要，而不是永久累积全部结果。
- 必须记录供应商返回的 cached token/cache hit 指标，不能只凭 Prompt 看起来相同就假设缓存生效。

### 9.7 安全与隐私不是最后再补

生产记忆系统至少需要：

- **强制租户隔离**：namespace/filter 在服务端从认证信息生成；任何检索路径都不能绕过。
- **最小化保存**：默认只保存完成功能所需的事实，不把“以后可能有用”当作无限收集的理由。
- **敏感信息检测**：输入层、抽取结果和落盘前多层检测；秘密管理数据不进入记忆库。
- **Prompt 注入隔离**：长期记忆仍是不可信数据，明确标记为 facts，不允许它覆盖 system/developer 指令。
- **用户控制**：提供查看、更正、删除、禁用记忆和保留期限设置。
- **加密与访问审计**：传输/静态加密，后台 worker 使用最小权限，记录谁在何时读取或修改记忆。
- **完整删除链路**：删除不仅是向量库里少一条，还包括 source of truth、缓存、索引、派生数据和备份生命周期。

### 9.8 生产系统如何评估

只看“模型好像记住了”不够，需要把写入和读取分别评估：

| 层面     | 关键指标/测试                                                       |
| -------- | ------------------------------------------------------------------- |
| 写入质量 | 候选事实 precision、漏记率、错误归类率、重复率、冲突解决正确率      |
| 检索质量 | Recall@K、Precision@K、MRR/NDCG、过期事实召回率、无关记忆注入率     |
| 生成效果 | 个性化正确率、事实忠实度、矛盾回答率、无记忆基线对比                |
| 性能成本 | p50/p95 写入与召回延迟、Embedding/LLM 成本、Prompt Token、Cache Hit |
| 可靠性   | 队列积压、重试/死信率、索引延迟、写入失败率、版本冲突率             |
| 安全     | 跨租户泄漏必须为零、删除传播测试、秘密落盘测试、Prompt 注入红队测试 |

应准备带有“该记/不该记、该召回/不该召回、旧值/新值、不同租户”的固定数据集，在每次修改抽取 Prompt、Embedding、阈值或 reranker 后回归测试。

## 10. 当前学习实现与生产系统对照

| 能力     | 当前项目                | 真实生产常见做法                                  |
| -------- | ----------------------- | ------------------------------------------------- |
| 用户规模 | 单用户 CLI              | 多租户认证与 namespace 隔离                       |
| 工作状态 | 内存 + 可选 JSON        | 数据库 checkpointer，按 thread 恢复               |
| 核心画像 | StructuredWorkingMemory | 版本化 Profile Store + 缓存                       |
| 语义事实 | 单个版本化 JSON         | PostgreSQL/pgvector 或向量数据库                  |
| 检索     | 全库线性余弦 top-k      | 元数据过滤 + ANN/混合检索 + rerank                |
| 写入     | 回答后同步、门控批处理  | 显式操作 hot path；隐式事实走 outbox/queue/worker |
| 冲突     | 相同 key 覆盖           | 来源优先级、时态、版本、乐观锁、语义去重          |
| 一致性   | 单进程内存与文件同步    | 核心 read-your-writes，向量索引最终一致           |
| 安全     | 本地敏感模式过滤        | ACL、租户过滤、加密、审计、删除合规               |
| 可观测性 | CLI 状态与错误          | trace、指标、告警、离线评估和红队测试             |

这个项目不是“假的工程做法”；它实现的是生产架构中最核心的数据流，只是把外部系统压缩成了本地对象：

```text
当前 JSON                  → 未来的数据库/向量 Store
Token + 停顿门控          → 未来的事件触发与后台 consolidation
SemanticOperationSink     → 未来的 Memory Service 写接口
SemanticAgent             → 未来的在线 retrieval/prompt middleware
search_semantic_memory    → 未来的 Agent memory tool
last_*_error              → 未来的 metrics / traces / alerts
```

推荐按以下顺序把学习实现升级成可上线原型：

1. 抽象 Store 接口，引入 `tenant_id/user_id/project_id/thread_id`，迁移到 PostgreSQL + pgvector。
2. 把 conversation state 迁移到数据库 checkpointer，语义事实与线程状态彻底分库/分表。
3. 引入 outbox + worker，让普通 consolidation 异步化；显式记忆保留同步确认和 session overlay。
4. 增加来源、版本、状态、有效期与审计表，实现乐观锁、tombstone 和删除传播。
5. 增加混合检索、rerank、Token 预算与 retrieval router，并建立固定评估集。
6. 最后再根据容量与吞吐决定是否从 pgvector 迁移到独立向量数据库，而不是一开始就堆基础设施。

一句话总结：

> **核心记忆低频更新并常驻上下文；长期事实默认只写存储，只有与当前任务相关时才被激活到 Prompt 尾部。真实生产系统还必须围绕它补上租户隔离、异步固化、一致性、审计、评估和删除闭环。**

---
Q: 真实工程中也是这样吗?

Viewed phase4_06_semantic_memory.md:253-400
Viewed phase4_06_semantic_memory.md:400-487

**核心数据流是一样的，但生产系统在"怎么做"上有很多额外维度。**

你的文档第 10 节（[phase4_06_semantic_memory.md: L449-L482](file:///Users/williamlao/Project/LLM/src/AIAgent/RAG/extra/phase4_06_semantic_memory.md#L449-L482)）其实已经很详细地对照过了。我用更通俗的方式帮你理清**哪些相同、哪些不同**：

---

### ✅ 相同的：核心架构思想

以下这些设计在生产系统中是通用的，当前项目已经完整实现了：

| 设计原则 | 说明 |
|:---|:---|
| **三层分离** | 工作记忆（常驻）+ 语义事实（按需召回）+ 情景经验（按需召回），层次划分一致 |
| **保存 ≠ 激活** | 写入长期存储不会自动塞进 Prompt，需要检索门槛才注入 |
| **统一抽取 → 按类分流** | LLM 一次性提取 operations，代码按 category 路由到不同存储 |
| **Prompt 按稳定性排序** | 稳定内容靠前、动态内容靠后，最大化 Cache |
| **门控批处理** | 不是每句话都触发固化，而是累积到一定量或检测到显式信号才触发 |

---

### ❌ 不同的：生产系统额外要解决的问题

| 维度 | 当前项目 | 真实生产 |
|:---|:---|:---|
| **写入是同步的** | `add_turn` 之后直接在主线程抽取、写文件 | 显式记忆（"请记住"）走 hot path 同步确认；**隐式记忆走异步后台 Worker**，不阻塞主回答 |
| **存储是本地 JSON** | 单文件读写，进程内原子替换 | PostgreSQL + pgvector 或独立向量库（Qdrant/Milvus），支持并发、事务、索引 |
| **检索是全库线性扫描** | 遍历所有条目计算 cosine | ANN 近似最近邻检索 + 元数据硬过滤（tenant/user/scope） + Rerank |
| **没有多租户** | 单用户 CLI，内存中一份数据 | 每条记忆都带 `tenant_id / user_id`，检索时**必须**从认证信息生成 filter，防止跨租户泄漏 |
| **冲突处理简单** | 同 key 直接覆盖 | 需要版本号 / 乐观锁、来源优先级（用户更正 > 模型推断）、时态事实（`valid_from/valid_to`）、语义去重 |
| **一致性** | 单进程不存在并发问题 | 显式写入需要 read-your-writes（用 session overlay）；向量索引可以最终一致 |
| **删除** | 直接从字典和文件中移除 | 需要 tombstone + 传播删除到向量索引、缓存、派生数据、备份，满足 GDPR 等合规 |
| **可观测性** | `last_*_error` 打印到 CLI | 结构化 trace、metrics、告警、离线评估数据集、红队测试 |

---

### 🎯 一句话总结

> **架构骨架是一样的**——三层分离、按需召回、门控固化、分流路由——这些在 Letta、LangGraph、ChatGPT Memory 等系统中都能看到同样的模式。当前项目把生产中需要数据库、消息队列、微服务的部分压缩成了本地的 Python 对象和 JSON 文件，**看清的是数据流原理，省略的是多租户、并发、异步和合规这些工程化的"脏活"**。
