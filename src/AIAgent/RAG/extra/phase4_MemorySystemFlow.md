# 三层记忆系统完整流程图

## 一、总览：一轮对话的完整生命周期

```mermaid
flowchart TB
    subgraph INPUT["① 用户输入"]
        Q["用户提问"]
    end

    subgraph READ["② 读取阶段（构建 Prompt）"]
        direction TB
        WM["工作记忆<br/>StructuredWorkingMemory<br/>──────────────<br/>核心状态（identity/preference/constraint）<br/>+ 当前线程状态（decision/pending_task）<br/>+ 近期对话历史 / 历史摘要"]
        SM_R["语义记忆检索<br/>SemanticMemory.recall(question)<br/>──────────────<br/>用当前问题 Embedding<br/>在长期事实库中 top-k 检索<br/>只有 score ≥ 门槛的才注入"]
        EM_R["情景记忆检索<br/>EpisodicMemory.recall(question)<br/>──────────────<br/>用当前问题 Embedding<br/>在历史经验库中 top-k 检索<br/>召回成功/失败的任务经验"]
    end

    subgraph PROMPT["③ 拼装 Prompt"]
        direction TB
        P1["System Prompt（Agent指令 + 工具定义）"]
        P2["结构化核心状态 JSON（常驻）"]
        P3["历史摘要 / 近期原始对话"]
        P4["本轮召回的语义事实"]
        P5["本轮召回的情景经验"]
        P6["当前用户问题"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph AGENT["④ Agent 推理循环"]
        direction TB
        LLM["LLM 推理"]
        TC{"有工具调用？"}
        TOOL["执行工具<br/>（RAG检索 / search_semantic_memory / ...）"]
        ANS["生成最终回答"]
        LLM --> TC
        TC -->|是| TOOL --> LLM
        TC -->|否| ANS
    end

    subgraph WRITE["⑤ 写入阶段（回答成功后）"]
        direction TB
        AT["memory.add_turn(question, answer)<br/>将本轮问答写入对话历史"]
        EP_W["情景记忆写入<br/>EpisodicMemory.record()<br/>──────────────<br/>记录完整执行轨迹<br/>（含工具调用步骤 or 异常信息）<br/>+ LLM 自动反思提炼"]
        GATE{"记忆固化门控<br/>──────────────<br/>Token 增量 ≥ 门槛？<br/>自然停顿 or 工具调用？<br/>包含显式记忆信号？"}
        SKIP["暂不抽取<br/>对话进入 pending 队列"]
        EXTRACT["LLM 统一抽取<br/>LLMWorkingStateExtractor<br/>──────────────<br/>一次性从 pending 对话中<br/>提取 operations 列表"]
        ROUTE{"按 category 分流路由"}
        WM_W["写入工作记忆<br/>StructuredWorkingMemory<br/>──────────────<br/>identity / preference /<br/>constraint / decision /<br/>pending_task<br/>→ 更新常驻活动状态"]
        SM_W["写入语义记忆<br/>SemanticMemory.apply_operations()<br/>──────────────<br/>category = fact<br/>→ 计算 Embedding<br/>→ 原子写入 JSON 文件<br/>→ 不进入常驻 Prompt"]
    end

    Q --> WM
    Q --> SM_R
    Q --> EM_R
    WM --> PROMPT
    SM_R --> PROMPT
    EM_R --> PROMPT
    PROMPT --> AGENT
    ANS --> AT
    ANS --> EP_W
    AT --> GATE
    GATE -->|未满足| SKIP
    GATE -->|满足| EXTRACT
    EXTRACT --> ROUTE
    ROUTE -->|"identity / preference / constraint<br/>decision / pending_task"| WM_W
    ROUTE -->|"fact"| SM_W

    style INPUT fill:#4a9eff,color:#fff,stroke:none
    style READ fill:#2d8cf0,color:#fff,stroke:none
    style PROMPT fill:#19be6b,color:#fff,stroke:none
    style AGENT fill:#ff9900,color:#fff,stroke:none
    style WRITE fill:#ed4014,color:#fff,stroke:none
```

---

## 二、读取阶段详解：三种记忆如何进入 Prompt

```mermaid
flowchart LR
    subgraph WM["工作记忆（常驻）"]
        direction TB
        W1["StructuredWorkingMemory<br/>.get_context_messages()"]
        W2["直接输出全量核心状态 JSON<br/>+ 基础对话历史"]
        W1 --> W2
    end

    subgraph SM["语义记忆（按需召回）"]
        direction TB
        S1["SemanticAgent.query()"]
        S2["semantic_memory.recall(question)"]
        S3["cosine_similarity ≥ min_score ?"]
        S4["format_context() → JSON 注入"]
        S5["不注入"]
        S1 --> S2 --> S3
        S3 -->|是| S4
        S3 -->|否| S5
    end

    subgraph EM["情景记忆（按需召回）"]
        direction TB
        E1["EpisodicAgent.query()"]
        E2["episodic_memory.recall(question)"]
        E3["cosine_similarity ≥ min_score ?"]
        E4["format_context() → 经验注入"]
        E5["不注入"]
        E1 --> E2 --> E3
        E3 -->|是| E4
        E3 -->|否| E5
    end

    Q["用户问题"] --> W1
    Q --> S1
    Q --> E1

    W2 --> FINAL["最终 Prompt Messages"]
    S4 --> FINAL
    E4 --> FINAL

    style WM fill:#2d8cf0,color:#fff,stroke:none
    style SM fill:#19be6b,color:#fff,stroke:none
    style EM fill:#ff9900,color:#fff,stroke:none
```

> [!IMPORTANT]
> 工作记忆**不做向量检索**，内容少且稳定，全量常驻。语义记忆和情景记忆都通过 Embedding 相似度检索，只有达到门槛的才注入。

---

## 三、写入阶段详解：三种记忆的写入触发机制

```mermaid
flowchart TB
    SUCCESS["Agent 成功回答"]

    subgraph EP["情景记忆写入"]
        direction TB
        EP1{"result 中有 steps？<br/>（即发生了工具调用）"}
        EP2["record(question, result=result)<br/>记录成功经验"]
        EP3["不记录<br/>（纯闲聊无工具调用）"]
        EP1 -->|是| EP2
        EP1 -->|否| EP3
    end

    FAIL["Agent 执行异常"] --> EP_ERR["record(question, error=exception)<br/>记录失败教训"]

    subgraph CONSOLIDATION["记忆固化（统一抽取 + 分流）"]
        direction TB
        PENDING["本轮问答进入 pending 队列"]
        POLICY{"TokenAndBreakUpdatePolicy<br/>.should_extract()"}
        SIG{"包含显式记忆信号？<br/>「请记住 / forget / 更正...」"}
        WAIT["继续累积<br/>等待下一轮"]
        FLUSH["flush_pending()<br/>批量抽取"]
        LLM_EX["LLMWorkingStateExtractor.extract()<br/>──────────────<br/>输入：pending 对话 + 已有条目<br/>输出：operations 列表"]
        APPLY["_apply_operations()"]
        R1["category ∈ {identity, preference,<br/>constraint, decision, pending_task}<br/>→ 更新 _entries 活动状态<br/>→ 重建 state cache<br/>→ 保存到 structured_state.json"]
        R2["category = fact<br/>→ semantic_sink.apply_operations()<br/>→ 计算 Embedding<br/>→ 原子写入 semantic_memory.json<br/>→ 不改变活动 Prompt"]

        PENDING --> SIG
        SIG -->|是：绕过 Token 门槛| FLUSH
        SIG -->|否| POLICY
        POLICY -->|未满足| WAIT
        POLICY -->|满足| FLUSH
        FLUSH --> LLM_EX --> APPLY
        APPLY --> R1
        APPLY --> R2
    end

    SUCCESS --> EP
    SUCCESS --> PENDING

    style EP fill:#ff9900,color:#fff,stroke:none
    style CONSOLIDATION fill:#ed4014,color:#fff,stroke:none
```

---

## 四、Prompt 消息排列顺序（Prompt Cache 优化）

按照从**最稳定**到**每轮变化**排列，最大化 Prompt Cache 命中率：

```text
┌─────────────────────────────────────────────────┐
│ 1. System Prompt（Agent 指令 + 工具定义）         │  ← 最稳定，几乎不变
├─────────────────────────────────────────────────┤
│ 2. 结构化核心状态 JSON                           │  ← 低频更新
│    identity / preference / constraint            │
│    decision / pending_task                        │
├─────────────────────────────────────────────────┤
│ 3. 历史摘要 (SummaryBufferMemory)                │  ← 偶尔更新
├─────────────────────────────────────────────────┤
│ 4. 近期原始对话轮次                              │  ← 追加增长
├─────────────────────────────────────────────────┤
│ 5. 本轮自动召回的语义事实                         │  ← 每轮变化
│    SemanticMemory.recall() → format_context()    │
├─────────────────────────────────────────────────┤
│ 6. 本轮召回的情景经验                             │  ← 每轮变化
│    EpisodicMemory.recall() → format_context()    │
├─────────────────────────────────────────────────┤
│ 7. 当前用户问题                                  │  ← 每轮变化
└─────────────────────────────────────────────────┘
```

> [!NOTE]
> 动态内容越靠后，前面稳定前缀的 Prompt Cache 命中率越高。这就是为什么语义事实和情景经验放在历史对话之后、用户问题之前。

---

## 五、三种记忆对比总结

| 维度                  | 工作记忆                                 | 语义记忆                             | 情景记忆                                     |
| :-------------------- | :--------------------------------------- | :----------------------------------- | :------------------------------------------- |
| **存储内容**    | 核心画像 + 线程状态 + 对话历史           | 去上下文化的稳定事实                 | 任务执行的完整轨迹与反思                     |
| **典型示例**    | `identity: user.name = 小明`           | `fact: user.city = 纽约`           | `成功: 用 RAG 回答了 Python 问题`          |
| **写入触发**    | Token 门槛 / 停顿 / 显式信号 → LLM 抽取 | 同左（固化后`fact` 分流过来）      | 每次任务执行完（有工具调用 or 报错）立即记录 |
| **写入方式**    | 更新内存`_entries` + JSON checkpoint   | 计算 Embedding + 原子写 JSON         | LLM 反思 + Embedding + 原子写 JSON           |
| **读取方式**    | **全量常驻** Prompt（无需检索）    | **向量检索** top-k，达门槛注入 | **向量检索** top-k，达门槛注入         |
| **Prompt 位置** | System 后、历史前（最稳定）              | 历史后、问题前（动态）               | 历史后、问题前（动态）                       |
| **作用域**      | 当前线程                                 | 跨线程 / 跨会话                      | 跨线程 / 跨会话                              |

---

## 六、代码层装饰器嵌套关系

[phase4_main.py](file:///Users/williamlao/Project/LLM/src/AIAgent/RAG/phase4_main.py#L457-L475) 中 Agent 的包装顺序：

```text
query_agent = agent                          # 底层 AgenticRAG
query_agent = EpisodicAgent(agent, ...)      # 包一层情景记忆
query_agent = SemanticAgent(query_agent, ...) # 再包一层语义记忆
```

调用 `query_agent.query(question, memory=memory)` 时的执行顺序：

```mermaid
sequenceDiagram
    participant Main as phase4_main
    participant SA as SemanticAgent
    participant EA as EpisodicAgent
    participant Agent as AgenticRAG
    participant WM as WorkingMemory

    Main->>SA: query(question, memory)
    SA->>SA: semantic_memory.recall(question)
    SA->>SA: 包装 _SemanticContextMemory
    SA->>EA: query(question, semantic_memory)
    EA->>EA: episodic_memory.recall(question)
    EA->>EA: 包装 _ExperienceContextMemory
    EA->>Agent: query(question, experience_memory)
    Agent->>WM: get_context_messages()
    Note over WM: 返回：核心状态 + 对话历史<br/>+ 语义事实 + 情景经验
    Agent->>Agent: LLM 推理循环（工具调用）
    Agent->>Agent: 生成最终回答
    Agent->>WM: add_turn(question, answer)
    Note over WM: 触发记忆固化门控检查<br/>满足条件 → LLM 抽取 → 分流
    Agent-->>EA: 返回 result
    EA->>EA: episodic_memory.record(question, result)
    Note over EA: 立即写入情景记忆
    EA-->>SA: 返回 result
    SA-->>Main: 返回 result
```
