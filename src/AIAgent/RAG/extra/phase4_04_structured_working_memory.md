# Phase 4.4：缓存友好的结构化工作记忆

## 1. 为什么不每轮都抽取

把姓名、偏好、约束和待办从对话中抽取成结构化状态，确实是工程中常见的做法。但“每轮都调一次 LLM”会带来两个问题：

- 每轮固定增加成本和延迟，即使只是普通闲聊。
- 如果结构化状态每轮改变，主 Agent 的 Prompt 前缀也会频繁改变，削弱前缀缓存的价值。

本实现参考 Claude Code，使用「Token 增量门槛 + 对话自然停顿点/工具调用次数限制」的双轨触发策略：

1. **Token 增量硬指标**：自上次提取后，未处理的 pending 回合新增 Token 数必须达到门槛（默认 150 tokens），避免频繁写入破坏 Prompt 缓存。
2. **触发机制（满足 Token 门槛后二选一）**：
   - **自然停顿点 (Natural Break)**：最新一轮助手的回答是普通自然文本，未发生 tool_call 或检索动作（表明当前工作阶段性结束），则触发提取。
   - **工具调用次数超限兜底**：如果大模型在对话中不断调用工具，但累计的工具调用次数达到了兜底阈值（默认 3 次），也会强制触发提取。

```text
新回合写入
   ├─ 检查 Token 增量是否满足 threshold (默认 150 tokens)？
   │      ├─ 否 → 只进入 pending，不调用抽取模型
   │      └─ 是 → 检查触发条件：
   │                ├─ 条件 A：当前属于对话自然停顿 (不含 tool_call 等) → 触发批量提取
   │                ├─ 条件 B：累计工具调用次数达到兜底阈值 (默认 3 次) → 触发批量提取
   │                └─ 否则 → 只进入 pending
```

## 2. 为什么 Prompt 顺序影响缓存

基于精确前缀匹配的 Prompt Cache 只能复用从开头连续不变的部分。因此上下文按更新频率排列：

```text
稳定 system / tools
  → 低频更新的结构化状态
  → 低频更新的历史摘要
  → 按时间追加的近期原文
  → 当前问题
```

没有信息变化时，抽取器返回空 operations，已缓存的结构化状态文本不会重建。只有真正发生 upsert/delete 或历史摘要压缩时，对应前缀才会改变。

## 3. 状态模型

每个条目使用 `category + key` 唯一定位：

```json
{
  "category": "preference",
  "key": "user.favorite_color",
  "value": "绿色",
  "created_turn": 1,
  "updated_turn": 6
}
```

支持六类信息：

- `identity`：身份、姓名、称呼
- `preference`：稳定偏好和习惯
- `constraint`：必须遵守或明确禁止的要求
- `decision`：已做出的选择
- `pending_task`：尚未完成的待办
- `fact`：其他对后续有用的明确事实

`upsert` 对相同 key 使用最新明确值；`delete` 精确删除。整个批次通过 Pydantic 校验后才应用，解析失败不会留下半更新状态。

## 4. 敏感信息不能进入工作记忆

仅在 Prompt 中告诉 LLM“不要保存密码”不够。应用操作前还会用确定性规则拦截密码、API Key、访问令牌、私钥和银行卡等敏感条目。

这个过滤器只保护「结构化状态」；用户刚输入的原始文本仍可能出现在近期对话窗口中。生产系统还需在输入层另行做秘密检测和脱敏。

## 5. 运行实验

```bash
uv run python src/AIAgent/RAG/phase4_main.py \
  --strategy summary \
  --structured-state \
  --structured-state-file memory.json \
  --token-budget 1200 \
  --summary-token-budget 400
```

可用命令：

- `/state`：查看结构化条目、更新轮次和 pending 数量。
- `/extract`：手动抽取未达到门槛的 pending 回合。
- `/forget preference user.favorite_color`：不调用 LLM，确定性删除指定条目。
- `/clear`：清空结构化状态、pending、摘要和近期原文。

## 6. 工程边界

当前 CLI 在门控触发后同步执行抽取，便于学习、观察和确定性测试。在在线服务中，可以保留 `WorkingStateExtractor` 接口，把同一批操作放到队列或后台 write-behind worker 中执行。

这一层仍然是会话内工作记忆，不持久化。下一阶段的情景记忆会开始记录任务经验，并在新任务开始时检索相似历史。

## 7. 架构定位

在整个 Agent 记忆体系中，当前模块具有承上启下的独特定位：

### 7.1 物理载体上：属于 4.1 短期工作记忆 (Working Memory) 并向 4.3 跨越
* **会话级物理管理，可选本地文件持久化**：记忆条目（`_entries`）在运行态内存中字典式维护，同时可通过 `--structured-state-file` 指定本地 JSON 路径。启用后，系统在初始化时自动从文件加载已有记忆，并在发生修改（提取、手动删除）或会话清空（`/clear`）时实时写回文件，实现了跨会话的记忆存续。
* **无损呈现方式**：它在运行态下作为 System Message 注入到每一轮 LLM 提示词的头部，担当“全局临时便签板”的角色。这对应了路线图中 *“将关键信息提取到工作记忆区”* 的核心概念。

### 7.2 逻辑设计上：是 4.3 语义记忆 (Semantic Memory) 的前哨站与桥梁
* **数据内容**：它提取并维护的是用户画像（如 `identity` 身份、`preference` 偏好）和硬约束（`constraint`），这些内容在本质上属于长期语义记忆的核心。
* **解决机制**：它在内存中实现的 `upsert`（对相同 key 覆盖更新）和 `delete` 逻辑，正是语义记忆在面对新旧事实矛盾时，进行 **“冲突检测与版本解决”** 的典型实践。

### 7.3 Phase 4 记忆文件分工一览
* **短期工作记忆 (Working Memory) —— 负责“会话内的上下文管理与临时便签”**
  * `ConversationWindowMemory` 👉 滑动窗口策略（按轮数硬裁剪）
  * `TokenBudgetMemory` 👉 Token 预算策略（按 Token 计数硬截断）
  * `SummaryBufferMemory` 👉 摘要缓冲策略（Summary Buffer，压缩非结构化历史）
  * `StructuredWorkingMemory` 👈 **当前模块**（结构化关键信息提取与缓存优化，用短期的容器，承载并实验长期的内容）
* **情景记忆 (Episodic Memory) —— 负责“回忆过去的任务执行经验”**
  * （未来开发）将执行日志与反思，连同向量索引存入磁盘，提供 few-shot 经验召回。
* **语义记忆 (Semantic Memory) —— 负责“持久化的用户画像”**
  * （已初步实现文件级持久化）将当前模块内存中的 KV 条目持久化写入 JSON 文件。未来可进一步扩展，将其保存至 SQLite 或向量数据库，以满足大规模多模态、跨会话和跨设备的永久记忆检索保留。
