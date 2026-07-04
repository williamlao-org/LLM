# Phase 4.4：缓存友好的结构化工作记忆

## 1. 为什么不每轮都抽取

把姓名、偏好、约束和待办从对话中抽取成结构化状态，确实是工程中常见的做法。但“每轮都调一次 LLM”会带来两个问题：

- 每轮固定增加成本和延迟，即使只是普通闲聊。
- 如果结构化状态每轮改变，主 Agent 的 Prompt 前缀也会频繁改变，削弱前缀缓存的价值。

本实现使用「显式信号 + 定期批处理」：

```text
成功问答
   ├─ 出现姓名/偏好/约束/决定/待办/更正/忘记信号
   │      → 立即批量抽取当前 pending
   ├─ 没有信号，pending 达到 5 轮
   │      → 兜底批量抽取
   └─ 其他情况
          → 只进入 pending，不调用抽取模型
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
