# Phase 4.2：Token Budget Memory

## 1. 为什么按轮数还不够

「最近 3 轮」看似稳定，但一轮可能只有几个字，也可能包含整篇文档。模型真正受限的是 Token 数，不是对话轮数。

Token Budget Memory 的规则是：

```text
写入新的 [user, assistant]
        ↓
计算所有历史问答的 Token
        ↓
若超过预算，从最早的完整问答开始淘汰
        ↓
直到 Token 总数 <= 预算
```

不会只删 user 或只删 assistant，因为那会破坏对话语义。如果最新一轮单独就超过预算，它也不会被截成半轮保留。

## 2. 这个预算是历史预算

`max_tokens` 不是模型公布的完整 context window。一个完整请求还要容纳：

```text
模型上下文上限
- system prompt
- tool schemas
- 当前用户问题
- 预期回答空间
- 安全余量
= 可分配给历史问答的 max_tokens
```

本步不自动推导生产环境预算，而是先用小预算观察裁剪行为。

## 3. Token 计数器为什么可注入

默认本地估算器使用一个可解释的粗略规则：

- 非 ASCII 字符：约 1 字符/token
- ASCII 字符：约 4 字符/token
- 每条 message 额外计入 4 个结构 Token

不同模型的 tokenizer 不同，因此这只是学习和离线测试用的估算。`TokenBudgetMemory` 接受 `token_counter` 参数，以后可直接注入 DeepSeek 对应的精确 tokenizer，无需改动记忆裁剪逻辑。

## 4. 对比实验

按轮数运行：

```bash
uv run python src/AIAgent/RAG/phase4_main.py --strategy turns --max-turns 3
```

按 Token 预算运行：

```bash
uv run python src/AIAgent/RAG/phase4_main.py --strategy tokens --token-budget 120
```

两边都输入三轮对话，但让其中一轮特别长。然后使用 `/memory` 比较：轮数策略仍会保留三轮，Token 策略会根据实际内容长度更早地淘汰旧问答。

## 5. 下一个问题

无论按轮数还是按 Token 裁剪，被淘汰的信息都会彻底消失。下一步将用 Summary Buffer 把旧问答压缩成摘要，在节省 Token 的同时保留关键信息。
