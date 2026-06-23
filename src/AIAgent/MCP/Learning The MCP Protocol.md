# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

对mcp协议具体不是太清楚，从代码层面，循序渐进一步步教我，要把逻辑也讲清楚，一步步来，不用一次性回答完

### Planner Response

好的！我们来循序渐进地学习 MCP（Model Context Protocol）。先让我看看你的项目，了解一下你的技术背景，然后从最基础的概念开始。

*Listed directory [AIAgent](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent) *

*Viewed [context.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/ReActMulti/context.py) *

### Planner Response

很好，我看到你在做 AI Agent 相关的项目，已经有 ReAct、Tool 调用、多 Agent 等经验。这是一个很好的起点——MCP 本质上就是**标准化了 AI Agent 和外部工具之间的通信方式**。

---

## 第 1 步：先搞清楚 MCP 要解决什么问题

看你的代码，你现在的 Tool 调用流程大概是这样的：

```
LLM → (function call) → 你自己写的 Python 函数 → 返回结果 → 塞回 message
```

比如你在 `context.py` 里处理的 `tool_results`，就是工具执行完的结果。

**问题来了：** 如果你想接入一个别人写好的工具（比如一个文件搜索服务、一个数据库查询服务），你现在只能：

1. 去看人家的 API 文档
2. 自己写一个 wrapper 函数
3. 把它注册到你的 Agent 的 tool 列表里

**每换一个工具提供方，你就得重新写一遍对接代码。** 工具提供方也头疼——他要为每个 Agent 框架都写一套适配器。

MCP 的核心思想就是：

> **定义一个统一的协议，让任何 "工具服务端" 和任何 "AI Agent 客户端" 都能即插即用地通信，不用写定制化的适配代码。**

用一个类比：
- **没有 MCP** → 每个充电器只能给特定手机充电（私有接口）
- **有了 MCP** → 大家都用 USB-C（统一协议）

---

## 第 2 步：MCP 的角色划分

MCP 里只有三个角色：

| 角色 | 是谁 | 类比 |
|------|------|------|
| **Host** | 你的 AI 应用（比如 IDE、Chat 界面） | 用户面前的那个 App |
| **Client** | Host 内部负责和 Server 通信的模块 | App 里的 "插头" |
| **Server** | 提供工具/资源/提示的外部服务 | 外部的 "工具箱" |

关系是：`Host` 内部创建 `Client`，`Client` 和 `Server` 一对一连接。

```
┌─────────────── Host (你的 AI App) ───────────────┐
│                                                    │
│   ┌─ Client A ──┐    ┌─ Client B ──┐              │
│   │             │    │             │              │
│   └──────┬──────┘    └──────┬──────┘              │
│          │                  │                      │
└──────────┼──────────────────┼──────────────────────┘
           │                  │
     ┌─────▼─────┐     ┌─────▼─────┐
     │  Server A  │     │  Server B  │
     │ (文件操作)  │     │ (数据库)   │
     └───────────┘     └───────────┘
```

---

**这是第一个知识块。** 在继续之前，你有什么疑问吗？理解了的话，下一步我会讲 **MCP 的通信机制（JSON-RPC 2.0）**，然后用代码演示最简单的 MCP Server 是什么样的。

### User Input

下一步

### Planner Response

## 第 3 步：MCP 的通信机制 —— JSON-RPC 2.0

MCP 没有发明新的通信格式，而是直接复用了 **JSON-RPC 2.0** 这个老标准。理解了 JSON-RPC，你就理解了 MCP 消息的骨架。

JSON-RPC 只有三种消息类型：

### 1. Request（请求）— 期望对方回复

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": { "path": "/tmp/test.txt" }
  }
}
```

关键字段：
- `id`：给这次请求编个号，对方回复时带上同一个 `id`，你就知道这是哪个请求的回复
- `method`：要调用的方法名（MCP 预定义了一组方法名，比如 `tools/call`、`tools/list`）
- `params`：参数

### 2. Response（响应）— 对 Request 的回复

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      { "type": "text", "text": "文件内容: hello world" }
    ]
  }
}
```

`id` 和请求的 `id` 对应。出错的话就是 `error` 而不是 `result`：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": { "code": -32602, "message": "文件不存在" }
}
```

### 3. Notification（通知）— 单向发送，不需要回复

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

**没有 `id` 字段** —— 这就是 Notification 和 Request 的唯一区别。

---

### 对比你现有代码的理解方式

你在 [context.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/ReActMulti/context.py) 里处理的 `tool_results` 结构：

```python
# 你现有的格式
{"tool_results": [{"id": "xxx", "name": "read_file", "result": {...}}]}
```

在 MCP 里，这个流程变成了标准化的 JSON-RPC 交互：

```
你的 Agent (Client)                     工具服务 (Server)
      │                                       │
      │──── Request: tools/list ──────────────▶│   ← "你有哪些工具？"
      │◀─── Response: [read_file, ...] ───────│   ← "我有这些"
      │                                       │
      │──── Request: tools/call ──────────────▶│   ← "帮我调 read_file"
      │     {name: "read_file", args: {...}}   │
      │◀─── Response: {content: [...]} ───────│   ← "结果在这"
      │                                       │
```

**本质上就是用 JSON-RPC 替代了你自己定义的 `tool_results` 格式，并且把 "发现工具" 和 "调用工具" 分成了两个标准方法。**

---

## 第 4 步：MCP 预定义的核心方法

MCP 在 JSON-RPC 之上定义了这些关键的 `method`：

| method | 方向 | 作用 |
|--------|------|------|
| `initialize` | Client → Server | 握手，交换双方能力 |
| `notifications/initialized` | Client → Server | 通知"初始化完成" |
| `tools/list` | Client → Server | 获取 Server 提供的所有工具列表 |
| `tools/call` | Client → Server | 调用某个工具 |
| `resources/list` | Client → Server | 获取 Server 提供的资源列表 |
| `resources/read` | Client → Server | 读取某个资源 |
| `prompts/list` | Client → Server | 获取 Server 提供的 prompt 模板 |
| `prompts/get` | Client → Server | 获取某个 prompt 的内容 |

现阶段只需要关注前四个（`initialize` + `tools/*`），其余的后面再说。

---

**总结这一步：** MCP 的消息就是 JSON-RPC 2.0，核心就是 Request/Response/Notification 三种消息。MCP 在此之上定义了一组标准 method 名（`tools/list`、`tools/call` 等），让 Client 和 Server 都知道"用什么方法名做什么事"。

**下一步**我会用 Python 代码写一个最简单的 MCP Server，让你看到它实际上在做什么。准备好了就说"下一步"。

### User Input

下一步

*Viewed [context.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/ReActMulti/context.py) *

### Planner Response

好，现在用代码来揭开 MCP Server 的真面目。

## 第 5 步：从零写一个最简单的 MCP Server（不用任何 SDK）

先不用任何库，纯手写，让你看到 MCP Server **本质上就是一个读 stdin、写 stdout 的程序**，消息格式是 JSON-RPC。

*Edited relevant file*

### Planner Response

好，代码已经创建。让我解读一下核心逻辑：

---

### 一个 MCP Server 的完整生命周期

看 [step1_bare_server.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step1_bare_server.py)，整个交互流程是这样的：

```
Client                              Server (你写的程序)
  │                                    │
  │─── initialize ────────────────────▶│  ① 握手："我是 Client，协议版本 xxx"
  │◀── Response: serverInfo + caps ────│     Server 回复自己的信息和能力
  │                                    │
  │─── notifications/initialized ─────▶│  ② Client 说"OK，握手完成"
  │    (Notification, 不回复)           │     Server 收到但不回复
  │                                    │
  │─── tools/list ────────────────────▶│  ③ "你有哪些工具？"
  │◀── Response: [{name:"add",...}] ───│     Server 返回工具列表
  │                                    │
  │─── tools/call {name:"add",...} ───▶│  ④ "帮我算 add(3, 5)"
  │◀── Response: {content:[{text:"8"}]}│     Server 执行并返回结果
  │                                    │
```

### 对比你现有代码

你在 [context.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/ReActMulti/context.py#L121-L139) 里处理的 `tool_results` 结构：

```python
# 你的格式
{"id": "xxx", "name": "read_file", "result": {"ok": True, "data": "..."}}
```

MCP 的 `tools/call` 返回格式：

```python
# MCP 格式
{"content": [{"type": "text", "text": "..."}], "isError": False}
```

**核心区别：**
- 你的格式是**自定义的**（`ok`/`err`/`data`），只有你的 Agent 认识
- MCP 格式是**标准化的**（`content` 数组 + `type`），任何 MCP Client 都认识

### 三个关键认知

1. **MCP Server 就是一个普通程序**，通过 stdin/stdout 通信，没有什么魔法
2. **工具的描述用 JSON Schema**（[第 67-79 行](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step1_bare_server.py#L67-L79)），这和 OpenAI function calling 的 `parameters` 完全一样
3. **工具的执行结果统一用 `content` 数组**（[第 96-99 行](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step1_bare_server.py#L96-L99)），支持文本、图片等多种类型

---

**下一步**我会用 MCP 官方 Python SDK 重写这个 Server，你会看到 SDK 帮你省掉了多少样板代码。准备好了就说"下一步"。

### User Input

下一步

### Planner Response

## 第 6 步：用 MCP 官方 Python SDK 重写

上一步你看到了底层的 JSON-RPC 处理有多啰嗦。现在用官方 SDK 重写同样的 Server，感受一下对比。

*Edited relevant file*

### User Input

continue

*Edited relevant file*

*User accepted the command `uv run src/AIAgent/MCP/step2_sdk_server.py`*

*User accepted the command `uv run src/AIAgent/MCP/step2_sdk_server.py`*

*Viewed [step2_sdk_server.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step2_sdk_server.py) *

### Planner Response

看到你已经成功跑起来了！从输出可以看到 initialize 握手成功了：

```json
// 你手动输入的请求
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {...}}
// Server 自动回复的响应
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{...},"serverInfo":{"name":"my-first-mcp-server","version":"1.28.0"}}}
```

那些 `Invalid JSON` 报错是因为你在终端里按了回车（发送了空行 `\n`），Server 解析空行时当然会 JSON 解码失败 —— 这是正常的，不是 bug。

现在 Server 端你已经理解了。**接下来关键一步：写 Client 端**，让你看到完整的两端是怎么对接的。

## 第 7 步：写 MCP Client —— 连接 Server、发现工具、调用工具

*Edited relevant file*

*User accepted the command `uv run src/AIAgent/MCP/step3_client.py`*

### Planner Response

Server 收到了 `tools/list` 但说"初始化还没完成"。这是因为 `ClientSession` 的 `initialize()` 需要显式调用。让我修一下：

*Viewed [session.py](file:///Users/slyh/MyDir/Project/LLM/.venv/lib/python3.10/site-packages/mcp/client/session.py) *

### Planner Response

看到了！`ClientSession` 进入 context manager 时并**不会**自动调用 `initialize()`，需要手动调。让我修复并运行：

*Edited relevant file*

### User Input

Continue

### Planner Response

好，先跑一下修好的 Client，看看完整效果：

*User accepted the command `uv run src/AIAgent/MCP/step3_client.py`*

### Planner Response

完美运行！来看输出里发生了什么：

```
① 握手完成 → 协议版本 2025-11-25，Server 声明了 tools/prompts/resources 能力
② tools/list → 拿到了 add 和 multiply，SDK 自动从 type hints 生成了 JSON Schema
③ tools/call("add", {a:3, b:5})   → 返回 8.0  ✅
④ tools/call("multiply", {a:4, b:7}) → 返回 28.0 ✅
```

---

现在你已经掌握了 Server + Client 的完整流程。接下来进入最关键的一步：

## 第 8 步：把 MCP 接入 LLM Agent —— 这才是 MCP 的真正用法

前面的 Client 是**人手动调 `call_tool`**。但 MCP 真正的场景是：**LLM 决定调哪个工具，Client 只是帮 LLM 执行**。

这就是你的 ReAct Agent 在做的事，只不过现在工具来源从"本地 Python 函数"变成了"MCP Server"。

*Edited relevant file*

### Planner Response

---

### 看 [step4_agent_with_mcp.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step4_agent_with_mcp.py) 里最关键的两个地方：

**1. MCP 工具 → OpenAI function calling 格式**（[第 54-63 行](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step4_agent_with_mcp.py#L54-L63)）

```python
# MCP 的 inputSchema 就是标准 JSON Schema，直接塞进 OpenAI 的 parameters 字段
openai_tools.append({
    "type": "function",
    "function": {
        "name": tool.name,               # MCP 工具名
        "description": tool.description,  # MCP 工具描述
        "parameters": tool.inputSchema,   # MCP 的 JSON Schema → OpenAI 的 parameters
    },
})
```

这不是巧合。MCP 特意把工具描述设计成和 function calling 对齐的。

**2. LLM 决定调工具 → MCP Client 执行**（[第 93-97 行](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step4_agent_with_mcp.py#L93-L97)）

```python
# LLM 说"我要调 add(3, 5)"
# → MCP Client 通过 JSON-RPC 发给 Server
# → Server 执行 add(3, 5) 返回 8
mcp_result = await session.call_tool(fn_name, fn_args)
```

**对比你的 ReAct Agent**：你之前是 LLM 输出 function call → 你在 Python 里直接调用本地函数。现在只是多了一层 MCP 协议中转，但工具可以来自**任何 MCP Server**。

---

### 完整数据流

```
用户: "计算 (3+5)*7"
        │
        ▼
┌─── LLM (GPT-4o) ───┐
│ 看到 tools: [add,   │
│   multiply]         │    ← tools 来自 MCP Server，不是你硬编码的
│                     │
│ 决定: call add(3,5) │
└─────────┬───────────┘
          │ function_call
          ▼
┌─── MCP Client ──────┐      JSON-RPC: tools/call
│ session.call_tool(   │ ──────────────────────────▶ MCP Server
│   "add", {a:3,b:5}) │                              执行 add(3,5)
│                      │ ◀────────────────────────── 返回 "8.0"
└─────────┬────────────┘
          │ tool result → 塞回 messages
          ▼
┌─── LLM (GPT-4o) ───┐
│ 收到 add=8          │
│ 决定: call           │
│  multiply(8, 7)     │
└─────────┬───────────┘
          │ ... 同样通过 MCP ...
          ▼
最终回答: "结果是 56"
```

---

**总结到这一步**，你已经从底层到应用层完整理解了：

| 步骤 | 你学到了什么 |
|------|------------|
| Step 1 | MCP 要解决的问题（工具的标准化通信） |
| Step 2 | 协议基础（JSON-RPC 2.0 的三种消息） |
| Step 3 | 手写 Server（看清底层就是 stdin/stdout + JSON） |
| Step 4 | SDK Server（装饰器注册工具，SDK 处理协议细节） |
| Step 5 | Client 端（连接、握手、发现工具、调用工具） |
| Step 6 | **接入 LLM Agent**（MCP 工具 → function calling → LLM 决策） |

**下一步**可以讲 MCP 的另外两大能力 **Resources（资源）** 和 **Prompts（提示模板）**，或者讲 **传输层（stdio vs HTTP）**。你想先看哪个？

### User Input

先讲传输层

*Edited relevant file*

*Edited relevant file*

*Edited relevant file*

*Edited relevant file*

*Edited relevant file*

*Edited relevant file*

*Edited relevant file*

*Viewed [Learning The MCP Protocol.md](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/Learning%20The%20MCP%20Protocol.md) *

### Planner Response

好，接下来讲传输层。前面你一直用的是 **stdio**（标准输入输出），但 MCP 还支持通过 HTTP 通信。

## 第 9 步：MCP 传输层 —— 消息怎么从 A 到 B

MCP 的设计是**协议层和传输层分离**的。JSON-RPC 消息是"说什么"，传输层是"怎么送达"。

### 三种传输方式对比

| 传输方式 | 通信方式 | 适用场景 | Client 和 Server 的关系 |
|---------|---------|---------|----------------------|
| **stdio** | 子进程的 stdin/stdout 管道 | 本地工具（CLI、文件操作） | 同一台机器，1对1 |
| **SSE** *(已淘汰)* | HTTP GET(SSE下行) + HTTP POST(上行) | 远程服务（旧方案） | 可跨网络，但有限制 |
| **Streamable HTTP** | HTTP POST（双向，支持 SSE 流） | 远程服务（新标准） | 可跨网络，支持多 Client |

### stdio 你已经很熟了

```
Client 进程
  │ fork/exec → 启动 Server 子进程
  │
  │  stdin  ────▶ │
  │  stdout ◀──── │  Server 子进程
  │               │
```

就是管道。优点是简单，缺点是 Server 必须在本地，且一个 Client 绑定一个 Server 进程。

### Streamable HTTP —— 远程 MCP 的新标准

这是 MCP 2025 年推出的新传输方式，取代了旧的 SSE 方案。核心思路：

```
Client                                    Server (HTTP 服务)
  │                                            │
  │─── POST /mcp ─────────────────────────────▶│
  │    Body: {"jsonrpc":"2.0","method":...}     │
  │                                            │
  │◀── 200 OK ────────────────────────────────│
  │    Body: {"jsonrpc":"2.0","result":...}     │  ← 简单请求：普通 JSON 响应
  │                                            │
  │─── POST /mcp ─────────────────────────────▶│
  │    Body: {"jsonrpc":"2.0","method":...}     │
  │                                            │
  │◀── 200 OK (Content-Type: text/event-stream)│  ← 长任务：SSE 流式响应
  │    data: {"jsonrpc":"2.0",...}              │     可以发进度、中间结果
  │    data: {"jsonrpc":"2.0","result":...}     │     最后发最终结果
  │                                            │
```

**一句话总结：所有请求都是 POST 到同一个端点，Server 可以选择返回普通 JSON 或 SSE 流。**

用代码看：

*Edited relevant file*

*Edited relevant file*

### Planner Response

先启动 HTTP Server：

*User accepted the command `uv run src/AIAgent/MCP/step5_http_server.py`*

### Planner Response

Server 已启动在 `http://127.0.0.1:8000`。现在运行 Client 连接它：

*User accepted the command `uv run src/AIAgent/MCP/step5_http_client.py`*

### Planner Response

完美！现在关掉 HTTP Server：

### Planner Response

看 Server 日志，HTTP 传输下每个操作对应的 HTTP 请求非常清晰：

```
# 握手
POST /mcp  → 200 OK          ← initialize 请求 + 响应
POST /mcp  → 202 Accepted    ← initialized 通知（Notification 没有 Response body）
GET  /mcp  → 200 OK          ← Client 打开 SSE 通道，用于接收 Server 主动推送

# 业务
POST /mcp  → 200 OK          ← tools/list
POST /mcp  → 200 OK          ← tools/call (add)
POST /mcp  → 200 OK          ← tools/call (multiply)

# 断开
DELETE /mcp → 200 OK         ← Client 关闭会话
```

---

### 传输层的核心认知

对比两个 Client 的代码，你会发现**建立连接之后，所有 API 调用完全一样**：

```python
# stdio（step3）
async with stdio_client(server_params) as (read, write):      # ← 只有这行不同
    async with ClientSession(read, write) as session:
        await session.initialize()          # 完全一样
        await session.list_tools()          # 完全一样
        await session.call_tool(...)        # 完全一样

# HTTP（step5）
async with streamablehttp_client(url) as (read, write, _):    # ← 只有这行不同
    async with ClientSession(read, write) as session:
        await session.initialize()          # 完全一样
        await session.list_tools()          # 完全一样
        await session.call_tool(...)        # 完全一样
```

**这就是传输层和协议层分离的威力**：换传输方式只改一行连接代码，业务逻辑零修改。

---

### 什么时候用哪种？

| 场景 | 选哪个 | 理由 |
|------|--------|------|
| IDE 里的本地工具 | **stdio** | 简单，不用起 HTTP 服务 |
| 团队共享的工具服务 | **Streamable HTTP** | 一个 Server 多个 Client |
| 部署在云端的工具 | **Streamable HTTP** | 跨网络访问 |
| 需要认证/鉴权 | **Streamable HTTP** | 可以加 HTTP headers |

---

**下一步**可以讲 **Resources（资源）和 Prompts（提示模板）**——MCP 除了 Tools 之外的另外两大能力。准备好了就说"下一步"。

### User Input

下一步

*User accepted the command `uv run src/AIAgent/MCP/step5_http_server.py`*

*Viewed [step5_http_client.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step5_http_client.py) *

*User accepted the command `uv run src/AIAgent/MCP/step5_http_client.py`*

*User accepted the command `uv run src/AIAgent/MCP/step5_http_client.py`*

### Planner Response

看到你已经跑了好几次 HTTP client，每次 Server 日志里都会创建新的 session ID 然后销毁——说明你已经完全理解 HTTP 传输了。

接下来讲 MCP 除了 Tools 之外的另外两大能力。

## 第 10 步：Resources（资源）和 Prompts（提示模板）

MCP Server 可以向 Client 暴露三种东西：

```
MCP Server 的三种能力
├── Tools      ← 你已经掌握了：LLM 主动调用，执行某个操作
├── Resources  ← 现在讲：提供数据给 LLM 读取（类似 REST API 的 GET）
└── Prompts    ← 现在讲：提供预制的 prompt 模板
```

### Resources vs Tools 的区别

这是最容易混淆的地方，用一个直觉来区分：

| | Tools | Resources |
|--|-------|-----------|
| 类比 | **函数调用**（做某件事） | **文件/数据读取**（拿某个东西） |
| 谁决定用 | **LLM** 在对话中自己决定 | **用户或应用**在对话前选好 |
| 有副作用吗 | 可能有（写文件、发请求） | 没有（只读） |
| 例子 | `run_sql(query)` | `db://tables/users` 的 schema |

**一句话：Tools 是"动词"，Resources 是"名词"。**

*Edited relevant file*

*User accepted the command `uv run src/AIAgent/MCP/step6_resources_prompts.py`*

### Planner Response

运行成功！来看关键的逻辑：

---

### Resources 的三种形态

看 [step6_resources_prompts.py](file:///Users/slyh/MyDir/Project/LLM/src/AIAgent/MCP/step6_resources_prompts.py) 的输出：

**1. 静态 URI 资源** — `config://app/settings`
```python
@server.resource(uri="config://app/settings")
def get_app_settings() -> str:
    return json.dumps({...})
```
URI 固定，每次 `resources/read` 时调用函数获取内容。

**2. 动态数据资源** — `data://system/status`
```python
@server.resource(uri="data://system/status")
def get_system_status() -> str:
    return json.dumps({"timestamp": datetime.now().isoformat(), ...})
```
每次读取返回不同内容（时间戳变了）。

**3. URI 模板资源** — `users://{user_id}/profile`
```python
@server.resource(uri="users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:   # URI 里的 {user_id} → 函数参数
    ...
```
类似 REST API 的路径参数。Client 请求 `users://alice/profile`，函数收到 `user_id="alice"`。

### Prompts 的用法

```python
@server.prompt(name="code-review")
def code_review_prompt(code: str, language: str = "python") -> str:
    return f"请审查以下 {language} 代码..."
```

Client 调用 `prompts/get("code-review", {"code": "...", "language": "python"})`，拿到渲染好的 messages，可以**直接拼接到 LLM 的对话历史**里。

### 在 Agent 中怎么用？

```
┌─── Agent 启动时 ──────────────────────────────────────┐
│                                                        │
│  1. resources/list → 发现有哪些数据源                    │
│  2. resources/read → 把关键数据塞进 system prompt        │
│     （比如数据库 schema、用户配置、项目文档）              │
│                                                        │
│  3. prompts/list → 发现有哪些 prompt 模板               │
│     → 展示给用户选择，或 Agent 自动选用                  │
│                                                        │
│  4. tools/list → 拿到工具列表给 LLM                     │
│                                                        │
├─── Agent 对话循环中 ──────────────────────────────────┐
│                                                        │
│  5. LLM 决定调 tools/call → 执行工具                    │
│  6. 需要时 resources/read → 补充上下文                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Tools 是 LLM 在对话中主动选择的，Resources 和 Prompts 通常是应用/用户在对话前就决定好的。**

---

到这里，MCP 的三大能力你都了解了。**下一步**可以讲：
- **实战：写一个真正有用的 MCP Server**（比如文件系统操作、数据库查询）
- **MCP 的高级特性**（Sampling、Roots、Notifications 等）

你想先看哪个？