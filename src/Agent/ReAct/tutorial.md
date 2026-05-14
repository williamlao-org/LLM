### **先用纸笔画清楚"循环-状态-接口"三件事，再动手；API 细节用一份精简速查表代替反复查文档**

你已经把目标定得很清楚了：不是"做出来一个能跑的 Agent"，而是"通过亲手实现，理解 ReAct 范式里每一个设计选择的必然性"。所以真正的难点不在代码量，而在动手之前能不能想透。下面是我给你的具体建议，分成三块：**动手前怎么思考、过程中怎么避免浪费时间、做完后怎么和论文对照**。

---

### **一、动手前：把 Agent 当成一个"状态机 + 解释器"来设计**

ReAct 的本质是一个循环：**LLM 生成文本 → 解析出动作 → 执行动作 → 把结果塞回上下文 → 再让 LLM 生成**。所以在写任何一行代码前，先在纸上回答下面这几个问题，每个问题都要能用一两句话说清楚，说不清楚就说明还没想透：

**1. 你的"会话状态"里到底有什么？**
一个 Agent run 从头到尾要维护哪些东西？至少包括：用户原始问题、累计的 Thought/Action/Observation 历史、已用 token 数、当前迭代轮次、工具调用失败计数。建议把它设计成一个明确的数据类（dataclass），而不是散落在各处的变量。这个状态对象的字段一旦定下来，整个主循环的样子就出来了。

**2. Prompt 的结构长什么样？**
ReAct 的 prompt 不是随便写的，它有严格的格式契约：系统提示里告诉模型可用工具、输出格式（`Thought: ... Action: ... Action Input: ...`），然后用 few-shot 例子锚定格式。你需要先把这个 prompt 模板的骨架画出来——尤其是 **Observation 怎么拼回去**（是追加在 assistant 消息后面，还是作为新的 user 消息？两种做法都见过，各有取舍，你要自己选一种并说出理由）。

**3. 解析器的边界在哪里？**
模型输出是自由文本，你要从中提取结构。先想清楚三种情况怎么处理：
- 正常输出 `Thought / Action / Action Input`：怎么用正则切？
- 模型直接给出 `Final Answer`：怎么识别并终止循环？
- 模型输出格式错乱（漏了 Action Input、Action 名字写错、JSON 不合法）：你是重试、是把错误信息当 Observation 喂回去让它自我修正、还是直接报错？

第三种情况是 ReAct 在工程上最容易翻车的地方，**你提前定好策略，写代码时就不会被卡住**。

**4. 工具的统一接口长什么样？**
五个工具差异很大（搜索返回文本、计算器返回数字、Python 执行器有副作用、文件读写有 IO、HTTP 有网络错误），但它们对主循环必须呈现**同一张脸**。建议先定义一个 Tool 抽象：名字、描述（给 LLM 看的）、参数 schema、`run(input) -> str` 方法、以及一个统一的异常包装。所有工具最终都把结果转成字符串 Observation。这一步想清楚了，加第六个、第七个工具就是几十行的事。

**5. 失败重试、token 预算、最大迭代——这三个"守卫"放在循环的哪一层？**
- 工具失败重试：放在工具调用那一层（指数退避 + 最多 N 次），失败后是抛异常还是把错误信息作为 Observation 喂回？后者更符合 ReAct 精神（让模型自己看到错误并调整）。
- Token 预算：在每次调用 LLM 前检查累计 token，超了就强制收尾（让模型基于现有信息给 Final Answer）。
- 最大迭代：放在主循环的 while 条件里，到了上限同样强制收尾。

把这三层守卫画在循环图上，你会发现它们是**三个不同层级的熔断器**，不能混在一起写。

**6. 流式输出怎么和"先解析再执行"协调？**
这是最容易想不清楚的一点。LLM 流式吐字时，你**不能**等它吐完再解析——但你也**必须**等到 `Action Input` 完整了才能执行工具。建议的做法是：流式把字符实时打到终端给用户看，同时在内部缓冲完整文本；遇到 `\nObservation:` 这种分隔符或检测到 Action Input 闭合时，停止接收（或忽略后续），然后做解析和执行。先把这个"边播边缓冲、到点切断"的逻辑想清楚，再写代码会非常顺。

把上面 6 个问题在纸上写完答案，你的架构图基本就出来了：一个主循环 + 一个状态对象 + 一个 prompt 构造器 + 一个解析器 + 一个工具注册表 + 三层守卫 + 一个流式打印器。**七个模块、各司其职，互相之间用清晰的数据结构通信。** 这就是你说的"架构能力"——不是用了什么高级模式，而是边界划得干净。

---

### **二、过程中：用一份"API 速查卡"代替反复查文档**

你不想浪费时间在查 API 上，这个完全可以解决。建议你在开工前花 20 分钟做一件事：**把这次会用到的所有外部 API 调用，做成一份单页速查**，贴在屏幕边上。具体包括：

- **LLM 调用**：DeepSeek/OpenAI 的 chat completions 请求体（messages 结构、stream=True 时返回的 chunk 格式、怎么累加 delta.content、怎么读 usage 里的 token 数）。把一个最小可运行的非流式调用和一个流式调用的代码片段抄一份在卡上即可，之后照着改。
- **搜索工具**：选一个你打算用的（比如 Tavily、SerpAPI、DuckDuckGo），把 endpoint、必填参数、返回字段挑出来。
- **HTTP 请求工具**：`requests` 的 get/post 签名、超时参数、异常类型（ConnectionError、Timeout、HTTPError）。
- **Python 执行器**：`exec` 和 `eval` 的区别、怎么捕获 stdout（`io.StringIO` + `contextlib.redirect_stdout`）、安全沙箱的最低限度（限制 builtins、设超时）。
- **文件读写**：就是 `open` 的几种模式，外加路径白名单怎么校验。

下面这份可以直接当你的"API 速查卡"用。

#### **API 速查卡**

**0. 环境变量约定**

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1

DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com

TAVILY_API_KEY=tvly-...
```

如果你想用同一套 `OpenAI` SDK 切 OpenAI/DeepSeek，核心只换两样：`api_key` 和 `base_url`。

---

**1. LLM：Chat Completions 非流式**

OpenAI:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用一句话解释 ReAct。"},
]

resp = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages,
    temperature=0.2,
)

text = resp.choices[0].message.content
usage = resp.usage
print(text)
print(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
```

DeepSeek:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用一句话解释 ReAct。"},
    ],
    temperature=0.2,
    response_format={"type": "json_object"},  # 需要 JSON 输出时再开
)

text = resp.choices[0].message.content
reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
usage = resp.usage
```

消息结构记这一版就够了：

```python
messages = [
    {"role": "system", "content": "...全局规则..."},
    {"role": "user", "content": "...用户问题..."},
    {"role": "assistant", "content": "...模型上轮输出..."},
    {"role": "user", "content": "Observation: ...工具结果..."},
]
```

如果你改用 API 原生 tool calling，工具结果消息会长这样：

```python
{"role": "tool", "tool_call_id": "...", "content": "...工具结果..."}
```

但你现在这个手写 ReAct 练习，先把 Observation 当普通 `user` 消息追加，最简单、最容易调试。

---

**2. LLM：Chat Completions 流式**

OpenAI:

```python
stream = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages,
    stream=True,
    stream_options={"include_usage": True},
)

parts = []
usage = None

for chunk in stream:
    if chunk.usage is not None:
        usage = chunk.usage
        continue

    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta
    piece = delta.content or ""
    if piece:
        print(piece, end="", flush=True)
        parts.append(piece)

text = "".join(parts)
```

DeepSeek 普通输出同样读 `delta.content`；如果使用思考模式，还可能有 `delta.reasoning_content`：

```python
stream = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    stream=True,
    stream_options={"include_usage": True},
)

reasoning_parts = []
answer_parts = []
usage = None

for chunk in stream:
    if chunk.usage is not None:
        usage = chunk.usage
        continue

    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta
    reasoning_piece = getattr(delta, "reasoning_content", None)
    answer_piece = delta.content or ""

    if reasoning_piece:
        reasoning_parts.append(reasoning_piece)
    if answer_piece:
        print(answer_piece, end="", flush=True)
        answer_parts.append(answer_piece)

reasoning = "".join(reasoning_parts)
text = "".join(answer_parts)
```

流式实现时记住三句话：
- 展示给用户：边收到 `delta.content` 边 `print(..., flush=True)`。
- 内部解析：同时把所有片段 append 到 `parts`，最后 `text = "".join(parts)` 再解析 JSON/Action。
- token 统计：打开 `stream_options={"include_usage": True}` 后，最后一个 usage chunk 才有总 token；如果流被中断，可能拿不到这个 usage。

---

**3. 搜索工具：Tavily**

HTTP endpoint:

```text
POST https://api.tavily.com/search
Authorization: Bearer $TAVILY_API_KEY
Content-Type: application/json
```

最小请求:

```python
import os
import requests

def tavily_search(query: str, max_results: int = 5) -> str:
    resp = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    lines = []
    for item in data.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"- {title}\n  {url}\n  {content}")
    return "\n".join(lines)
```

常用请求字段：
- `query`: 必填，搜索词。
- `search_depth`: `"basic"` 更快，`"advanced"` 更深。
- `max_results`: 返回条数。
- `include_answer`: 是否返回 Tavily 生成的摘要答案。
- `include_raw_content`: 是否带网页正文，`False` 省 token。
- `topic`: 可选，常见值如 `"general"` / `"news"`。
- `days`, `start_date`, `end_date`: 做新闻或时间过滤时再用。

常用返回字段：
- 顶层：`query`, `answer`, `results`, `response_time`, `usage`, `request_id`。
- 每条结果：`title`, `url`, `content`, `score`, `raw_content`, `favicon`。

对 Agent 来说，Observation 不要塞完整 JSON，建议压成：

```text
Search results:
1. title
   url
   snippet/content
2. ...
```

---

**4. HTTP 请求工具：requests**

```python
import requests

def http_get(url: str, params: dict | None = None) -> dict:
    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": "mini-react-agent/0.1"},
        timeout=(5, 20),  # (连接超时, 读取超时)
    )
    resp.raise_for_status()
    return {
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", ""),
        "text": resp.text[:4000],
    }

def http_post(url: str, payload: dict) -> dict:
    resp = requests.post(url, json=payload, timeout=(5, 20))
    resp.raise_for_status()
    return resp.json()
```

异常捕获模板：

```python
import requests

try:
    data = http_get("https://example.com")
except requests.exceptions.Timeout as exc:
    result = f"HTTP timeout: {exc}"
except requests.exceptions.ConnectionError as exc:
    result = f"HTTP connection error: {exc}"
except requests.exceptions.HTTPError as exc:
    status = exc.response.status_code if exc.response is not None else "unknown"
    result = f"HTTP status error {status}: {exc}"
except requests.exceptions.RequestException as exc:
    result = f"HTTP request error: {exc}"
```

给 Agent 用时，永远设置 `timeout`，永远 `raise_for_status()`，Observation 里只放状态码、content-type、截断后的正文。

---

**5. Python 执行器：exec / eval / stdout 捕获**

区别：
- `eval(expr, globals, locals)`: 只执行一个表达式，有返回值，比如 `"1 + 2"`。
- `exec(code, globals, locals)`: 执行一段语句，没有直接返回值，比如 `"x = 1\nprint(x)"`。

捕获 stdout:

```python
import contextlib
import io

def run_python(code: str) -> str:
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "print": print,
        "sorted": sorted,
        "round": round,
    }
    globals_ = {"__builtins__": safe_builtins}
    locals_ = {}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, globals_, locals_)
    except Exception as exc:
        return f"Python error: {type(exc).__name__}: {exc}"

    output = buf.getvalue()
    if not output and "_" in locals_:
        output = repr(locals_["_"])
    return output[:4000] or "(no output)"
```

最低限度安全提醒：
- 限制 `__builtins__` 只能降低风险，不能算真正沙箱。
- 不要暴露 `open`, `import`, `exec`, `eval`, `compile`, `input`, `globals`, `locals`, `vars`, `dir`, `getattr`, `setattr`, `delattr`。
- 真要执行不可信代码，应放到独立进程/容器里，并加超时、内存限制、文件系统限制。

---

**6. 文件读写：open + 路径白名单**

常用模式：
- `"r"`: 读文本，文件必须存在。
- `"w"`: 写文本，覆盖已有内容。
- `"a"`: 追加文本。
- `"x"`: 创建新文件，已存在则报错。
- `"rb"` / `"wb"`: 二进制读写。
- `encoding="utf-8"`: 文本文件默认都显式写上。

路径白名单模板：

```python
from pathlib import Path

ROOT = Path("./workspace").resolve()

def safe_path(user_path: str) -> Path:
    path = (ROOT / user_path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        raise ValueError("path escapes workspace")
    return path

def read_file(user_path: str) -> str:
    path = safe_path(user_path)
    return path.read_text(encoding="utf-8")[:8000]

def write_file(user_path: str, content: str) -> str:
    path = safe_path(user_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(ROOT)}"
```

给 Agent 用时，文件 Observation 也要截断；读文件最多返回前几千字符，写文件返回路径和字节数即可。

---

**7. 这次项目最推荐的工具选择**

为了先把 ReAct 主循环跑通，建议第一版只接这些：
- `search(query) -> str`: Tavily，返回 3-5 条标题、URL、摘要。
- `python(code) -> str`: 只给数学/字符串处理用，先别开放文件和网络。
- `read_file(path) -> str`: 路径限定在项目工作目录。
- `write_file(path, content) -> str`: 路径限定在项目工作目录。
- `http_get(url) -> str`: 第二阶段再加，因为网络错误和网页正文清洗会分散注意力。

不要一开始就接 OpenAI/DeepSeek 原生 function calling。你现在练的是 ReAct 循环本身，先让模型输出你规定的 JSON：

```json
{
  "reasoning": "...",
  "tool_call": {
    "name": "search",
    "arguments": {"query": "..."}
  },
  "final_answer": null
}
```

然后你自己 `json.loads()`、查工具表、执行工具、把 `tool_result` 追加回 messages。这个闭环比 API 原生 tool calling 更能帮你理解 Agent 是怎么转起来的。

参考资料：[OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat)、[OpenAI streaming chunks](https://platform.openai.com/docs/api-reference/chat-streaming?api-mode=chat)、[DeepSeek Chat Completion API](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion)、[DeepSeek streaming example](https://api-docs.deepseek.com/guides/reasoning_model_api_example_streaming)、[Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)、[Python `eval/exec`](https://docs.python.org/3/library/functions.html)、[Python `contextlib.redirect_stdout`](https://docs.python.org/3/library/contextlib.html)、[Python `pathlib.Path.resolve/relative_to`](https://docs.python.org/3/library/pathlib.html)。

这张卡放在手边，整个项目你几乎不需要再开浏览器查任何 API。这比"边写边查"的效率高一个数量级，而且能让你保持在思考主循环上，而不是被各种参数细节打断心流。

---

### **三、动手顺序：自底向上 vs 自顶向下，建议混合**

我推荐的顺序，**不是从主循环开始写**，因为没有工具就没法测；也不是从工具开始堆，因为容易陷入细节忘了整体。混合做法：

1. **先写一个最小闭环（30 分钟）**：硬编码一个工具（比如计算器），prompt 里只放一个 few-shot 例子，主循环只支持非流式、无重试、无预算。能让 "1+1=?" 跑通从 Thought 到 Final Answer 的一轮迭代。这一步的目的是**验证你对 ReAct 流程的理解是不是真的对了**。

2. **然后补齐解析器的鲁棒性**：手动构造几个"模型输出畸形"的字符串，喂给解析器，看它怎么反应。这一步会逼你想清楚边界情况。

3. **再抽象工具接口**：把硬编码的计算器改造成 Tool 类，注册到工具表，主循环改成"查表执行"。这一步做完，加新工具就是模板化的。

4. **加守卫**：迭代上限、token 预算、工具重试，分三次提交，每次只加一层，加完跑一遍冒烟测试。

5. **最后做流式**：流式是叠加在已有循环上的"输出层改造"，留到最后做最稳妥。

每完成一步，停下来问自己一句：**"如果让我现在跟别人讲清楚这一块为什么这么设计，我说得出来吗？"** 说不出来就回头补思考，不要硬推进。

---

### **四、做完后：带着具体问题去读论文**

你计划写完再读原论文，这个顺序非常对。读的时候不要顺着读，**带着你写代码时遇到的真实困惑去检索性地读**，比如：

- 为什么 Thought 要显式输出？——回想你自己实现时，如果不让模型先写 Thought 直接出 Action，会发生什么？（大概率是模型乱选工具）
- 为什么 Observation 要严格格式化？——回想你解析器最容易崩在哪里？（格式不一致就崩）
- 论文里的 prompt 例子和你写的有什么不同？差异背后是什么考量？
- 论文报告的失败模式（hallucinated action、reasoning error），你在自己的 Agent 上复现得出来吗？

这种"带着伤痕读论文"的体验，比顺着读一遍深刻十倍。

---

最后给你一个心态上的提醒：**这个项目最有价值的产出不是那份代码，而是你在动手前画的那张架构图、和动手中你给自己写的那些设计决策注释**（比如"这里选择把工具错误作为 Observation 喂回而不是抛异常，因为……"）。代码写完跑通的那一刻满足感很短，但那些笔记和决策记录会留很久。建议你专门开一个 `DESIGN.md`，边想边写进去，这是你这次练习真正要带走的东西。
