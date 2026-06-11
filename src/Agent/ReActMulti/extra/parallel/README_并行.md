# Python 并行速查笔记

> 配套代码：本目录下 `parallel_0*.py`，每个都能直接 `python xxx.py` 跑。

## 一句话决策

**任务运行时，CPU 在「等」还是在「算」？** —— 这是一切的起点。

```
            要并发/并行
                │
        任务在等 还是 在算？
         ┌──────┴──────┐
       在等(IO)        在算(CPU)
         │                │
    并发量多大?       进程池(绕GIL)
   ┌─────┴─────┐      任务要"够重"
 几十~几百    上千~上万   否则开销>收益
 线程池        asyncio
(随便塞)    (必须一异步到底)
```

| 任务类型 | 典型例子 | 用什么 | 为什么 |
|---------|---------|--------|--------|
| I/O 密集（等） | 爬虫、调 API、读写文件、查库 | **线程池** | 等待时释放 GIL，线程便宜 |
| I/O + 海量并发 | 上万个连接 | **asyncio** | 线程太多扛不住，协程开销极小 |
| CPU 密集（算） | 数值计算、图像、加解密 | **进程池** | 只有多进程能绕开 GIL |
| 任务很小很碎 | 算一堆小数字 | **串行就好** | 并行开销比任务本身还大 |

---

## GIL：理解这一切的核心

- **线程**共享同一解释器 → 被 GIL 锁住 → **纯计算**没法真并行。
- 但线程在**等待**（sleep / 网络 / 磁盘）时会**释放 GIL** → 所以 I/O 任务多线程很有效。
- **进程**各有独立解释器和内存 → 没有共享 GIL → **CPU 计算能真并行**，吃满多核。

实测（`parallel_03_process.py`，4 个累加任务，10 核机器）：

| 跑法 | 耗时 | 解读 |
|------|------|------|
| 串行 | 1.88s | 一个个跑 |
| 线程池 | 1.87s | **和串行一样** → GIL 让纯计算多线程无效 |
| 进程池 | 0.57s | 真并行，吃满多核 |

---

## 1. 线程池 `ThreadPoolExecutor`（I/O 密集首选）

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(download, urls))   # 结果按输入顺序返回
```

- `with` 自动关闭线程池。
- `max_workers` = 最多同时几个在跑；I/O 任务可设得比核心数多（反正在等）。

## 2. `submit` + `as_completed`（要"先完成先处理" / 单独处理异常）

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=6) as pool:
    fut_to_url = {pool.submit(download, url): url for url in urls}  # dict 反查
    for fut in as_completed(fut_to_url):        # 谁先做完谁先冒泡
        url = fut_to_url[fut]
        try:
            print("✅", fut.result())            # 任务里的异常在这里重新抛出
        except Exception as e:
            print("❌", url, e)                  # 单任务异常被隔离，不搞崩全局
```

- `submit` 立刻返回 **Future**（"取餐小票"），不阻塞。
- `as_completed` 按**完成时间**顺序 yield，与提交顺序无关。
- **`map` vs `submit`**：整齐、单参数、要顺序 → `map`；要先完成先处理 / 复杂参数 / 单独处理异常 → `submit`。
- ⚠️ `future.result(timeout=...)` 只在对"还没完成"的 future 调用时才有意义；经过 `as_completed` 拿到的 future 早已完成，timeout 设多少都无所谓。

## 3. 进程池 `ProcessPoolExecutor`（CPU 密集）

```python
from concurrent.futures import ProcessPoolExecutor

def run_processes():
    with ProcessPoolExecutor(max_workers=4) as pool:   # 几乎只改类名
        return list(pool.map(cpu_heavy, TASKS))

if __name__ == "__main__":   # ★ 必须！否则子进程重新 import → 无限递归创建进程
    run_processes()
```

进程的三个代价（务必记住）：
1. **启动慢、开销大**：任务太小则"开销 > 收益"，反而比串行慢。只用在"每个任务都够重"时。
2. **参数/返回值必须能 `pickle`**：不能传 lambda、局部函数、文件/socket/连接；传大对象复制很慢。
3. **必须有 `if __name__ == "__main__":` 保护**（macOS/Windows）。新手最常踩的坑。
4. CPU 任务 `max_workers` 设成 `os.cpu_count()` 左右就到顶，再多只增加切换开销。

## 4. asyncio（单线程事件循环，海量 I/O 并发）

```python
import asyncio

async def download(url):          # async def 定义协程，调用它不执行，只返回协程对象
    await asyncio.sleep(1)        # await = 非阻塞让出点，挂起自己把 CPU 交回事件循环
    return f"{url} 内容"

async def main():
    results = await asyncio.gather(*(download(u) for u in urls))  # 并发跑，结果按序返回

asyncio.run(main())              # 启动事件循环
```

概念对应：

| reactor 概念 | Python |
|-------------|--------|
| 事件循环 | `asyncio.run()` 启动的 loop |
| 可挂起任务 | `async def` 协程 |
| 非阻塞让出点 | `await` |
| 并发注册 | `asyncio.create_task` / `gather` |

### ⚠️ 致命铁律：一异步到底，不能有任何阻塞调用

只有一个线程 → **一个同步阻塞调用就卡死整个事件循环**，并发退化成串行。
（实测：`parallel_05b_blocking_trap.py` 把 `asyncio.sleep` 换成 `time.sleep`，1s → 5s）

- 用 `aiohttp` 而非 `requests`
- 用 `asyncpg` / `aiomysql` 而非同步数据库驱动
- 用 `await asyncio.sleep` 而非 `time.sleep`

逃生舱（必须调同步阻塞函数时）：把它丢进线程池，不卡事件循环。

```python
result = await asyncio.to_thread(requests.get, url)
```

---

## 通用原则

- **先测再优化**：用 `time.perf_counter()` 实测瓶颈在哪，别凭感觉。很多时候串行就够快，并行反而引入 bug。
- **怎么拆任务往往比"线程还是进程"更难**：比如 10GB 文件得先想好按块切分再分发。
- asyncio 比线程池**更难落地**（一异步到底的门槛），但换来更高的并发上限。

## 配套文件

| 文件 | 内容 |
|------|------|
| `parallel_01_threadpool.py` | 线程池 + `map`，串行 vs 并行对比 |
| `parallel_02_submit.py` | `submit` + `as_completed` + 异常隔离 |
| `parallel_03_process.py` | 进程池，GIL 实证（线程没用、进程有用）|
| `parallel_05_asyncio.py` | asyncio 基础 |
| `parallel_05b_blocking_trap.py` | asyncio 阻塞陷阱演示 |
