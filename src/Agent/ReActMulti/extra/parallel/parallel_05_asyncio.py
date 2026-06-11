"""
第 5 步：asyncio —— 单线程 + 事件循环实现 IO 并发

和第 1 步对照：同样"下载 5 个网页，每个等 1 秒"。
线程池用 5 个线程；asyncio 用 1 个线程，靠事件循环在 await 处切换。

概念对应：
  async def       -> 定义协程（调用它不执行，只返回协程对象）
  await           -> 非阻塞让出点：挂起自己，把 CPU 交回事件循环
  asyncio.gather  -> 把多个协程一起丢进 loop 并发跑
  asyncio.run     -> 启动事件循环
"""

import asyncio
import time


async def download(url):
    print(f"  开始下载 {url}")
    # ★关键：必须用 asyncio.sleep，不能用 time.sleep！
    #   asyncio.sleep 会"让出"事件循环（非阻塞等待）；
    #   time.sleep 会卡死整个线程 = 卡死整个事件循环 = 退化成串行。
    await asyncio.sleep(1)          # 模拟"等网络"，这 1 秒让给别的协程
    print(f"  下载完成 {url}")
    return f"{url} 的内容"


async def main():
    urls = [f"http://site-{i}.com" for i in range(5)]

    t0 = time.perf_counter()
    # gather: 把 5 个协程同时注册到事件循环，并发等待
    results = await asyncio.gather(*(download(u) for u in urls))
    print(f"\n并发耗时: {time.perf_counter() - t0:.2f}s")
    print("结果按输入顺序返回:", results[:2], "...")


if __name__ == "__main__":
    asyncio.run(main())   # 唯一的入口：启动事件循环，跑到 main 结束
