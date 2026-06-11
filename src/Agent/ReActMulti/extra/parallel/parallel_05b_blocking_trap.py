"""
第 5 步(b)：asyncio 的致命坑 —— 在协程里调用阻塞函数

同样 5 个任务，唯一区别：把 await asyncio.sleep(1) 换成同步的 time.sleep(1)。
结果：事件循环被卡死，并发退化成串行，耗时从 1s 变 5s。
"""

import asyncio
import time


async def bad_download(url):
    print(f"  开始 {url}")
    time.sleep(1)          # ❌ 同步阻塞！整个事件循环卡在这 1 秒，谁也跑不了
    print(f"  完成 {url}")


async def main():
    urls = [f"http://site-{i}.com" for i in range(5)]
    t0 = time.perf_counter()
    await asyncio.gather(*(bad_download(u) for u in urls))
    print(f"\n耗时: {time.perf_counter() - t0:.2f}s  (退化成串行了！)")


if __name__ == "__main__":
    asyncio.run(main())
