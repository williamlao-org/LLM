"""
第 1 步：线程池，处理 I/O 密集任务

场景：要"下载"5 个网页，每个要等 1 秒网络响应。
我们用 time.sleep(1) 模拟"等网络"这个动作——这期间 CPU 是空闲的。
"""

import time
from concurrent.futures import ThreadPoolExecutor


def download(url) -> str:
    print(f"  开始下载 {url}")
    time.sleep(1)  # 模拟等网络：这 1 秒 CPU 啥也没干，纯等待
    print(f"  下载完成 {url}")
    return f"{url} 的内容"


urls = [f"http://site-{i}.com" for i in range(10)]


# ---------- 串行：一个接一个 ----------
def run_serial():
    results = []
    for url in urls:
        results.append(download(url))
    return results


# ---------- 并行：5 个线程一起等 ----------
def run_parallel():
    # with 语句确保用完自动关闭线程池
    with ThreadPoolExecutor(max_workers=5) as pool:
        # pool.map 和内置 map 用法一样，但会把任务分发到线程里
        results = list(pool.map(download, urls))
    return results


if __name__ == "__main__":
    print("=== 串行 ===")
    t0 = time.perf_counter()
    run_serial()
    print(f"串行耗时: {time.perf_counter() - t0:.2f}s\n")

    print("=== 并行（线程池）===")
    t0 = time.perf_counter()
    run_parallel()
    print(f"并行耗时: {time.perf_counter() - t0:.2f}s")
