"""
第 3 步：进程池 —— 真正绕开 GIL 做 CPU 并行

这次任务是"纯 Python 计算"（一个大循环累加），不是等待。
对照三种跑法：
  1) 串行
  2) 线程池  —— 因为 GIL，纯计算几乎不会变快（甚至更慢）
  3) 进程池  —— 各进程独立解释器，吃满多核，真正变快

关键：线程池→进程池，代码几乎只改一个类名。
"""

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def cpu_heavy(n):
    """纯 Python 计算：累加，没有任何等待。这种任务被 GIL 卡得死死的。"""
    total = 0
    for i in range(n):
        total += i * i
    return total


# 4 个一样的重计算任务
TASKS = [20_0000] * 4


def run_serial():
    return [cpu_heavy(n) for n in TASKS]


def run_threads():
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(cpu_heavy, TASKS))


def run_processes():
    with ProcessPoolExecutor(max_workers=4) as pool:
        return list(pool.map(cpu_heavy, TASKS))


def timeit(name, fn):
    t0 = time.perf_counter()
    fn()
    print(f"{name:12s}: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    # 注意：进程池在 Windows/macOS 上必须放在 __main__ 保护下，
    # 否则子进程会重新 import 本文件、无限递归创建进程。这是硬性要求。
    timeit("串行", run_serial)
    timeit("线程池", run_threads)
    timeit("进程池", run_processes)
