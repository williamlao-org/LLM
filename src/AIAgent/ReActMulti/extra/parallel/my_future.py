"""
手写一个最简版 Future + 回调机制，还原 CPython 内部的实现原理
"""

import threading
import time


class SimpleFuture:
    def __init__(self):
        self._result = None
        self._done = False
        self._callbacks = []          # 存回调函数的列表
        self._condition = threading.Condition()  # 底层同步原语

    def set_result(self, value):
        """工作线程调用：任务完成，设置结果"""
        with self._condition:
            self._result = value
            self._done = True
            self._condition.notify_all()  # 唤醒所有在 result() 里等待的线程

        # 在工作线程自己的调用栈上，顺序执行所有回调
        for cb in self._callbacks:
            cb(self)

    def result(self):
        """消费方调用：拿结果，没好就阻塞等"""
        with self._condition:
            while not self._done:
                self._condition.wait()  # 释放锁，挂起线程，等 notify
            return self._result

    def add_done_callback(self, fn):
        """注册回调：把函数引用存进列表，仅此而已"""
        self._callbacks.append(fn)


def simple_as_completed(futures):
    """
    还原 as_completed 的核心逻辑：
    谁先完成就先 yield 谁
    """
    queue = []                        # 完成的 future 放这里
    lock = threading.Lock()
    ready = threading.Condition(lock) # 用来挂起/唤醒 as_completed 的线程

    def on_done(future):
        # 这个函数在工作线程的调用栈上执行（不是 as_completed 的线程）
        with ready:
            queue.append(future)
            ready.notify()            # 唤醒下面的 wait()

    for f in futures:
        f.add_done_callback(on_done)  # 每个 future 注册同一个回调

    received = 0
    while received < len(futures):
        with ready:
            while not queue:
                ready.wait()          # 队列空就挂起，等 notify
            future = queue.pop(0)     # 拿到第一个完成的
        received += 1
        yield future                  # 谁先完成先 yield 谁


def worker(future, url):
    """模拟工作线程：干完活，自己调 set_result"""
    time.sleep(0.5)
    print(f"  [工作线程] {url} 完成，调 set_result")
    future.set_result(f"{url} 的内容")
    # set_result 内部：
    #   1. 存结果
    #   2. notify_all 唤醒 result() 的等待者
    #   3. 遍历 _callbacks，挨个调用  ← 回调就在这里，普通 for 循环


if __name__ == "__main__":
    urls = ["http://site-1.com", "http://site-2.com", "http://site-3.com"]

    futures = [SimpleFuture() for _ in urls]

    # 启动工作线程
    for future, url in zip(futures, urls):
        t = threading.Thread(target=worker, args=(future, url))
        t.start()

    # as_completed：谁先完成先拿到
    for f in simple_as_completed(futures):
        print(f"[主线程] 拿到结果: {f.result()}")
