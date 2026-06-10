"""
第 2 步：submit + as_completed —— 比 map 更灵活

核心概念：
- pool.submit(fn, *args) 立刻返回一个 Future（"取餐小票"），不阻塞。
- Future 代表"一个还没算完的结果"，可以查状态 / 取结果 / 拿异常。
- as_completed(futures) 是一个迭代器：谁先算完就先 yield 谁，不按提交顺序。

对比第 1 步的 map：
- map 必须传一个可迭代参数、结果按顺序；
- submit 可以传任意参数，还能让"先完成的先处理"。
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


def download(url, retry):
    # 故意让每个任务耗时不同，这样才能看出"谁先完成谁先返回"
    cost = random.uniform(0.5, 2.0)
    time.sleep(cost)
    if "site-3" in url:
        raise ValueError(f"{url} 挂了！")  # 故意让一个任务报错
    return f"{url} 内容(retry={retry}, 耗时{cost:.2f}s)"


urls = [f"http://site-{i}.com" for i in range(6)]


def main():
    with ThreadPoolExecutor(max_workers=6) as pool:
        # 1) 提交任务，拿到一堆 Future。
        #    用 dict 把 future 映射回它对应的 url，方便出错时知道是谁。
        future_to_url = {pool.submit(download, url, retry=3): url for url in urls}

        # 2) as_completed：谁先跑完就先进入循环，顺序由"完成时间"决定
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result(0.1)  # 取结果；若任务里抛了异常，这里会重新抛出
                print(f"✅ {result}")
            except Exception as e:
                # 单个任务的异常被隔离在这里，不会搞崩整个程序
                print(f"❌ {url} 失败: {e}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n总耗时: {time.perf_counter() - t0:.2f}s")
