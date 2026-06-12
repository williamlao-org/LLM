### **从 epoll 到 async/await：用户态协程调度器是如何把"事件循环 + 状态机"包装成同步代码外观的**

你既然手写过 epoll，那对内核侧的事件就绪通知、水平触发/边沿触发、`epoll_wait` 的阻塞语义已经熟悉。剩下要打通的其实就一件事：**上层语言（Python、JavaScript、Rust、C# 等）是如何把"回调地狱"重新折叠回一段段看起来像同步代码的 `async fn` 的**。这个过程不是魔法，本质是三件东西的组合——**操作系统的就绪事件通知机制（epoll/kqueue/IOCP）+ 用户态的事件循环（Reactor/Proactor）+ 编译器把函数切成状态机（Coroutine/Continuation）**。下面我从历史一路推到现代实现，把每一层为什么长成现在这样讲清楚。

---

### **一、起点：阻塞 I/O 与"一连接一线程"模型**

在最早的 Unix 网络编程里，最自然的写法就是一个 socket 配一个进程或线程：

```c
while ((cfd = accept(sfd, ...)) >= 0) {
    if (fork() == 0) { handle(cfd); exit(0); }
}
```

Apache 的 prefork、早期的 FTP/Telnet 服务器都是这种结构。它的优点是逻辑直白，调用 `read` 就阻塞等数据，回来就接着写，控制流和数据流完全线性。问题也直白：**一个连接占一个内核调度实体**。线程切换需要陷入内核、保存恢复寄存器、刷 TLB；每个线程默认 8MB 栈（Linux glibc），一万个连接光栈就要 80GB 虚拟地址空间。这就是 1999 年 Dan Kegel 那篇著名的 [The C10K problem](http://www.kegel.com/c10k.html) 提出的核心矛盾：硬件能扛 1 万并发，软件模型扛不住。

为了解决这个问题，业界从两个方向同时演化。一个方向是把线程变轻（后来催生了 Go 的 goroutine、Java 的虚拟线程 Project Loom），另一个方向是**让一个线程同时管很多个 fd**——也就是 I/O 多路复用，最终演化出 async/await。我们走第二条路。

---

### **二、I/O 多路复用的演化：select → poll → epoll/kqueue/IOCP**

`select` 是 4.2BSD（1983）引入的，接口简单但有三个硬伤：fd_set 是位图、上限 FD_SETSIZE（通常 1024）、每次调用都要把整张位图从用户态拷到内核态、内核还要 O(n) 线性扫描。`poll`（System V）把位图换成数组解决了上限，但 O(n) 扫描和拷贝问题没动。

真正的转折点是 2002 年 Linux 2.5.44 引入的 `epoll`，以及 BSD 的 `kqueue`、Windows 的 IOCP。它们的关键改进是：

- **状态保留在内核**：`epoll_ctl(EPOLL_CTL_ADD)` 一次性把 fd 注册进内核的红黑树，之后调用 `epoll_wait` 不再传整个集合。
- **就绪列表 O(1) 取出**：内核在 fd 状态变化时（通过回调 `ep_poll_callback`）直接把它挂到就绪链表，`epoll_wait` 只复制就绪那部分。
- **触发模式**：LT（水平触发）兼容 poll 语义，ET（边沿触发）只在状态变化时通知一次，配合非阻塞 fd 可以最大化吞吐。

到这一步，内核已经能高效告诉你"哪些 fd 现在可以不阻塞地读/写了"。**但请注意一个语义差异**：epoll/kqueue 是 **Readiness Notification（就绪通知）**——内核告诉你"现在去读不会阻塞"，数据搬运还是你自己用 `read` 做；而 Windows IOCP 是 **Completion Notification（完成通知）**——你提交一个"读 4KB 到这个 buffer"的请求，内核帮你把数据搬好之后再通知你。Linux 在 2019 年合入的 [io_uring](https://kernel.dk/io_uring.pdf) 才补上了 Proactor 模型。这两种语义会直接影响上层 async 运行时的设计选择，后面会再提。

---

### **三、Reactor 模式：把内核事件翻译成用户态回调**

光有 epoll 还不够。你得在用户态写一个循环：

```c
while (running) {
    int n = epoll_wait(epfd, events, MAX, timeout);
    for (int i = 0; i < n; i++) {
        Handler* h = events[i].data.ptr;
        if (events[i].events & EPOLLIN)  h->on_read();
        if (events[i].events & EPOLLOUT) h->on_write();
    }
    run_expired_timers();
    run_pending_tasks();
}
```

这就是 1995 年 Douglas Schmidt 在 ACE 框架论文里总结的 **Reactor 模式**：一个事件分发器（Demultiplexer，对应 `epoll_wait`）+ 一组事件处理器（Handler）。它的精髓是**控制反转**——不再是你的代码主动调 `read`，而是 Reactor 在事件就绪时回调你的 handler。

libevent（2002）、libev（2007）、libuv（2011，Node.js 用的那个）都是这一层的代表。Node.js 的"单线程事件循环"指的就是 libuv 这一层；Python 的 `asyncio` 默认 selector_events、Rust 的 `mio`、Java 的 NIO Selector 也都在这一层。

到此为止，性能问题解决了，但**编程模型变得反人类**。你想表达"读完请求头，再读 body，再写响应，再关连接"这种线性逻辑，必须拆成四个 handler，状态用结构体在 handler 之间传：

```c
struct conn { int state; char* buf; size_t off; ... };
void on_read(conn* c) {
    switch (c->state) {
        case READ_HEADER: ...; c->state = READ_BODY; break;
        case READ_BODY:   ...; c->state = WRITE_RESP; epoll_mod(EPOLLOUT); break;
    }
}
```

这就是 Node.js 早期臭名昭著的 **callback hell**：业务逻辑被切成碎片，错误处理散落各处，控制流在源代码里不可见。所有后续的语言级努力，本质都是在回答同一个问题——**怎么让程序员写线性代码，运行时却以 Reactor 方式执行**。

---

### **四、第一次折叠：Promise/Future——把回调变成值**

第一步是把"一个未来才会有的值"抽象成头等对象。这个想法可以追溯到 1977 年 Henry Baker 和 Carl Hewitt 的 future，1985 年 Multilisp 的 promise，但在工业界普及是因为 2009 年 [CommonJS Promises/A](http://wiki.commonjs.org/wiki/Promises/A) 和后来的 Promises/A+，最终 ES6（2015）把 `Promise` 标准化。

一个 Promise 大致长这样（伪代码）：

```js
class Promise {
  state = 'pending';        // pending | fulfilled | rejected
  value = undefined;
  callbacks = [];
  resolve(v) { this.state='fulfilled'; this.value=v; this.callbacks.forEach(cb=>cb(v)); }
  then(cb) { 
    if (this.state==='fulfilled') queueMicrotask(()=>cb(this.value));
    else this.callbacks.push(cb);
    return new Promise(...); // 链式
  }
}
```

它没有解决"控制流碎片化"的问题——`.then().then().then()` 仍然是回调，只是变成了链式。但它做对了两件关键的事：**第一，它让"未完成的计算"成为可以被 return、被组合（`Promise.all`）的值**；**第二，它统一了成功/失败的传播路径**，让错误能像同步代码一样沿链冒泡到 `.catch`。这两件事是 async/await 能存在的前提：因为 `await` 必须有一个东西可以"等"。

在 Python 里这个对象叫 `Future`（`asyncio.Future`），在 Rust 里叫 `Future` trait，在 C# 里叫 `Task<T>`，在 Java 里叫 `CompletableFuture`。名字不同，本质都是"一个被 Reactor 持有、状态完成时会触发回调的占位符"。

---

### **五、第二次折叠：协程——把函数切成可暂停的状态机**

光有 Promise 还不够，要让代码看起来同步，必须能让一个函数"在中间停下来，等 Promise 完成后再从同一个地方继续执行"。这就是**协程（coroutine）**。

协程的概念由 Melvin Conway 在 1958 年提出，比线程还早。它的本质是**协作式多任务**：一个执行体可以主动 `yield` 出 CPU，并保留所有局部状态，下次被唤醒时从 yield 处继续。实现协程在工程上有两条路：

#### **5.1 有栈协程（stackful）**

每个协程拥有独立的栈（通常 4KB\~64KB，按需增长）。切换时把当前 CPU 寄存器（rip、rsp、rbx、r12-r15 等被调用者保存寄存器）存到协程控制块，把目标协程的寄存器恢复出来。Go 的 goroutine、Lua coroutine、boost.context、libco（微信开源）都属于这一类。

切换的核心代码非常短，x86-64 大概十几条 mov：

```asm
; save current ctx
mov [rdi+0], rsp
mov [rdi+8], rbp
mov [rdi+16], rbx
...
; load next ctx
mov rsp, [rsi+0]
mov rbp, [rsi+8]
...
ret
```

优点是**任何普通函数都能在里面 yield**（包括第三方库、递归、深层调用栈），用户代码完全无感。缺点是每个协程都要预留栈空间，且与 C ABI 强耦合，跨语言、跨编译器不友好。Go 的解决办法是栈可增长（最初 segmented stack，后来改成 copy stack），起步只要 2KB，所以一台机器跑百万 goroutine 是常态。

#### **5.2 无栈协程（stackless）**

不保留独立栈，而是**由编译器把函数体转写成一个状态机**。所有跨 yield 点的局部变量被提升为状态机结构体的字段。Python 的 generator/async、JavaScript 的 async function、C# 的 async、Rust 的 async fn、C++20 的 coroutine 都属于这一类。

举个最直观的例子。你写：

```python
async def fetch_two(url1, url2):
    a = await http_get(url1)
    b = await http_get(url2)
    return a + b
```

编译器（或 Python 的字节码生成器）大致会把它转写成等价于：

```python
class FetchTwoCoro:
    def __init__(self, url1, url2):
        self.state = 0
        self.url1, self.url2 = url1, url2
        self.a = None
    def send(self, value):
        if self.state == 0:
            self.fut = http_get(self.url1)
            self.state = 1
            return self.fut          # 把 Future 抛给调度器
        if self.state == 1:
            self.a = value           # 调度器把结果送进来
            self.fut = http_get(self.url2)
            self.state = 2
            return self.fut
        if self.state == 2:
            b = value
            raise StopIteration(self.a + b)
```

注意几个关键点：

1. **每个 `await` 都对应一个状态切分点**。函数体在源代码层面是顺序的，但运行时被切成 N+1 段，N 是 await 数量。
2. **跨 await 存活的局部变量（这里的 `self.a`）被提升到对象字段**。栈帧在 yield 时被销毁，恢复时不需要重建栈，只需要重新进入 `send` 跳到对应 state。
3. **`send(value)` 是恢复点**：调度器拿到上一个 Future 的结果后调 `send(result)`，协程从上次 yield 处取到这个值继续执行。
4. **协程本身不调度自己**——它只是 yield 出"我在等这个 Future"，谁来 poll 它、什么时候 poll，是事件循环的事。

这就是 async/await 的全部秘密：**编译器做 CPS 变换（Continuation-Passing Style transformation），把"等到 X 完成后做 Y"这种延续显式化成状态机**。无栈协程的好处是几乎零内存开销（只有状态机结构体那么大，几十字节）、不依赖汇编、可以跨平台；代价是**只有标记为 async 的函数才能 await**，普通函数不能在中间挂起——这就是 Bob Nystrom 那篇 [What Color is Your Function?](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/) 抱怨的"函数染色"问题。

---

### **六、把三层粘起来：事件循环如何驱动协程**

现在三块零件齐了：epoll（内核就绪通知）、Reactor（用户态事件分发）、协程（可暂停的函数）。它们怎么协作？我用 Python `asyncio` 做最具体的例子，因为它的实现最容易直接读源码。

#### **6.1 一次完整的 await 旅程**

考虑 `result = await reader.read(4096)`：

1. **协程执行到 await**。`reader.read` 是个 async 函数，它内部最终会调到 `loop.sock_recv(sock, 4096)`，这个方法做的事是：
   - 创建一个 `Future` 对象 `fut`。
   - 调 `loop.add_reader(sock.fileno(), self._sock_recv_cb, fut, sock, 4096)`，这一步对应 `epoll_ctl(EPOLL_CTL_ADD, sock, EPOLLIN)`，并在事件循环的 fd → callback 字典里登记一条。
   - 返回 `fut`。
2. **协程 yield 这个 Future**。状态机停在当前 state，控制权回到调用 `coro.send()` 的人——也就是事件循环。
3. **事件循环把 Future 和协程关联起来**。具体做法：协程被包在一个 `Task` 对象里，`Task.__step` 调用 `coro.send(None)` 拿到 yield 出来的 Future，然后调 `fut.add_done_callback(self.__wakeup)`。意思是"这个 Future 完成时，请回来调 `Task.__wakeup` 把我重新塞进就绪队列"。
4. **事件循环回到主循环**，调 `selector.select(timeout)`（底层就是 `epoll_wait`），阻塞等待任何一个注册过的 fd 就绪。
5. **数据到达，内核唤醒 `epoll_wait`**。事件循环拿到就绪 fd 列表，查字典找到 `_sock_recv_cb`，调用它。这个回调里执行 `data = sock.recv(4096); fut.set_result(data); loop.remove_reader(sock.fileno())`。
6. **`fut.set_result` 触发 done_callback**，把 `Task.__wakeup` 扔进事件循环的 ready 队列（一个 `collections.deque`）。
7. **事件循环下一轮**先排干 ready 队列：调 `Task.__wakeup`，它再调 `coro.send(data)`，协程从原 await 处恢复，`result` 拿到 data，继续往下执行……直到下一个 await 或函数返回。

整个过程没有线程切换、没有锁、没有内核态用户态来回切（除了 epoll_wait 那一次必要的阻塞）。一个线程靠状态机的快速切换，把上千个连接的 I/O 等待时间填满。

#### **6.2 事件循环主循环骨架**

抽象出来其实非常简单：

```python
def run_forever(self):
    while not self._stopping:
        # 1. 计算最近一个定时器到期的时间
        timeout = self._compute_timeout()
        # 2. 阻塞等待 I/O 或定时器
        events = self._selector.select(timeout)
        # 3. 处理 I/O 就绪
        for key, mask in events:
            self._add_callback(key.data, mask)
        # 4. 处理到期定时器
        self._process_scheduled()
        # 5. 排干 ready 队列（执行所有就绪的回调，包括 Task.__wakeup）
        ntodo = len(self._ready)
        for _ in range(ntodo):
            handle = self._ready.popleft()
            handle._run()
```

这个结构在 Node.js 的 libuv（多了 timers/pending/idle/poll/check/close 几个 phase）、Tokio 的 `current_thread` runtime、C# 的 `SynchronizationContext` 里几乎一模一样。差异主要在：单线程还是多线程窃取、Reactor 还是 Proactor、定时器用最小堆还是时间轮。

---

### **七、不同语言的具体落地**

虽然原理一致，各语言因为运行时哲学不同，做出的取舍也不同。我挑四个有代表性的对比一下。

#### **7.1 JavaScript / Node.js**

ES6 引入 generator（`function*` + `yield`）后，社区先用 co 库手动写"自动驱动 generator"的逻辑：拿到 yield 出的 Promise，等它 resolve，再 `gen.next(value)`。ES2017 直接把这个模式语法糖化成 `async/await`，所以 `async function` 本质就是"返回 Promise 的 generator + 自动驱动器"。

事件循环是 [libuv](https://libuv.org/) 提供的，分 6 个 phase：timers、pending callbacks、idle/prepare、poll（这一步是 epoll_wait）、check（`setImmediate`）、close。Promise 的 `.then` 回调走 microtask 队列，每个宏任务结束后排干一次 microtask，这就是为什么 `Promise.resolve().then()` 比 `setTimeout(fn, 0)` 先执行。

#### **7.2 Python / asyncio**

历史最曲折。2001 年 PEP 255 引入 generator，2009 年 PEP 342 让 generator 能 `yield` 后再被 `send` 值进去（这是协程的雏形），2014 年 PEP 3156 引入 `asyncio` 和 `yield from`，2015 年 PEP 492 在 Python 3.5 加入 `async def` / `await` 关键字，正式独立于 generator。

asyncio 默认是 Reactor 模型，Selector 用 epoll/kqueue/IOCP 之一。它有个微妙的设计选择：协程对象本身被 `Task` 包裹后才参与调度，`Task` 是 `Future` 的子类，这就让"任务"和"等待结果"统一了类型，Task 也能被 await。社区另有 [uvloop](https://github.com/MagicStack/uvloop)（基于 libuv 的 C 实现）和 [Trio](https://trio.readthedocs.io/) （主推结构化并发）作为替代。

#### **7.3 Rust**

Rust 的 async 设计是这套体系里最"工程派"的，因为它要满足两个硬约束：零成本抽象、无 GC。

`async fn` 被编译器展开成一个实现 `Future` trait 的匿名状态机：

```rust
trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}
enum Poll<T> { Ready(T), Pending }
```

注意它和 Python/JS 的根本差异：**Rust 的 Future 是惰性的**，你不 `.await` 它就什么都不做；而且 Future 不持有自己的栈或回调，**它只在被 poll 时往前推进一步**。如果还没好，它返回 `Pending`，并通过 `Context` 拿到的 `Waker` 注册"好了之后请叫我"，然后把控制权交还。

谁来 poll？运行时（[Tokio](https://tokio.rs/)、async-std、smol）。Tokio 的 worker 线程从任务队列拿 Task（包着顶层 Future），调 `task.poll(cx)`。Future 内部如果遇到 I/O，会向 [mio](https://github.com/tokio-rs/mio)（Rust 版的 libevent）注册 fd 和 Waker，然后返回 Pending。Tokio 的 reactor 线程跑 `epoll_wait`，事件就绪时调对应 Waker.wake()，把 Task 重新塞回执行队列。

Rust 的状态机里有自引用问题（一个变量指向同结构体里另一个变量的栈地址），这就是 [`Pin`](https://doc.rust-lang.org/std/pin/index.html) 存在的原因——保证状态机一旦开始 poll 就不能被移动到内存其他位置。这是为了用编译期类型系统替代 GC 而付出的代价。

#### **7.4 Go**

Go 是反例：它**没有 async/await**。goroutine 是有栈协程，调度器（GMP 模型）在用户态做抢占式调度，I/O 操作在 runtime 层被自动拦截——比如你调 `conn.Read`，runtime 实际上把 fd 设为非阻塞、注册到 netpoll（Linux 上就是 epoll）、然后把当前 goroutine park 住、调度器换上别的 goroutine 跑。fd 就绪时 netpoll 唤醒对应 goroutine。

所以 Go 程序员写 `n, err := conn.Read(buf)` 看起来是阻塞同步的，运行时却是异步的。Go 选择了"语言层面看不见 async"的路线：代价是必须有重型 runtime（GC + 调度器 + 栈管理），不能像 Rust 那样嵌入受限环境。这条路线最近被 Java 的 [Project Loom](https://openjdk.org/projects/loom/)（虚拟线程，JDK 21 正式发布）复刻。

---

### **八、几个容易混淆的细节**

**关于"非阻塞 I/O"和"异步 I/O"。** 很多文档把这两个混用，其实严格区分：非阻塞 I/O 指 `read` 没数据立刻返回 `EAGAIN`，搬数据还是同步的；异步 I/O 指你提交请求，内核搬完数据通知你（POSIX AIO、Windows IOCP、io_uring）。Linux 的 async 生态长期以来是"非阻塞 + epoll"模拟出来的，io_uring 是真正的异步 I/O，所以 Tokio、libuv 都在加 io_uring 后端。

**关于 microtask 和 macrotask。** JS 的 Promise 回调走 microtask 队列，每个 macrotask（一个 I/O callback、一个 setTimeout 回调）执行完就排干一次 microtask。这就是为什么 `await` 后续的代码总是在同一个事件循环 tick 里继续，而不会被其他 I/O 插队——这保证了 await 的"看起来同步"的语义。

**关于 CPU 密集任务。** async/await 模型有个根本前提：**协程必须主动 yield**。如果你在某个 async 函数里跑一个 10 秒的纯计算 for 循环，整个事件循环就被卡 10 秒，所有其他连接都饿死。这不是 bug，是模型本身的特性。解决办法是把 CPU 任务扔到线程池（`loop.run_in_executor`、`tokio::task::spawn_blocking`），让事件循环线程只做 I/O 调度。Go 的抢占式调度可以避免这个问题，async/await 不行。

**关于结构化并发。** 早期的 `asyncio.create_task`、JS 的 `Promise.all` 有个共同问题：父任务取消后，子任务可能成孤儿继续跑，资源泄漏。Trio 提出的 nursery、Kotlin 的 coroutineScope、Python 3.11 的 [`TaskGroup`](https://docs.python.org/3/library/asyncio-task.html#task-groups)、Java Loom 的 StructuredTaskScope 都是同一个思想：**子任务的生命周期不能超出父任务的作用域**，作用域退出时所有未完成子任务必须被取消并 join。这是 async 生态最近五年的主要进步方向。

---

### **九、把整张图压缩成一句话**

如果要用一句话概括 async/await 的实现：

> **编译器把 async 函数转写成一个以 await 为切分点的状态机对象，事件循环用 epoll 之类的多路复用器监听内核就绪事件，事件就绪时通过 Waker/Future/Promise 这层间接对象把对应的状态机推进一步，于是程序员看到的线性代码，运行起来其实是无数个状态机轮流在一个线程里前进。**

你之前手写 epoll 时面对的是"一堆 fd + 一个 epoll_wait + 自己维护状态结构体"。语言层 async/await 做的事，是把"自己维护状态结构体"这件事自动化了——通过编译器的 CPS 变换，让状态机的字段就是源代码里的局部变量，状态机的状态就是源代码里 await 的位置。Reactor、事件循环、Future/Promise 这些中间层，都只是为了把"内核的就绪事件"翻译成"状态机的下一次 poll"而存在的脚手架。

理解了这一层，你再去读 [CPython 的 `Lib/asyncio/`](https://github.com/python/cpython/tree/main/Lib/asyncio)、[Tokio 的 runtime crate](https://github.com/tokio-rs/tokio/tree/master/tokio/src/runtime)、[Node 的 libuv](https://github.com/libuv/libuv/tree/v1.x/src/unix) 源码，你会发现它们都在解决同一组问题：怎么把 fd 注册进 epoll、怎么把就绪事件翻译成 Waker.wake、怎么调度被唤醒的任务、怎么跟定时器/线程池/信号配合。差异只在工程取舍——单线程 vs 多线程窃取、有栈 vs 无栈、惰性 poll vs 主动 push、Reactor vs Proactor。把这些坐标轴在脑子里建立起来，看任何一门语言的 async 实现都不会再有黑盒。