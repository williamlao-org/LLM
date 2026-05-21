// ============================================================
// 第三步：Generator + Promise = 穷人版 async/await
// ============================================================
//
// 核心想法：
//   1. 用 generator 写"看起来像同步"的代码
//   2. 每次 yield 出一个 Promise
//   3. 外面有个"驱动器"，等 Promise 完成后，把结果用 next() 塞回去
//
// 这就是 async/await 被发明之前，人们实际用的方案（co 库，2013年）

// ---- 先准备两个返回 Promise 的异步函数 ----

function fetchUser(id) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({ id, name: "小明" });
        }, 500);
    });
}

function fetchOrders(userId) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve([
                { id: 101, item: "键盘" },
                { id: 102, item: "鼠标" },
            ]);
        }, 500);
    });
}

// ---- 用 generator 写异步逻辑 ----
// 注意看：这段代码长得几乎和 async/await 一模一样
// 唯一的区别是 async → function*，await → yield

function* main() {
    console.log("开始");

    const user = yield fetchUser(1);
    //                ↑ yield 出一个 Promise
    //    ↑ 等驱动器把 Promise 的结果塞回来，user 就拿到了值

    console.log("拿到用户:", user);

    const orders = yield fetchOrders(user.id);
    console.log("拿到订单:", orders);

    return `${user.name} 有 ${orders.length} 个订单`;
}

// ---- 关键：驱动器（runner） ----
// 这就是让 generator 自动跑起来的那个"外部力量"
// async/await 的运行时本质上就是这个东西

function run(generatorFn) {
    const it = generatorFn();  // 创建迭代器

    function step(nextValue) {
        const result = it.next(nextValue);
        // result = { value: Promise对象, done: false/true }

        if (result.done) {
            // generator 跑完了，result.value 是 return 的值
            console.log("最终结果:", result.value);
            return;
        }

        // result.value 是一个 Promise
        // 等它完成，把结果塞回 generator
        const promise = result.value;
        promise.then((resolvedValue) => {
            // Promise 完成了！把结果通过 next() 塞回去
            // 这会让 generator 从 yield 那里"醒来"，
            // 并且 yield 表达式的值就是 resolvedValue
            step(resolvedValue);
        });
    }

    step();  // 启动第一步
}

// ---- 跑起来 ----
run(main);

// 输出：
// 开始
// (500ms后) 拿到用户: { id: 1, name: '小明' }
// (再500ms后) 拿到订单: [ { id: 101, item: '键盘' }, { id: 102, item: '鼠标' } ]
// 最终结果: 小明 有 2 个订单
