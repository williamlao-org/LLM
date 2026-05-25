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

function fetchUser (id) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve({ id, name: "小明" });
        }, 500);
    });
}

function fetchOrders (userId) {
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

function* main () {
    console.log("开始");

    const user = yield fetchUser(1);
    //                ↑ yield 出一个 Promise
    //    ↑ 等驱动器把 Promise 的结果塞回来，user 就拿到了值

    console.log("拿到用户:", user);

    const orders = yield fetchOrders(user.id);
    console.log("拿到订单:", orders);

    return `${user.name} 有 ${orders.length} 个订单`;
}

// 主动调用
// const it = main();
// const it1 = it.next();
// let it2, it3;
// console.log("第一次 yield:", it1);
// // 输出：第一次 yield: { value: Promise { <pending> }, done: false }
// it1.value.then((user) => {
//     it2 = it.next(user); // 把 user 塞回 generator，继续执行到下一个 yield
//     it2.value.then((orders) => {
//         it3 = it.next(orders); // 把 orders 塞回 generator，继续执行到 return
//     });
// });

// 重复的逻辑包装成迭代函数
// function step (it, nextValue) {
//     const result = it.next(nextValue);
//     if (result.done) {
//         console.log("最终结果:", result.value);
//         return result.value;   // ← 我 return 了！
//     }
//     const promise = result.value;
//     promise.then((resolvedValue) => {
//         step(it, resolvedValue);
//     });
// }


// const answer=step(iter) 同步得到的return值是undefined，采用promise和then来拿到最终结果

// const iter = main()
// const stepPromise = new Promise((resolve) => {
//     function step (it, nextValue) {
//         const result = it.next(nextValue);
//         if (result.done) {
//             resolve(result.value);
//             return;
//         }
//         const promise = result.value;
//         promise.then((resolvedValue) => {
//             step(it, resolvedValue);
//         });
//     }

//     step(iter);
// })

// stepPromise.then((finalResult) => {
//     console.log("最终结果:", finalResult);
// })

// 把main()提取成参数，变成一个通用的 run() 函数，可以传入任何 generator 函数来执行。
// function run (generatorFn) {
//     const it = generatorFn();
//     return new Promise((resolve) => {
//         function step (nextValue) {
//             const result = it.next(nextValue);
//             if (result.done) {
//                 resolve(result.value);
//                 return;
//             }
//             const promise = result.value;
//             promise.then((resolvedValue) => {
//                 step(resolvedValue);
//             });
//         }

//         step();
//     })
// }

// run(main).then((finalResult) => {
//     console.log("最终结果:", finalResult);
// })

// 上面这个 run() 函数已经能处理正常的流程了，但如果 Promise 失败了，整个流程就没法继续了。
function run (generatorFn) {
    const it = generatorFn();
    return new Promise((resolve, reject) => {
        function step (method, value) {
            // 也可以直接 result = it[method](value)，js 里it["next"](value) 就是 it.next(value)
            let result;

            try {
                if (method === "next") {
                    result = it.next(value);
                }
                else if (method === "throw") {
                    result = it.throw(value); 
                }
            } catch (err) { // 如果generator里没有try/catch，执行it.throw()时会抛出错误，这时我们在这里捕获到，就可以reject整个run()返回的Promise了。
                reject(err);
                return;
            }

            if (result.done) {
                resolve(result.value);
                return;
            }

            const promise = result.value;
            promise
                .then((resolvedValue) => {
                    step("next", resolvedValue);
                })
                .catch((err) => {
                    // 如果 Promise 失败了，也要把错误塞回 generator，让它有机会处理。
                    step("throw", err);
                });
        }

        step('next');
    })
}

// // 你写的                              // 原生
// run(function* () {                     async function () {
//     const user = yield fetchUser(1);       const user = await fetchUser(1);
//     return user.name;                      return user.name;
// })                                     }()


run(function* () {
    const user = yield fetchUser(1);
    const orders = yield fetchOrders(user.id);
    return `${user.name} 有 ${orders.length} 个订单`;
}).then((finalResult) => {
    console.log("最终结果:", finalResult);
})

async function test () {
    const user = await fetchUser(1);
    const orders = await fetchOrders(user.id);
    return `${user.name} 有 ${orders.length} 个订单`;
}

test().then((finalResult) => {
    console.log("最终结果:", finalResult);
})