// ============================================================
// 第四步：用你的 MyPromise 实现完整的 async/await
// ============================================================

// ---- 你的 MyPromise（原封不动搬过来） ----

class MyPromise {
    constructor(executor) {
        this.state = 'pending';
        this.value = undefined
        this.callbacksFulfilled = []
        this.callbacksRejected = []

        const resolve = (v) => {
            if (this.state === 'pending') {
                this.state = 'fulfilled'
                this.value = v
                this.callbacksFulfilled.forEach(fn => queueMicrotask(() => fn(v)))
            }
        }

        const reject = (e) => {
            if (this.state === 'pending') {
                this.state = 'rejected'
                this.value = e
                this.callbacksRejected.forEach(fn => queueMicrotask(() => fn(e)))
            }
        }
        try {
            executor(resolve, reject);
        } catch (e) {
            reject(e);
        }
    }
    then (fnResolve, fnReject) {
        function handleResult (result, resolve, reject) {
            if (result instanceof MyPromise) {
                result.then(resolve, reject);
            } else {
                resolve(result);
            }
        }
        return new MyPromise((resolve, reject) => {
            if (this.state === 'fulfilled') {
                queueMicrotask(() => {
                    if (fnResolve) {
                        try {
                            const result = fnResolve(this.value)
                            handleResult(result, resolve, reject);
                        } catch (e) {
                            reject(e)
                        }
                    } else {
                        resolve(this.value)
                    }
                })
            }
            else if (this.state === 'rejected') {
                queueMicrotask(() => {
                    if (fnReject) {
                        try {
                            const result = fnReject(this.value)
                            handleResult(result, resolve, reject);
                        } catch (e) {
                            reject(e)
                        }
                    } else {
                        reject(this.value)
                    }
                })
            }
            else { 
                if (fnResolve)
                    this.callbacksFulfilled.push((value) => {
                        try {
                            const result = fnResolve(value)
                            handleResult(result, resolve, reject);
                        } catch (e) {
                            reject(e)
                        }
                    })
                else
                    this.callbacksFulfilled.push((value) => {
                        resolve(value)
                    })

                if (fnReject)
                    this.callbacksRejected.push((err) => {
                        try {
                            const result = fnReject(err)
                            handleResult(result, resolve, reject);
                        } catch (e) {
                            reject(e)
                        }
                    })
                else
                    this.callbacksRejected.push((err) => {
                        reject(err)
                    })
            }
        })
    }
    catch (fnReject) {
        return this.then(undefined, fnReject)
    }
}


// ---- 完整的驱动器：myAsync ----
// 这就是你要自己实现的 async 关键字的本质

function myAsync(generatorFn) {
    // myAsync 返回一个函数
    // 调用这个函数时，返回一个 MyPromise
    // 这就是为什么 async function 的返回值永远是 Promise
    return function(...args) {
        return new MyPromise((resolve, reject) => {
            const it = generatorFn(...args);  // 创建迭代器

            function step(method, value) {
                let result;
                try {
                    result = it[method](value);
                    // method 是 "next" 或 "throw"
                    // it.next(value)  → 把 value 塞给 yield，继续执行
                    // it.throw(error) → 把 error 扔进 yield 那个位置，生成器里能 try/catch 接住
                } catch (e) {
                    // 生成器里有未捕获的异常 → 整个 async 函数 reject
                    reject(e);
                    return;
                }

                if (result.done) {
                    // 生成器跑完了（return 了）
                    // async 函数的返回值 = 生成器的 return 值
                    resolve(result.value);
                    return;
                }

                // result.value 应该是一个 Promise（就像 await 后面跟的东西）
                // 等它完成 → 把结果塞回 generator
                // 等它失败 → 把错误扔进 generator
                const promise = result.value;
                if (promise instanceof MyPromise) {
                    promise.then(
                        (val) => step("next", val),     // 成功：继续跑
                        (err) => step("throw", err)     // 失败：往生成器里扔错误
                    );
                } else {
                    // yield 后面不是 Promise？直接当值塞回去
                    step("next", promise);
                }
            }

            step("next");  // 启动
        });
    };
}


// ============================================================
// 测试！
// ============================================================

// ---- 准备异步函数（用你的 MyPromise） ----

function fetchUser(id) {
    return new MyPromise((resolve) => {
        setTimeout(() => {
            if (id === 1) resolve({ id, name: "小明" });
            else resolve(null);
        }, 300);
    });
}

function fetchOrders(userId) {
    return new MyPromise((resolve, reject) => {
        setTimeout(() => {
            if (userId === 1) {
                resolve([
                    { id: 101, item: "键盘" },
                    { id: 102, item: "鼠标" },
                ]);
            } else {
                reject(new Error("找不到订单"));
            }
        }, 300);
    });
}


// ---- 测试 1：正常流程 ----
// 注意看：function* 就是 async function，yield 就是 await
// myAsync 把 generator 包装成"异步函数"

const getInfo = myAsync(function* (userId) {
    console.log("开始查询...");
    
    const user = yield fetchUser(userId);       // ← 这就是 await fetchUser(userId)
    console.log("拿到用户:", user);

    const orders = yield fetchOrders(user.id);  // ← 这就是 await fetchOrders(user.id)
    console.log("拿到订单:", orders);

    return `${user.name} 有 ${orders.length} 个订单`;
});

// getInfo 就像一个 async function，调用它返回 MyPromise
getInfo(1).then(result => {
    console.log("✅ 最终结果:", result);
});


// ---- 测试 2：错误处理 ----
// generator 里的 try/catch 能接住 yield 出去的 Promise 的 reject！

const getInfoSafe = myAsync(function* (userId) {
    try {
        const user = yield fetchUser(userId);
        if (!user) throw new Error("用户不存在");

        const orders = yield fetchOrders(999);  // 故意传错的 id
        return `${user.name} 有 ${orders.length} 个订单`;
    } catch (e) {
        console.log("⚠️  generator 里 catch 到错误:", e.message);
        return "查询失败: " + e.message;
    }
});

// 延迟执行，避免和测试1的输出混在一起
setTimeout(() => {
    console.log("\n--- 测试2: 错误处理 ---");
    getInfoSafe(1).then(result => {
        console.log("✅ 最终结果:", result);
    });
}, 1000);


// ---- 测试 3：对比原生 async/await ----
// 一模一样的逻辑，用原生语法写

setTimeout(async () => {
    console.log("\n--- 测试3: 对比原生 async/await ---");

    // 用原生 Promise 版本的 fetchUser
    function fetchUserNative(id) {
        return new Promise((resolve) => {
            setTimeout(() => resolve({ id, name: "小明" }), 300);
        });
    }
    function fetchOrdersNative(userId) {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve([
                    { id: 101, item: "键盘" },
                    { id: 102, item: "鼠标" },
                ]);
            }, 300);
        });
    }

    const user = await fetchUserNative(1);
    console.log("拿到用户:", user);
    const orders = await fetchOrdersNative(user.id);
    console.log("拿到订单:", orders);
    console.log("✅ 最终结果:", `${user.name} 有 ${orders.length} 个订单`);
}, 2500);
