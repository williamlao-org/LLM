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
            this.state = 'rejected'
            this.value = e
            this.callbacksRejected.forEach(fn => queueMicrotask(() => fn(e)))
        }
        try {
            executor(resolve, reject);
        } catch (e) {
            reject(e);
        }
    }
    then(fnResolve, fnReject) {
        function handleResult(result, resolve, reject) {
            if (result instanceof MyPromise) {
                result.then(resolve, reject);
            } else {
                resolve(result);
            }
        }
        return new MyPromise((resolve, reject) => {
            if (this.state === 'fulfilled') {
                // 同样是 `.then()`，有时候 fn 同步执行，有时候异步执行。
                // 调用者无法预测自己的代码执行顺序——这很危险。
                // 所以标准 Promise 做了一个保证：
                // then 的回调永远异步执行，即使 Promise 已经 fulfilled 了。

                queueMicrotask(() => {
                    try {
                        const result = fnResolve(this.value)
                        handleResult(result, resolve, reject);
                    } catch (e) {
                        reject(e)
                    }
                })

            }
            else if (this.state === 'rejected') {
                queueMicrotask(() => {
                    try {
                        const result = fnReject(this.value)
                        handleResult(result, resolve, reject);
                    } catch (e) {
                        reject(e)
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

        }
        )
    }
    catch(fnReject) {
        return this.then(undefined, fnReject)
    }
}


function getUser(id) {
    const p = new MyPromise((resolve) => {
        setTimeout(() => {
            resolve({ name: "小明" });
        }, 1000);
    });
    return p;
}

u = getUser(1);
u.then((user) => {
    setTimeout(() => {
        console.log(user);
    }, 0);
});
u.catch((e) => {
    setTimeout(() => {
        console.log(e);
    }, 0);
});

const p3 = new MyPromise((resolve) => {
    resolve(1);
});

p3
    .then(v => {
        console.log("p3 first:", v);
        return v + 1;
    })
    .then(v => {
        console.log("p3 second:", v);
        return v + 1;
    })
    .then(v => {
        console.log("p3 third:", v);
    });

p3
    .then(v => {
        throw new Error("炸了")
    })
    .then(v => {
        console.log("这行不会执行")
    })
    .catch(e => {
        console.log("捕获到:", e.message)  // 应该打印 "捕获到: 炸了"
    })