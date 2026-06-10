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
        // 处理 then/catch 回调的返回值：
        // - 普通值：让 then 返回的新 Promise fulfilled
        // - MyPromise：让 then 返回的新 Promise 跟随它的最终状态
        function handleResult (result, resolve, reject) {
            if (result instanceof MyPromise) {
                result.then(resolve, reject);
            } else {
                resolve(result);
            }
        }
        // then 一定会返回一个新的 Promise。
        // 这里的 this 是上一个 Promise，resolve/reject 属于这个新 Promise。
        return new MyPromise((resolve, reject) => {
            if (this.state === 'fulfilled') {
                // 同样是 `.then()`，有时候 fn 同步执行，有时候异步执行。
                // 调用者无法预测自己的代码执行顺序——这很危险。
                // 所以标准 Promise 做了一个保证：
                // then 的回调永远异步执行，即使 Promise 已经 fulfilled 了。

                queueMicrotask(() => {
                    if (fnResolve) {
                        try {
                            const result = fnResolve(this.value)
                            handleResult(result, resolve, reject);
                        } catch (e) {
                            // 回调自己抛错时，下一个 Promise 进入 rejected。
                            reject(e)
                        }
                    } else {
                        // 没有成功回调时，成功值原样穿透到下一个 Promise。
                        resolve(this.value)
                    }
                })


            }
            else if (this.state === 'rejected') {
                queueMicrotask(() => {
                    // 不管是 fnReject 还是 fnResolve，正常情况都会返回值并路由到下一个 Promise 的resolve。只有 fnReject/ fnResolve 抛错时，才会路由到下一个 Promise 的 reject。
                    // 因此 catch 只需要捕获 fnReject/ fnResolve 抛出的错误，因此try catch 在 if 中，而不是if在 try 中。
                    if (fnReject) {
                        try {
                            const result = fnReject(this.value)
                            handleResult(result, resolve, reject);
                        } catch (e) {
                            // 错误处理函数自己抛错时，下一个 Promise 继续 rejected。
                            reject(e)
                        }
                    } else {
                        // 没有失败回调时，失败原因原样穿透到下一个 Promise。
                        reject(this.value)
                    }
                })
            }
            else {
                // 当前 Promise 还 pending：先把“唤醒下一个 Promise”的函数挂到当前 Promise 上。
                // 等当前 Promise resolve/reject 后，构造函数里的队列会依次执行这些函数。
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
                        // p.then().then(...)：没有 fnResolve，也要把 value 传给 then 返回的新 Promise。
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
                        // p.then().catch(...)：没有 fnReject，也要把 err 传给 then 返回的新 Promise。
                        reject(err)
                    })
            }

        }
        )
    }
    catch (fnReject) {
        return this.then(undefined, fnReject)
    }
}


export default MyPromise;