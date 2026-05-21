import MyPromise from './myPromise.js';

function testMyPromiseCatch () {
    const p3 = new MyPromise((resolve) => {
        resolve(1);
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

}

testMyPromiseCatch();

function testMyPromiseChain () {

    function getUser (id) {
        const p = new MyPromise((resolve) => {
            setTimeout(() => {
                resolve({ name: "小明" });
            }, 1000);
        });
        return p;
    }

    function getOrders (user) {
        const p = new MyPromise((resolve) => {
            setTimeout(() => {
                resolve([{ id: 1, item: "书" }, { id: 2, item: "笔" }]);
            }, 1000);
        });
        return p;
    }

    function getOrderDetails (order) {
        const p = new MyPromise((resolve) => {
            setTimeout(() => {
                resolve({ ...order, price: 100 });
            }, 1000);
        });
        return p;
    }

    getUser(1)
        .then(user => getOrders(user))
        .then(orders => getOrderDetails(orders))
        .then(detail => console.log(detail))

    // 如果要同时拿到 user 和 orders，可以在 then 里返回一个新的 Promise，等它 resolve 后再继续链式调用。
    getUser(1)
        .then(user => {
            return getOrders(user).then(orders => {
                return { user, orders }
            })
        })
        .then(({ user, orders }) => {
            return getOrderDetails(orders[0]).then(detail => {
                return { user, orders, detail }
            })
        })
        .then(({ user, orders, detail }) => {
            console.log(user, orders, detail)
        })
}


testMyPromiseChain();