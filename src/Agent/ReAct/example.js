// ============================================================
// 例 1：最简单的情况 —— 同步 resolve
// executor 里直接算出一个值，立刻 resolve
// ============================================================
const p1 = new Promise((resolve, reject) => {
    // "生产"：算出 2 + 3 的结果
    const result = 2 + 3;
    // "投递"：把 5 塞进 Promise
    resolve(result);
});

p1.then((value) => {
    // "消费"：value 就是上面 resolve 传进来的 5
    console.log("例1:", value);  // 例1: 5
});


// ============================================================
// 例 2：异步 resolve —— 1秒后才有值
// executor 立刻执行，但 resolve 被延迟调用
// ============================================================
const p2 = new Promise((resolve, reject) => {
    console.log("例2: executor 立刻执行了");

    // "生产"被推迟到 1 秒后
    setTimeout(() => {
        const data = { name: "小明", age: 20 };
        resolve(data);  // 1 秒后才"投递"
    }, 1000);
});

// then 先注册好回调，等 resolve 被调用时才触发
p2.then((value) => {
    console.log("例2:", value);  // 例2: { name: '小明', age: 20 }  （1秒后打印）
});


// ============================================================
// 例 3：reject 的情况
// executor 里发现出错了，调 reject 而不是 resolve
// ============================================================
const p3 = new Promise((resolve, reject) => {
    const age = -5;
    if (age < 0) {
        // "生产"了一个错误
        reject(new Error("年龄不能为负数"));
    } else {
        resolve(age);
    }
});

p3.then(
    (value) => console.log("例3 成功:", value),       // 不会执行
    (error) => console.log("例3 失败:", error.message) // 例3 失败: 年龄不能为负数
);


// ============================================================
// 例 4：真实场景 —— 把回调风格的 API 包装成 Promise
// 这才是 Promise 最常见的用法：包装已有的异步操作
// ============================================================
//
// 假设有一个老式的回调风格函数：
function fetchUserCallback(id, callback) {
    setTimeout(() => {
        if (id === 1) callback(null, { id: 1, name: "小红" });
        else callback(new Error("用户不存在"));
    }, 500);
}

// 用 Promise 包装它：
function fetchUser(id) {
    return new Promise((resolve, reject) => {
        // executor 里调用老式 API
        fetchUserCallback(id, (err, user) => {
            //                      ↑
            //            这个回调在 500ms 后被调用
            //            此时才决定 resolve 还是 reject
            if (err) reject(err);
            else resolve(user);   // ← 把 user 投递进 Promise
        });
    });
}

// 使用时，onFul 拿到的就是 resolve 投递的 user 对象
fetchUser(1).then((user) => {
    console.log("例4:", user);  // 例4: { id: 1, name: '小红' }
});

fetchUser(999).then(
    (user) => console.log("例4:", user),
    (err)  => console.log("例4 错误:", err.message)  // 例4 错误: 用户不存在
);


// ============================================================
// 例 5：链式 then —— 上一个 then 的返回值变成下一个 then 的 value
// ============================================================
const p5 = new Promise((resolve, reject) => {
    resolve(10);
});

p5
    .then((value) => {
        console.log("例5 第一步:", value);   // 10
        return value * 2;                    // return 20 → 自动包成 Promise.resolve(20)
    })
    .then((value) => {
        console.log("例5 第二步:", value);   // 20
        return value + 5;                    // return 25
    })
    .then((value) => {
        console.log("例5 第三步:", value);   // 25
    });


// ============================================================
// 例 6：对比——如果没有 Promise，同样的逻辑长什么样（回调地狱）
// ============================================================
//
// 需求：查用户 → 查订单 → 查订单详情
//
// --- 回调地狱版 ---
// fetchUserCallback(1, (err, user) => {
//     if (err) { handleError(err); return; }
//     fetchOrdersCallback(user.id, (err, orders) => {
//         if (err) { handleError(err); return; }
//         fetchDetailCallback(orders[0].id, (err, detail) => {
//             if (err) { handleError(err); return; }
//             console.log(detail);    // 终于拿到了，但已经缩进三层
//         });
//     });
// });
//
// --- Promise 版 ---
// fetchUser(1)
//     .then(user => fetchOrders(user.id))
//     .then(orders => fetchDetail(orders[0].id))
//     .then(detail => console.log(detail))
//     .catch(err => handleError(err));   // 错误处理只需要写一次！
