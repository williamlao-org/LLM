// ============================================================
// 第一步：理解 Generator —— "一个能停下来的函数"
// ============================================================

// 普通函数：调用一次，从头跑到尾，没法中途暂停
function normal() {
    console.log("A");
    console.log("B");
    console.log("C");
    return "done";
}

normal(); // 打印 A B C，返回 "done"，一口气跑完

console.log("--- 分割线 ---");

// 生成器函数：加了 *，里面可以用 yield
// yield 的意思就是"我先停在这，把控制权还给外面"
function* gen() {
    console.log("A");
    yield "暂停点1";     // 执行到这里就停了，把 "暂停点1" 交出去
    console.log("B");
    yield "暂停点2";     // 又停了
    console.log("C");
    return "done";       // 最后一次，函数真的结束了
}

// 调用生成器函数不会执行函数体！它只是创建一个"迭代器对象"
const it = gen();
console.log("生成器创建了，但函数体还没跑");

// 调用 it.next() 才会让函数体开始跑，跑到第一个 yield 就停
const r1 = it.next();
console.log("第一次 next 返回:", r1);
// → 打印 "A"，然后返回 { value: "暂停点1", done: false }

const r2 = it.next();
console.log("第二次 next 返回:", r2);
// → 打印 "B"，然后返回 { value: "暂停点2", done: false }

const r3 = it.next();
console.log("第三次 next 返回:", r3);
// → 打印 "C"，然后返回 { value: "done", done: true }
