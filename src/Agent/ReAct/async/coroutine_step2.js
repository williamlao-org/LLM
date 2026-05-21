// ============================================================
// 第二步：next() 可以往生成器里面"塞值"
// ============================================================

// yield 不只是"输出"一个值，它还能"接收"一个值！
// 当外面调 it.next(someValue) 时，
// 生成器里面的 yield 表达式的返回值就是 someValue

function* conversation() {
    // 第一次 next() 让函数跑到这里停住
    // yield "你叫什么？" 把问题交给外面
    const name = yield "你叫什么？";
    // ↑ 第二次 next("小明") 时，name 就变成了 "小明"

    console.log(`你好，${name}！`);

    const age = yield "你多大了？";
    // ↑ 第三次 next(20) 时，age 就变成了 20

    console.log(`${name} 今年 ${age} 岁`);

    return "对话结束";
}

const it = conversation();

// 第一次 next()：启动生成器，跑到第一个 yield
const q1 = it.next();
console.log("生成器问:", q1.value);  // "你叫什么？"

// 第二次 next("小明")：把 "小明" 塞给第一个 yield，函数从那里继续跑
const q2 = it.next("小明");
console.log("生成器问:", q2.value);  // "你多大了？"

// 第三次 next(20)：把 20 塞给第二个 yield
const q3 = it.next(20);
console.log("生成器说:", q3.value);  // "对话结束"
console.log("结束了吗:", q3.done);   // true
