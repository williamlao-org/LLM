// ============================================================
// Generator 完全教学：从闭包到双向通道
// ============================================================


// ============ 第一章：闭包 —— Generator 的基石 ============
//
// 函数执行时，局部变量住在"栈"上，函数一结束就销毁。
// 但如果返回的内部函数引用了这些变量，引擎会把它们搬到"堆"上保活。
// 这个 "函数 + 它捕获的外部变量" 的组合体，就叫闭包（Closure）。
//
// 说白了就是 "打包带走"。

function counter() {
  let count = 0;           // ← 被闭包捕获，搬到堆上

  return {
    add()  { count++; },
    get()  { return count; }
  };
}

const c = counter();       // counter() 执行完了，但 count 还活着
c.add();
c.add();
console.log(c.get());      // 2 —— 两个方法共享同一个 count


// ============ 第二章：Generator 基础 —— 单向输出 ============
//
// function* 定义一个生成器函数，yield 让它执行到一半暂停，把值吐出去。
// 必须先拿到同一个实例，再反复调 .next() 才能逐步推进。

function* myGenerator() {
  const a = 1;
  yield a;          // 第一次 .next() 到这里暂停，吐出 1
  const b = 2;
  yield a + b;      // 第二次 .next() 到这里暂停，吐出 3
}                   // 第三次 .next() → done: true

const gen = myGenerator();
console.log(gen.next());   // { value: 1, done: false }
console.log(gen.next());   // { value: 3, done: false }
console.log(gen.next());   // { value: undefined, done: true }
console.log(gen.next());   // { value: undefined, done: true } （之后永远是这个）


// ============ 第三章：编译器视角 —— 状态机 ============
//
// 编译器（V8 / Babel）看到 function* 和 yield 后，
// 会把它重写成一个闭包 + 状态机。
// 下面就是上面 myGenerator 的等价手写版本：

function myGeneratorCompiled() {
  let state = 0;     // 状态机指针（闭包捕获）
  let a, b;          // 原本的局部变量（闭包捕获）

  return {
    next: function () {
      switch (state) {
        case 0:
          a = 1;
          state = 1;
          return { value: a, done: false };       // 挂起，把控制权交给调用者

        case 1:
          b = 2;
          state = 2;
          return { value: a + b, done: false };   // 再次挂起

        default:
          return { value: undefined, done: true }; // 已结束，无论调多少次都返回这个
      }
    }
  };
}

const gen2 = myGeneratorCompiled();
console.log(gen2.next());  // { value: 1, done: false }
console.log(gen2.next());  // { value: 3, done: false }
console.log(gen2.next());  // { value: undefined, done: true }
console.log(gen2.next());  // { value: undefined, done: true }


// ============ 第四章：双向通道 —— yield 的完整能力 ============
//
// yield 不光能把值吐出去，还能接收外部塞回来的值：
//
//   const input = yield output;
//
// 这一行跨越了两次 .next() 调用：
//
//   本次 .next()       → 执行 yield output，把 output 吐出去，挂起
//   下次 .next(val)    → 恢复执行，yield 表达式求值为 val，赋给 input
//
// input 拿到的不是 output，而是未来 .next(val) 塞回来的 val。
// 一个出去，一个进来，只不过写在了同一行。

function* chat() {
  const question = yield "你好，请问你叫什么？";   // 吐出问候，等待回答
  yield `你好，${question}！`;                     // 用回答拼出回复
}

const gen3 = chat();
console.log(gen3.next());          // { value: "你好，请问你叫什么？", done: false }
console.log(gen3.next("小明"));    // { value: "你好，小明！", done: false }
//                     ↑ "小明" 塞回去，成为上一个 yield 的求值结果，赋给 question


// ============ 第五章：双向通道的编译版本 ============
//
// 把上面的 chat 生成器手动编译成闭包状态机：
// 关键：next 方法接收 input 参数，每个 case 恢复时第一件事就是
// 把 input 赋给上次 yield 左边的变量 —— 这就是"双向通道的入方向"。

function chatCompiled() {
  let state = 0;
  let question;       // 闭包捕获

  return {
    next: function (input) {
      switch (state) {
        case 0:
          // 第一次 .next()，不需要接收 input（第一次调用的入参会被忽略）
          state = 1;
          return { value: "你好，请问你叫什么？", done: false };

        case 1:
          question = input;   // ← 核心！恢复执行的第一件事：
                              //   完成上次 const question = yield ... 的赋值
          state = 2;
          return { value: `你好，${question}！`, done: false };

        default:
          return { value: undefined, done: true };
      }
    }
  };
}

const gen4 = chatCompiled();
console.log(gen4.next());          // { value: "你好，请问你叫什么？", done: false }
console.log(gen4.next("小明"));    // { value: "你好，小明！", done: false }
console.log(gen4.next());          // { value: undefined, done: true }


// ============ 总结 ============
//
// 闭包：函数能记住并访问它被创建时所在作用域的变量，即使那个作用域早已执行完毕。
//       说白了就是 "打包带走"。
//
// Generator 本质：编译器把 function* 改写成 闭包 + 状态机。
//   - 局部变量从栈上提升到堆上（靠闭包保活）
//   - yield 变成 switch/case 的断点
//   - .next() 就是推动 state 往下走
//
// 双向通道：
//   - 出：yield value → 吐出值给 .next() 的返回值
//   - 入：.next(val) → val 成为上一个 yield 表达式的求值结果
//
// 当 yield 具备了双向通信能力，它就是协程（Coroutine）：
//   ✅ 交出执行权（暂停自己）
//   ✅ 恢复执行（.next() 把执行权交回来）
//   ✅ 传递上下文（.next(val) 双向传数据）
//
// 这也是 async/await 的底层原理：
//   await 本质就是 yield 一个 Promise，
//   外部执行器在 Promise resolve 后调用 .next(resolvedValue) 把结果塞回来。

