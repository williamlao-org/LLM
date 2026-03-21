"""
examples.py - CoT 运行示例

这个文件展示了如何使用 Zero-Shot CoT 和 Few-Shot CoT。
直接运行这个文件即可查看效果:
    python -m src.Agent.CoT.examples

注意：
    这里使用的是内置的模拟 LLM（mock），不需要真实的 API Key。
    在实际使用中，你需要传入真实的 LLM 调用函数。
"""

from .few_shot_cot import FewShotCoT
from .prompts import LOGIC_EXAMPLES
from .zero_shot_cot import ZeroShotCoT


def demo_zero_shot_cot():
    """演示 Zero-Shot CoT"""
    print("\n" + "=" * 60)
    print("🧪 演示 1: Zero-Shot CoT")
    print("=" * 60)
    print("原理: 只需在问题后加 'Let's think step by step.'")
    print("不需要提供任何示例！\n")

    # 创建 Zero-Shot CoT（使用默认的模拟 LLM）
    cot = ZeroShotCoT()

    # 测试问题
    question = "A farm has 23 chickens. The farmer buys 2 boxes, each containing 12 chickens. How many chickens are there in total?"

    # 运行推理
    result = cot.run(question)

    print(f"\n✅ 最终结果:")
    print(result)


def demo_few_shot_cot():
    """演示 Few-Shot CoT"""
    print("\n" + "=" * 60)
    print("🧪 演示 2: Few-Shot CoT (算术)")
    print("=" * 60)
    print("原理: 在 prompt 中提供带推理过程的示例")
    print("模型会模仿示例的推理格式来回答新问题\n")

    # 创建 Few-Shot CoT（使用默认的算术示例 + 模拟 LLM）
    cot = FewShotCoT()

    # 新问题
    question = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"

    # 运行推理
    result = cot.run(question)

    print(f"\n✅ 最终结果:")
    print(result)


def demo_custom_examples():
    """演示使用自定义示例的 Few-Shot CoT"""
    print("\n" + "=" * 60)
    print("🧪 演示 3: Few-Shot CoT (自定义逻辑推理示例)")
    print("=" * 60)
    print("使用自定义的逻辑推理示例集\n")

    # 使用逻辑推理示例集
    cot = FewShotCoT(examples=LOGIC_EXAMPLES)

    question = "All birds can fly. A penguin is a bird. Can a penguin fly?"

    result = cot.run(question)

    print(f"\n✅ 最终结果:")
    print(result)


def demo_comparison():
    """对比 Zero-Shot CoT 和 Few-Shot CoT"""
    print("\n" + "=" * 60)
    print("🧪 演示 4: Zero-Shot vs Few-Shot 对比")
    print("=" * 60)

    question = "小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？"

    print(f"\n📋 同一个问题: {question}\n")

    # Zero-Shot CoT
    print("--- Zero-Shot CoT ---")
    zero_shot = ZeroShotCoT()
    zero_result = zero_shot.run(question)
    print(f"答案: {zero_result.answer}\n")

    # Few-Shot CoT
    print("--- Few-Shot CoT ---")
    few_shot = FewShotCoT()
    few_result = few_shot.run(question)
    print(f"答案: {few_result.answer}\n")


def demo_show_prompt_only():
    """
    只展示构造出的 prompt，不调用 LLM。
    这对于理解 CoT 的核心 —— prompt 构造 —— 非常有帮助。
    """
    print("\n" + "=" * 60)
    print("🔍 演示 5: 查看构造的 Prompt（不调用 LLM）")
    print("=" * 60)
    print("这个演示展示 CoT 的核心产物 —— prompt 本身\n")

    question = "A store had 45 apples. They sold 12 in the morning and 15 in the afternoon. How many are left?"

    # Zero-Shot CoT 的 prompt
    zero_shot = ZeroShotCoT()
    zero_prompt = zero_shot.build_prompt(question)
    print("📝 Zero-Shot CoT Prompt:")
    print("-" * 40)
    print(zero_prompt)
    print("-" * 40)

    # Few-Shot CoT 的 prompt
    few_shot = FewShotCoT()
    few_prompt = few_shot.build_prompt(question)
    print("\n📝 Few-Shot CoT Prompt:")
    print("-" * 40)
    print(few_prompt)
    print("-" * 40)

    print("\n💡 对比可以看出:")
    print("   - Zero-Shot: 只有问题 + 触发词，简洁但依赖模型自身的能力")
    print("   - Few-Shot:  包含多个示例，冗长但给模型提供了明确的推理格式")


if __name__ == "__main__":
    print("🎓 Chain-of-Thought (CoT) 思维链 学习示例")
    print("=" * 60)

    demo_show_prompt_only()
    demo_zero_shot_cot()
    demo_few_shot_cot()
    demo_custom_examples()
    demo_comparison()

    print("\n" + "=" * 60)
    print("🎉 学习完成！")
    print("=" * 60)
    print("\n📚 关键总结:")
    print("  1. CoT 的核心是 Prompt 构造，不是模型架构")
    print("  2. Zero-Shot CoT: 简单，加一句 'Let's think step by step'")
    print("  3. Few-Shot CoT: 更强，但需要准备好的示例")
    print("  4. 实际使用时，把 mock LLM 替换成真实 API 即可")
    print("  5. CoT 在大模型 (>100B) 上效果最好")
