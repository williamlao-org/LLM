"""
few_shot_cot.py - Few-Shot CoT 实现

Few-Shot CoT 是最经典的 CoT 方法：
    在 prompt 中提供几个带推理过程的示例，引导模型模仿这种推理格式。

来自论文: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
          (Wei et al., 2022, Google Brain)

核心思想:
    传统的 few-shot prompting:  (问题, 答案) 对
    Few-Shot CoT:               (问题, 推理过程, 答案) 三元组
                                          ↑
                                    这就是 CoT 的关键区别!

为什么 Few-Shot 比 Zero-Shot 更好？
    1. 示例提供了具体的推理格式，模型知道要怎么"展开"推理
    2. 示例中的推理风格会被模型模仿（推理粒度、语言风格等）
    3. 示例暗含了问题的类型，帮助模型判断需要哪种推理能力

示例选择的重要性:
    - 示例应该与问题类型相关（算术问题用算术示例）
    - 示例的推理过程要准确且清晰
    - 一般 4-8 个示例就足够了
"""

from .base import CoTBase
from .prompts import (
    ARITHMETIC_EXAMPLES,
    FEW_SHOT_EXAMPLE_TEMPLATE,
    FEW_SHOT_QUERY_TEMPLATE,
)


class FewShotCoT(CoTBase):
    """
    Few-Shot Chain-of-Thought 实现。

    用法:
        # 使用默认的算术示例
        cot = FewShotCoT()
        result = cot.run("What is 15 + 27?")

        # 使用自定义示例
        my_examples = [
            {
                "question": "...",
                "reasoning": "...",
                "answer": "..."
            }
        ]
        cot = FewShotCoT(examples=my_examples)
        result = cot.run("My question")

    参数:
        llm_call_fn: 调用 LLM 的函数 (可选)
        examples: few-shot 示例列表，每个示例是一个 dict:
                  {"question": str, "reasoning": str, "answer": str}
    """

    def __init__(self, llm_call_fn=None, examples: list[dict] = None):
        super().__init__(llm_call_fn=llm_call_fn)
        # 如果没有提供示例，使用预定义的算术示例
        self.examples = examples or ARITHMETIC_EXAMPLES

    def build_prompt(self, question: str) -> str:
        """
        构造 Few-Shot CoT 的 prompt。

        构造过程:
            1. 依次拼接每个示例（包含问题 + 推理过程 + 答案）
            2. 在末尾附上新问题 + "Let's think step by step."

        最终的 prompt 结构:
            ┌──────────────────────────────────────┐
            │ Q: [示例问题1]                        │
            │ A: [推理过程1]                        │
            │ Therefore, the answer is [答案1].     │
            │                                      │
            │ Q: [示例问题2]                        │
            │ A: [推理过程2]                        │
            │ Therefore, the answer is [答案2].     │
            │                                      │
            │ ...更多示例...                        │
            │                                      │
            │ Q: [新问题]      ← 用户的实际问题      │
            │ A: Let's think step by step.          │
            └──────────────────────────────────────┘

        模型看到这个 prompt 后，会模仿示例的格式来推理新问题。
        """
        prompt = ""

        # Part 1: 拼接所有 few-shot 示例
        for i, example in enumerate(self.examples):
            prompt += FEW_SHOT_EXAMPLE_TEMPLATE.format(
                question=example["question"],
                reasoning=example["reasoning"],
                answer=example["answer"],
            )
            # 示例之间加一个空行，提高可读性（对模型理解也有帮助）
            if i < len(self.examples) - 1:
                prompt += "\n"

        # Part 2: 附上新问题
        prompt += FEW_SHOT_QUERY_TEMPLATE.format(question=question)

        return prompt

    def add_example(self, question: str, reasoning: str, answer: str):
        """
        添加一个新的 few-shot 示例。

        你可以在运行时动态添加示例来改善模型的表现。
        当你发现模型在某类问题上表现不好时，可以添加该类问题的示例。

        Args:
            question: 示例问题
            reasoning: 推理过程
            answer: 最终答案
        """
        self.examples.append(
            {
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
            }
        )

    def set_examples(self, examples: list[dict]):
        """
        替换所有 few-shot 示例。

        Args:
            examples: 新的示例列表
        """
        self.examples = examples


# ============================================================
# 学习笔记：Few-Shot CoT 的最佳实践
# ============================================================
#
# 1. 示例数量:
#    - 通常 4-8 个示例效果最好
#    - 太少 → 模型可能不会稳定地模仿推理格式
#    - 太多 → 可能超出上下文窗口，且边际收益递减
#
# 2. 示例质量:
#    - 推理步骤要正确！错误的推理会误导模型
#    - 推理要详细但不冗余
#    - 每一步都应该是逻辑上必要的
#
# 3. 示例多样性:
#    - 示例应该覆盖不同的推理模式
#    - 例如：加法、减法、乘法、多步骤混合等
#
# 4. 示例与问题的相关性:
#    - 示例的类型应该与目标问题相关
#    - 用算术示例来回答算术问题，逻辑示例来回答逻辑问题
#
# 5. 推理格式的一致性:
#    - 所有示例应该使用相同的推理格式
#    - 例如都以 "Therefore, the answer is ..." 结尾
#    - 这样模型更容易模仿
