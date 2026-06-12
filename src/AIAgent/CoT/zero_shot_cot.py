"""
zero_shot_cot.py - Zero-Shot CoT 实现

Zero-Shot CoT 是最简单的 CoT 方法：
    不需要任何示例，只需要在问题后加上一句 "Let's think step by step."

来自论文: "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022)

核心发现:
    仅仅在 prompt 中加入 "Let's think step by step" 这句话，
    就能让 LLM 的推理能力显著提升，尤其是在算术和逻辑推理任务上。

为什么这句话有效？
    1. LLM 在训练时见过大量"step by step"形式的推理文本
    2. 这个触发词激活了模型内部的"推理模式"
    3. 模型会生成中间步骤，每个步骤都为下一步提供了额外的上下文

实际工作流程（论文中的两阶段方法）:
    阶段1 (Reasoning Extraction): prompt = 问题 + "Let's think step by step"
                                   → LLM 生成推理链
    阶段2 (Answer Extraction):     prompt = 问题 + 推理链 + "Therefore, the answer is"
                                   → LLM 给出最终答案
"""

from .base import CoTBase
from .prompts import ZERO_SHOT_COT_TRIGGER, ZERO_SHOT_TEMPLATE


class ZeroShotCoT(CoTBase):
    """
    Zero-Shot Chain-of-Thought 实现。

    用法:
        cot = ZeroShotCoT()
        result = cot.run("What is 15 + 27?")
        print(result)

    参数:
        llm_call_fn: 调用 LLM 的函数 (可选，默认使用模拟)
        trigger: 触发推理的关键句 (默认: "Let's think step by step.")
    """

    def __init__(self, llm_call_fn=None, trigger: str = None):
        super().__init__(llm_call_fn=llm_call_fn)
        # 你可以自定义触发词，但默认的 "Let's think step by step." 已经被证明最有效
        self.trigger = trigger or ZERO_SHOT_COT_TRIGGER

    def build_prompt(self, question: str) -> str:
        """
        构造 Zero-Shot CoT 的 prompt。

        做的事情非常简单：
            问题 + "Let's think step by step."

        就这样。就是这么简单。这就是 Zero-Shot CoT 的全部。

        Example:
            输入: "What is 2 + 3?"
            输出:
                Q: What is 2 + 3?
                A: Let's think step by step.
        """
        prompt = ZERO_SHOT_TEMPLATE.format(
            question=question,
            trigger=self.trigger,
        )
        return prompt


# ============================================================
# 学习笔记：为什么 Zero-Shot CoT 能工作？
# ============================================================
#
# 1. 大模型内部存储了大量的推理模式
#    - 训练数据中有很多 "step by step" 格式的文本（教科书、教程等）
#    - "Let's think step by step" 这个前缀激活了这些模式
#
# 2. 自回归生成的特性使得中间步骤成为自我提示
#    - 模型生成: "首先..." → 这成为下一步的上下文
#    - 模型继续: "然后..." → 基于前面所有步骤继续推理
#    - 每一步推理都在不断"缩小搜索空间"
#
# 3. 对比实验显示:
#    ┌─────────────────────────────┬──────────────┐
#    │ 方式                        │ MultiArith   │
#    │                             │ 准确率        │
#    ├─────────────────────────────┼──────────────┤
#    │ Standard (直接回答)          │ 17.7%        │
#    │ Zero-Shot CoT               │ 78.7%        │
#    │ Few-Shot CoT (8 exemplars)  │ 93.0%        │
#    └─────────────────────────────┴──────────────┘
#    (数据来自原论文，模型: InstructGPT-175B)
#
# 4. 不同触发词的效果对比:
#    - "Let's think step by step." → 最好
#    - "Let's work this out..."   → 也不错
#    - "First, ..."               → 效果一般
#    - 空 (无触发词)               → 效果最差
