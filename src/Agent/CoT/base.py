"""
base.py - CoT 基类

这是所有 CoT 方法的抽象基类，定义了统一的接口。

CoT 的核心流程（无论哪种变体）都是：
    1. 构造 prompt（包含推理引导）
    2. 调用 LLM 获取响应（推理过程 + 答案）
    3. 从响应中提取最终答案

为什么需要基类？
    - Few-Shot CoT 和 Zero-Shot CoT 的 prompt 构造方式不同
    - 但它们的调用流程和答案提取逻辑是相同的
    - 基类封装公共逻辑，子类只需实现 prompt 构造
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CoTResult:
    """
    CoT 推理的结果，包含完整信息。

    Attributes:
        question: 原始问题
        prompt: 构造好的完整 prompt（发给 LLM 的内容）
        raw_response: LLM 返回的原始响应（包含推理链）
        reasoning: 提取出的推理过程
        answer: 提取出的最终答案
    """

    question: str
    prompt: str
    raw_response: str
    reasoning: str
    answer: str

    def __str__(self) -> str:
        return (
            f"{'='*60}\n"
            f"问题: {self.question}\n"
            f"{'-'*60}\n"
            f"推理过程:\n{self.reasoning}\n"
            f"{'-'*60}\n"
            f"最终答案: {self.answer}\n"
            f"{'='*60}"
        )


class CoTBase(ABC):
    """
    Chain-of-Thought 基类。

    这个类定义了 CoT 的核心接口和公共逻辑。
    子类 (FewShotCoT, ZeroShotCoT) 只需要实现 build_prompt() 方法。

    使用方式：
        cot = SomeCoTSubclass(llm_call_fn=my_llm_function)
        result = cot.run("什么是 2 + 3？")
        print(result)

    参数：
        llm_call_fn: 一个可调用对象，接受 prompt (str)，返回 LLM 的响应 (str)
                     这样设计是为了与任何 LLM API 解耦
                     例如：可以是 OpenAI、HuggingFace、本地模型等

    为什么用函数而不是直接集成某个 API？
        → 解耦！CoT 是一种 prompt 技术，不应该绑定特定的 LLM 服务
        → 你可以传入任何函数：API 调用、本地模型推理、甚至是 mock 函数用于测试
    """

    def __init__(self, llm_call_fn=None):
        """
        Args:
            llm_call_fn: 调用 LLM 的函数。签名: (prompt: str) -> str
                         如果为 None，则使用内置的模拟函数（用于学习和测试）
        """
        if llm_call_fn is not None:
            self.llm_call = llm_call_fn
        else:
            # 默认使用模拟 LLM，方便学习时不需要真实 API
            self.llm_call = self._mock_llm_call

    @abstractmethod
    def build_prompt(self, question: str) -> str:
        """
        构造 CoT prompt。

        这是子类必须实现的核心方法。
        不同的 CoT 方法区别就在于如何构造 prompt：
            - Few-Shot CoT: 拼接示例 + 新问题
            - Zero-Shot CoT: 问题 + "Let's think step by step"

        Args:
            question: 用户的问题

        Returns:
            构造好的完整 prompt 字符串
        """
        pass

    def run(self, question: str) -> CoTResult:
        """
        执行完整的 CoT 推理流程。

        流程:
            1. build_prompt()  →  构造 prompt
            2. llm_call()      →  调用 LLM，获取包含推理链的响应
            3. extract_answer() → 从响应中提取最终答案

        Args:
            question: 用户的问题

        Returns:
            CoTResult 对象，包含推理过程和最终答案
        """
        # Step 1: 构造 prompt
        prompt = self.build_prompt(question)
        print(f"\n📝 构造的 Prompt:\n{'-'*40}\n{prompt}\n{'-'*40}")

        # Step 2: 调用 LLM
        raw_response = self.llm_call(prompt)
        print(f"\n🤖 LLM 原始响应:\n{'-'*40}\n{raw_response}\n{'-'*40}")

        # Step 3: 提取推理过程和最终答案
        reasoning, answer = self.extract_answer(raw_response)

        # 封装结果
        result = CoTResult(
            question=question,
            prompt=prompt,
            raw_response=raw_response,
            reasoning=reasoning,
            answer=answer,
        )

        return result

    def extract_answer(self, response: str) -> tuple[str, str]:
        """
        从 LLM 的响应中提取推理过程和最终答案。

        常见的答案标记模式:
            - "Therefore, the answer is X"
            - "The answer is X"
            - "So the answer is X"

        Args:
            response: LLM 的原始响应文本

        Returns:
            (reasoning, answer) 元组
        """
        # 尝试匹配 "the answer is ..." 模式
        # re.IGNORECASE: 忽略大小写
        # re.DOTALL: 让 . 也能匹配换行符
        answer_pattern = r"(?:therefore|so|thus|hence),?\s*the answer is\s*(.+?)\.?\s*$"
        match = re.search(answer_pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            answer = match.group(1).strip()
            # 推理过程 = 整个响应去掉最终答案部分
            reasoning = response[: match.start()].strip()
        else:
            # 如果没有匹配到标准模式，把最后一行当作答案
            lines = response.strip().split("\n")
            answer = lines[-1].strip() if lines else ""
            reasoning = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""

        return reasoning, answer

    def _mock_llm_call(self, prompt: str) -> str:
        """
        模拟 LLM 调用（用于学习和测试）。

        这不是真正的 LLM！只是一个简单的模拟，用来演示 CoT 的流程。
        在实际使用中，你应该传入真实的 LLM 调用函数。

        原理：它会检测 prompt 中的问题，返回预设的推理响应。
        """
        # 简单的关键词匹配来模拟不同类型的回答
        prompt_lower = prompt.lower()

        if "parking lot" in prompt_lower or "cars" in prompt_lower:
            return (
                "There are already 3 cars in the parking lot. "
                "Then 2 more cars arrive. "
                "We need to add 3 + 2 = 5. "
                "Therefore, the answer is 5."
            )
        elif "chicken" in prompt_lower or "鸡" in prompt_lower:
            return (
                "The farm starts with 23 chickens. "
                "Then 2 boxes are bought, each containing 12 chickens. "
                "The number of new chickens is 2 × 12 = 24. "
                "The total is 23 + 24 = 47. "
                "Therefore, the answer is 47."
            )
        elif "apple" in prompt_lower or "苹果" in prompt_lower:
            return (
                "小明开始有 5 个苹果。"
                "给了小红 2 个，剩余 5 - 2 = 3 个。"
                "又买了 3 个，变成 3 + 3 = 6 个。"
                "Therefore, the answer is 6."
            )
        else:
            return (
                "Let me think about this step by step. "
                "Based on the information given, I need to analyze the problem carefully. "
                "After considering all factors, "
                "Therefore, the answer is [需要真实 LLM 来回答]."
            )
