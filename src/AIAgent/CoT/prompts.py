"""
prompts.py - Prompt 模板管理

CoT 的本质就是 Prompt 工程。这个模块管理所有的 prompt 模板。

核心思想：
    - Few-Shot CoT: 在 prompt 中提供带推理过程的示例，引导模型模仿
    - Zero-Shot CoT: 只需要一句 "Let's think step by step" 就能触发推理

为什么要单独管理 Prompt？
    1. 复用：同一个模板可以用于不同的问题
    2. 可维护：修改模板不需要改动逻辑代码
    3. 可扩展：方便添加新的模板类型
"""

# ============================================================
# Zero-Shot CoT 的 Prompt 模板
# ============================================================

# 这就是那句著名的"魔法咒语"
# 来自论文: "Large Language Models are Zero-Shot Reasoners" (2022)
# 只需要在问题后面加上这一句，就能显著提升 LLM 的推理能力
ZERO_SHOT_COT_TRIGGER = "Let's think step by step."

ZERO_SHOT_TEMPLATE = """Q: {question}
A: {trigger}
"""

# ============================================================
# Few-Shot CoT 的 Prompt 模板
# ============================================================

# 单个示例的模板
# 注意：每个示例必须包含完整的推理过程(reasoning)，这是 Few-Shot CoT 的关键
FEW_SHOT_EXAMPLE_TEMPLATE = """Q: {question}
A: {reasoning}
Therefore, the answer is {answer}.
"""

# 新问题的模板（放在所有示例之后）
FEW_SHOT_QUERY_TEMPLATE = """Q: {question}
A: Let's think step by step.
"""

# ============================================================
# 答案提取的 Prompt 模板
# ============================================================

# Zero-Shot CoT 实际上有两个阶段：
#   阶段1: 让模型生成推理过程
#   阶段2: 从推理过程中提取最终答案
# 这是阶段2用的模板

ANSWER_EXTRACTION_TEMPLATE = """Q: {question}
A: {reasoning}
Therefore, the answer (arabic numerals) is"""


# ============================================================
# 预定义的 Few-Shot 示例集（用于数学推理）
# ============================================================

# 这些是经典的算术推理示例
# 每个示例都包含：question（问题）、reasoning（推理过程）、answer（最终答案）
ARITHMETIC_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "reasoning": "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees.",
        "answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "reasoning": "There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars.",
        "answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "reasoning": "Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates.",
        "answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "reasoning": "Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops.",
        "answer": "8",
    },
]

# 逻辑推理示例
LOGIC_EXAMPLES = [
    {
        "question": "Tom is older than Jack. Jack is older than Mary. Is Tom older than Mary?",
        "reasoning": "Tom is older than Jack, and Jack is older than Mary. Since the 'older than' relationship is transitive, if Tom > Jack and Jack > Mary, then Tom > Mary.",
        "answer": "Yes, Tom is older than Mary",
    },
    {
        "question": "All dogs are animals. Buddy is a dog. Is Buddy an animal?",
        "reasoning": "We know that all dogs are animals (premise 1). We also know that Buddy is a dog (premise 2). By deductive reasoning (modus ponens), since Buddy belongs to the category 'dog', and all dogs are animals, Buddy must be an animal.",
        "answer": "Yes, Buddy is an animal",
    },
]


few_shot_prompt = """
请解决以下逻辑数学问题。

示例 1：l
问题：餐厅原本有 50 个苹果。厨师用了 20 个做派，然后又买了 10 个。现在有几个？
思考过程：
1. 初始苹果数量：50
2. 用掉 20 个，剩下：50 - 20 = 30
3. 又买 10 个，总计：30 + 10 = 40
答案：40

示例 2：
问题：Roger 有 5 个网球。他又买了两罐网球，每罐有 3 个。他现在有几个网球？
思考过程：
1. 初始网球数量：5
2. 买了两罐，每罐 3 个，所以新增数量：2 * 3 = 6
3. 最终总计：5 + 6 = 11
答案：11

真实问题：
问题：一家书店原本有 120 本科幻小说。上午卖出了总数的四分之一，下午又进货了 45 本。接着，店员将剩下的科幻小说平均分到 3 个书架上。请问每个书架上有几本？
思考过程：
"""

zero_shot_prompt = """
问题：一家书店原本有 120 本科幻小说。上午卖出了总数的四分之一，下午又进货了 45 本。接着，店员将剩下的科幻小说平均分到 3 个书架上。请问每个书架上有几本？
答案：Let's think step by step.
"""
