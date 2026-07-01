import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# ========== 数据模型 ==========

class QueryStep(BaseModel):
    step_id: int = Field(gt=0, description="当前步骤的正整数序号")
    query: str = Field(min_length=1, description="需要检索的子查询语句")
    depends_on: Optional[List[int]] = Field(
        default=None, 
        description="该查询所依赖的前置步骤 step_id 列表。若无依赖则为 null 或空列表"
    )

class QueryPlan(BaseModel):
    steps: List[QueryStep] = Field(
        min_length=1,
        description="为了回答复杂问题拆解出的一系列按逻辑顺序排列的子查询",
    )


# ========== 核心拆分器 ==========

class QueryDecomposer:
    """
    负责将复杂的多跳问题拆分为一系列带有依赖关系的子查询。
    """
    
    SYSTEM_PROMPT = """你是一个专业的搜索意图分析和任务拆解专家。
你的任务是将用户复杂的、多跳的问题，拆解为一系列结构化的子查询。

规则：
1. 思考回答该问题所需的必要中间事实，按照逻辑推导的顺序拆分步骤。
2. 每个子查询应该是一个完整、独立的检索词或短语问句。
3. 如果一个子查询依赖之前某个子查询的结果作为线索（例如第一步查“作者是谁”，第二步查“该作者的其他作品”），请在 `depends_on` 字段中注明前置步骤的 step_id。
4. 如果原始问题非常简单，只需单步检索，则拆分为一个步骤。
5. 必须返回符合指定格式的 JSON 结构。

示例输入：
"Transformer提出者之一Ashish Vaswani目前就职的公司是做什么的？"

示例输出思考过程：
要回答此问题，首先需要查明"Ashish Vaswani目前就职的公司"，然后才能查"这家公司是做什么的"。

示例输出JSON：
{
  "steps": [
    {
      "step_id": 1,
      "query": "Ashish Vaswani 目前就职的公司 现状",
      "depends_on": null
    },
    {
      "step_id": 2,
      "query": "【Step 1 的公司】 是做什么的 业务介绍",
      "depends_on": [1]
    }
  ]
}
"""

    REPLAN_SYSTEM_PROMPT = """你是自适应多跳检索的重规划器。

初始计划中的某一步在改写重试后仍未获得充分证据。请利用已经确认的事实，
替换失败步骤和所有尚未执行的步骤；不要重复已经完成的查询。

规则：
1. 只返回后续替换步骤，不返回已完成步骤。
2. 新步骤 ID 必须从指定的 next_step_id 开始使用正整数，且不能重复。
3. depends_on 只能引用已完成步骤 ID，或本次计划中排在当前步骤之前的新步骤。
4. 每一步必须是可独立执行的具体检索查询。
5. 不输出思维链，只通过 submit_query_plan 提交结构化计划。"""

    def __init__(self, llm_client: OpenAI, model: str):
        self.llm_client = llm_client
        self.model = model

    def _request_plan(self, messages: list[dict]) -> QueryPlan:
        """调用 LLM 获取结构化计划；解析失败时向调用方抛错。"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_query_plan",
                    "description": "提交针对复杂问题的多步检索计划",
                    "parameters": QueryPlan.model_json_schema(),
                },
            }
        ]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": 0.1,
        }

        try:
            response = self.llm_client.chat.completions.create(
                **kwargs,
                tool_choice={
                    "type": "function",
                    "function": {"name": "submit_query_plan"},
                },
            )
        except Exception as error:
            message = str(error).lower()
            if "tool_choice" not in message and "thinking" not in message:
                raise
            response = self.llm_client.chat.completions.create(
                **kwargs,
                tool_choice="auto",
            )

        message = response.choices[0].message
        if message.tool_calls:
            raw = message.tool_calls[0].function.arguments
        else:
            raw = message.content or ""

        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("查询计划必须是 JSON 对象")
        return QueryPlan.model_validate(data)

    @staticmethod
    def _validate_plan(
        plan: QueryPlan,
        completed_ids: set[int] | None = None,
        minimum_step_id: int = 1,
    ) -> None:
        """确保步骤 ID 唯一，且依赖只能指向已经可用的步骤。"""
        available = set(completed_ids or set())
        new_ids: set[int] = set()

        for step in plan.steps:
            if step.step_id < minimum_step_id:
                raise ValueError(
                    f"Step {step.step_id} 小于允许的新步骤起始 ID {minimum_step_id}"
                )
            if step.step_id in available or step.step_id in new_ids:
                raise ValueError(f"重复的 step_id: {step.step_id}")

            invalid_dependencies = [
                dependency
                for dependency in (step.depends_on or [])
                if dependency not in available and dependency not in new_ids
            ]
            if invalid_dependencies:
                raise ValueError(
                    f"Step {step.step_id} 引用了尚不可用的依赖: "
                    f"{invalid_dependencies}"
                )
            new_ids.add(step.step_id)

    def decompose(self, question: str, verbose: bool = False) -> QueryPlan:
        """
        将复杂问题拆分为查询计划
        """
        if verbose:
            print(f"\n  🧠 [QueryDecomposer] 正在拆解复杂问题: \"{question}\"")

        try:
            plan = self._request_plan([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ])
            self._validate_plan(plan)
            
            if verbose:
                print("  📝 [QueryDecomposer] 拆解结果:")
                for step in plan.steps:
                    dep = f" (依赖: {step.depends_on})" if step.depends_on else ""
                    print(f"     Step {step.step_id}: {step.query}{dep}")
                    
            return plan
        except Exception as e:
            if verbose:
                print(f"  ⚠️ [QueryDecomposer] 解析失败，降级为单步查询: {e}")
            # 降级处理：不拆分，直接作为一个步骤
            return QueryPlan(steps=[QueryStep(step_id=1, query=question, depends_on=None)])

    def replan(
        self,
        original_question: str,
        completed_steps: list[dict],
        failed_step: QueryStep,
        remaining_steps: list[QueryStep],
        next_step_id: int,
        failed_context: dict | None = None,
        verbose: bool = False,
    ) -> QueryPlan:
        """保留已完成事实，只替换失败步骤和后续未执行计划。"""
        completed_ids = {
            int(item["step_id"])
            for item in completed_steps
            if "step_id" in item
        }
        payload = {
            "original_question": original_question,
            "completed_steps": completed_steps,
            "failed_step": failed_step.model_dump(),
            "failed_context": failed_context or {},
            "remaining_steps": [step.model_dump() for step in remaining_steps],
            "next_step_id": next_step_id,
        }

        if verbose:
            print(
                f"     🧭 重规划失败 Step {failed_step.step_id} 之后的剩余计划..."
            )

        plan = self._request_plan([
            {"role": "system", "content": self.REPLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ])
        self._validate_plan(
            plan,
            completed_ids=completed_ids,
            minimum_step_id=next_step_id,
        )
        return plan
