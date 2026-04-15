"""
工具注册表 + 工具实现

新增工具只需要：
1. 在 TOOLS 列表加一项（给模型看的描述）
2. 写一个 _exec_xxx 函数（真正执行的代码）
3. 在 TOOL_MAP 加一条映射
"""

import json


# ============================================================
# 工具描述（给模型看的，模型根据 description 决定调不调）
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 北京"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如 3*5+2",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
]


# ============================================================
# 工具实现（模型看不到这些代码）
# ============================================================
def _exec_get_weather(city: str) -> str:
    fake_data = {"北京": "晴，23°C", "上海": "多云，19°C"}
    return fake_data.get(city, f"{city}：暂无数据")


def _exec_calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


# 工具名 -> 实现函数
TOOL_MAP = {
    "get_weather": _exec_get_weather,
    "calculator": _exec_calculator,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """统一的工具执行入口"""
    func = TOOL_MAP.get(name)
    if func is None:
        return f"未知工具: {name}"
    args = json.loads(arguments_json)
    return str(func(**args))
