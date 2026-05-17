import httpx
import os

from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def calculate(num1, num2):
    return num1 + num2


def web_search(query: str, max_results):
    resp = httpx.post(
        url="https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        results.append({"title": title, "snippet": content, "url": url})
    return results

def execute_python(code:str):
    


calculate_tool = Tool(
    name="calculate",
    description="Add two numbers and return the sum.",
    parameters={
        "type": "object",
        "properties": {
            "num1": {"type": "number", "description": "The first number"},
            "num2": {"type": "number", "description": "The second number"},
        },
        "required": ["num1", "num2"],
    },
    func=calculate,
)

web_search_tool = Tool(
    name="web_search",
    description="Search the web for information about a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
            },
        },
        "required": ["query", "max_results"],
    },
    func=web_search,
)


tools: list[Tool] = []

tools.append(calculate_tool)
tools.append(web_search_tool)