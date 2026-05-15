import httpx
from dotenv import load_dotenv
import os
import json

load_dotenv()


def web_search(query: str, max_results=5):
    resp = httpx.post(
        url="https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": True,
            "max_results": max_results,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))


web_search("claude官方是不是这几天说额度上涨了？")
