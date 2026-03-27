import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict, cast
from urllib.parse import urlencode
from urllib.request import urlopen

from openai import OpenAI


# 意图
Route = Literal[
    "respond_direct",  # 直接生成回复
    "call_tool",  # 调用工具
    "clarify",  # 先澄清后回复
    "reject_or_safe_reply",  # 拒绝或安全回复
    "handoff_human",  # 转接人工
]
Domain = Literal["weather", "general", "unknown"]
ToolName = Literal["weather_api", "joke_db"]
Message = Dict[str, Any]
Messages = List[Message]
NodeName = Literal[
    "understand",
    "plan",
    "tool_select",
    "tool_run",
    "generate",
    "validate",
    "retry",
    "fallback",
    "handoff",
    "end",
]

MODEL_PRIMARY = os.getenv("OPENAI_MODEL", "Qwen3.5-35B-A3B-UD-Q4_K_L.gguf")
MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "Qwen2.5-14B-Instruct")
# BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.64.0.4:8080/v1")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-sp-aa6a935d0d864e1c910cbe757e162fdc")

MODEL_PRIMARY = os.getenv("OPENAI_MODEL", "glm-5")
MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "Qwen2.5-14B-Instruct")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-sp-aa6a935d0d864e1c910cbe757e162fdc")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fsm-agent")


class State(TypedDict):
    session_id: str
    user_input: str
    messages: Messages

    route: Route
    domain: Domain
    confidence: float

    plan: Optional[str]
    need_tool: bool
    selected_tool: Optional[ToolName]
    tool_name: Optional[ToolName]
    tool_args: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]

    draft_answer: Optional[str]
    final_answer: Optional[str]

    retry_count: int
    max_retries: int
    model_index: int

    risk_flag: bool
    error: Optional[str]
    next_state: NodeName
    done: bool


class StateStore:
    def __init__(self) -> None:
        self._db: Dict[str, State] = {}

    def load(self, session_id: str) -> Optional[State]:
        return self._db.get(session_id)

    def save(self, state: State) -> None:
        self._db[state["session_id"]] = state


store = StateStore()


def transition(state: State, to_state: NodeName, reason: str) -> None:
    from_state = state["next_state"]
    state["next_state"] = to_state
    logger.info(
        json.dumps(
            {
                "event": "transition",
                "session_id": state["session_id"],
                "from": from_state,
                "to": to_state,
                "reason": reason,
                "retry_count": state["retry_count"],
                "model_index": state["model_index"],
            },
            ensure_ascii=False,
        )
    )


def call_llm(
    messages: Messages, temperature: float, max_tokens: int, model: str
) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=cast(Any, messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()


def call_llm_with_retry(
    state: State, messages: Messages, temperature: float, max_tokens: int
) -> str:
    models = [MODEL_PRIMARY, MODEL_FALLBACK]
    model = models[min(state["model_index"], len(models) - 1)]
    last_err: Optional[Exception] = None

    for i in range(3):
        try:
            return call_llm(
                messages, temperature=temperature, max_tokens=max_tokens, model=model
            )
        except Exception as e:
            last_err = e
            sleep_s = min(2.0 * (2**i), 8.0) + random.uniform(0, 0.4)
            logger.warning(
                "llm call failed model=%s attempt=%s err=%s", model, i + 1, e
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"LLM调用失败 model={model} err={last_err}")


def parse_first_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        v = json.loads(snippet)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def tool_weather(city: str) -> Dict[str, Any]:
    geo_params = urlencode(
        {"name": city, "count": 1, "language": "zh", "format": "json"}
    )
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{geo_params}"
    with urlopen(geo_url, timeout=8) as geo_resp:
        gj = json.loads(geo_resp.read().decode("utf-8"))

    if not gj.get("results"):
        return {"ok": False, "error": f"未找到城市: {city}"}

    item = gj["results"][0]
    lat = item["latitude"]
    lon = item["longitude"]
    cname = item.get("name", city)

    weather_params = urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code",
            "timezone": "auto",
        }
    )
    weather_url = f"https://api.open-meteo.com/v1/forecast?{weather_params}"
    with urlopen(weather_url, timeout=8) as weather_resp:
        wj = json.loads(weather_resp.read().decode("utf-8"))
    cur = wj.get("current", {})

    return {
        "ok": True,
        "city": cname,
        "temperature_2m": cur.get("temperature_2m"),
        "weather_code": cur.get("weather_code"),
    }


def tool_joke() -> Dict[str, Any]:
    return {
        "ok": True,
        "joke": "程序员去相亲，问对方会不会做饭。对方说会。他说太好了，我会吃。",
    }


TOOLS = {
    "weather_api": tool_weather,
    "joke_db": lambda _: tool_joke(),
}


def node_understand(state: State) -> None:
    prompt = [
        {
            "role": "system",
            "content": (
                "你是路由决策器。仅输出JSON。字段: "
                "route, domain, confidence, risk_flag, tool_name, tool_args。"
                "route只能是 respond_direct/call_tool/clarify/reject_or_safe_reply/handoff_human。"
                "domain只能是 weather/general/unknown。"
                "当route=call_tool时，tool_name只能是 weather_api 或 joke_db。"
                "若不需要工具，tool_name必须为 null，tool_args必须为 {}。"
            ),
        },
        {"role": "user", "content": state["user_input"]},
    ]
    try:
        raw = call_llm_with_retry(state, prompt, temperature=0.0, max_tokens=160)
        obj = parse_first_json(raw) or {}
        route = obj.get("route", "clarify")
        domain = obj.get("domain", "unknown")
        confidence = float(obj.get("confidence", 0.0))
        risk_flag = bool(obj.get("risk_flag", False))
        tool_name = obj.get("tool_name")
        tool_args = obj.get("tool_args", {})

        valid_routes = {
            "respond_direct",
            "call_tool",
            "clarify",
            "reject_or_safe_reply",
            "handoff_human",
        }
        if route not in valid_routes:
            route = "clarify"

        if domain not in {"weather", "general", "unknown"}:
            domain = "unknown"

        if tool_name not in {"weather_api", "joke_db"}:
            tool_name = None

        if not isinstance(tool_args, dict):
            tool_args = {}

        if route != "call_tool":
            tool_name = None
            tool_args = {}

        state["route"] = route
        state["domain"] = domain
        state["confidence"] = max(0.0, min(1.0, confidence))
        state["risk_flag"] = risk_flag
        state["tool_name"] = tool_name
        state["tool_args"] = tool_args

        if state["risk_flag"] or state["route"] == "handoff_human":
            transition(state, "handoff", "risk_or_handoff")
        else:
            transition(state, "plan", "decided")
    except Exception as e:
        state["error"] = f"understand失败: {e}"
        transition(state, "retry", "understand_error")


def node_plan(state: State) -> None:
    low_conf = state["confidence"] < 0.45
    if low_conf:
        state["plan"] = "低置信度，先澄清用户需求"
        state["need_tool"] = False
        transition(state, "generate", "low_confidence")
        return

    if state["route"] == "call_tool":
        state["plan"] = "按决策调用工具并融合结果"
        state["need_tool"] = True
        transition(state, "tool_select", "route_call_tool")
        return

    if state["route"] in {"clarify", "respond_direct", "reject_or_safe_reply"}:
        state["plan"] = "闲聊直接生成"
        state["need_tool"] = False
        transition(state, "generate", "route_generate")
        return

    transition(state, "fallback", "unknown_route")


def node_tool_select(state: State) -> None:
    candidate_tool = state["tool_name"]
    if candidate_tool in TOOLS:
        state["selected_tool"] = candidate_tool
        if candidate_tool == "weather_api":
            city = state["tool_args"].get("city") or extract_city_heuristic(
                state["user_input"]
            )
            state["tool_args"] = {"city": city}
        else:
            state["tool_args"] = {}
        transition(state, "tool_run", "tool_selected")
        return

    state["selected_tool"] = None
    transition(state, "generate", "no_valid_tool")


def extract_city_heuristic(text: str) -> str:
    for city in ["北京", "上海", "广州", "深圳", "杭州", "成都"]:
        if city in text:
            return city
    return "北京"


def node_tool_run(state: State) -> None:
    try:
        tool_name = state["selected_tool"]
        if not tool_name:
            transition(state, "generate", "empty_tool")
            return

        fn = TOOLS[tool_name]
        if tool_name == "weather_api":
            result = fn(state["tool_args"]["city"])
        else:
            result = fn({})

        state["tool_result"] = result
        transition(state, "generate", "tool_done")
    except Exception as e:
        state["error"] = f"tool_run失败: {e}"
        transition(state, "retry", "tool_error")


def node_generate(state: State) -> None:
    try:
        sys_prompt = "你是一个有帮助的中文助手。回答简洁、准确。"

        context = {
            "plan": state["plan"],
            "route": state["route"],
            "domain": state["domain"],
            "confidence": state["confidence"],
            "tool_result": state["tool_result"],
        }

        if state["route"] == "clarify" or state["confidence"] < 0.45:
            user_prompt = f"用户输入: {state['user_input']}。请先提出一个澄清问题。"
        elif state["route"] == "reject_or_safe_reply":
            user_prompt = (
                f"用户输入: {state['user_input']}。"
                "请给出安全、礼貌、简短的替代性答复，不要执行敏感请求。"
            )
        else:
            user_prompt = (
                f"用户输入: {state['user_input']}\n"
                f"上下文: {json.dumps(context, ensure_ascii=False)}\n"
                "请直接给最终答复。"
            )

        msgs = [
            {"role": "system", "content": sys_prompt},
            *state["messages"][-12:],
            {"role": "user", "content": user_prompt},
        ]
        answer = call_llm_with_retry(state, msgs, temperature=0.4, max_tokens=260)
        state["draft_answer"] = answer
        transition(state, "validate", "generated")
    except Exception as e:
        state["error"] = f"generate失败: {e}"
        transition(state, "retry", "generate_error")


def node_validate(state: State) -> None:
    try:
        check_msgs = [
            {
                "role": "system",
                "content": (
                    "你是结果校验器。仅输出JSON。字段: pass, reason。" "pass为布尔值。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户输入: {state['user_input']}\n"
                    f"候选答案: {state['draft_answer']}\n"
                    f"工具结果: {json.dumps(state['tool_result'], ensure_ascii=False)}"
                ),
            },
        ]
        raw = call_llm_with_retry(state, check_msgs, temperature=0.0, max_tokens=80)
        obj = parse_first_json(raw) or {}
        is_pass = bool(obj.get("pass", False))

        if is_pass:
            state["final_answer"] = state["draft_answer"]
            state["messages"].append({"role": "user", "content": state["user_input"]})
            state["messages"].append(
                {"role": "assistant", "content": state["final_answer"] or ""}
            )
            transition(state, "end", "validated")
        else:
            transition(state, "retry", "validate_failed")
    except Exception as e:
        state["error"] = f"validate失败: {e}"
        transition(state, "retry", "validate_error")


def node_retry(state: State) -> None:
    state["retry_count"] += 1

    if state["retry_count"] <= state["max_retries"]:
        if state["retry_count"] >= 2:
            state["model_index"] = 1
        transition(state, "plan", "retry_again")
        return

    transition(state, "fallback", "retry_exhausted")


def node_fallback(state: State) -> None:
    state["final_answer"] = (
        "我暂时无法稳定完成这次请求。"
        "我可以先给你一个保守建议，或你可以换个问法再试一次。"
    )
    transition(state, "end", "fallback_answer")


def node_handoff(state: State) -> None:
    state["final_answer"] = "你的请求需要人工处理，我已为你转人工。"
    transition(state, "end", "handoff")


def node_end(state: State) -> None:
    state["done"] = True


NODES = {
    "understand": node_understand,
    "plan": node_plan,
    "tool_select": node_tool_select,
    "tool_run": node_tool_run,
    "generate": node_generate,
    "validate": node_validate,
    "retry": node_retry,
    "fallback": node_fallback,
    "handoff": node_handoff,
    "end": node_end,
}


def init_state(
    session_id: str,
    user_input: str,
    prev_messages: Optional[Messages] = None,
) -> State:
    return {
        "session_id": session_id,
        "user_input": user_input,
        "messages": prev_messages or [],
        "route": "clarify",
        "domain": "unknown",
        "confidence": 0.0,
        "plan": None,
        "need_tool": False,
        "selected_tool": None,
        "tool_name": None,
        "tool_args": {},
        "tool_result": None,
        "draft_answer": None,
        "final_answer": None,
        "retry_count": 0,
        "max_retries": 3,
        "model_index": 0,
        "risk_flag": False,
        "error": None,
        "next_state": "understand",
        "done": False,
    }


def process_turn(session_id: str, user_input: str) -> State:
    old = store.load(session_id)
    prev_messages = old["messages"] if old else []

    state = init_state(
        session_id=session_id, user_input=user_input, prev_messages=prev_messages
    )

    max_steps = 50
    steps = 0
    while not state["done"]:
        steps += 1
        if steps > max_steps:
            state["error"] = "状态机超过最大步数"
            state["next_state"] = "fallback"

        fn = NODES.get(state["next_state"])
        if fn is None:
            state["error"] = f"未知节点: {state['next_state']}"
            state["next_state"] = "fallback"
            continue

        fn(state)

    store.save(state)
    return state


if __name__ == "__main__":
    sid = "demo-session-001"
    print("输入 exit 退出")
    while True:
        q = input("你: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        s = process_turn(sid, q)
        print("助手:", s["final_answer"])
        if s["error"]:
            print("[error]", s["error"])
