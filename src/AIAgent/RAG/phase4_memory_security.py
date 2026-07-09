"""Phase 4 memory components shared deterministic safety helpers."""

from __future__ import annotations

import math
import re
from typing import Any


_SENSITIVE_LABEL = re.compile(
    r"(?i)(api.?key|password|passwd|access.?token|refresh.?token|"
    r"private.?key|bearer|银行卡|密码|令牌|私钥)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_. -]?key|password|passwd|access[_. -]?token|"
    r"refresh[_. -]?token|private[_. -]?key|密码|令牌|私钥)"
    r"(\s*[:=是为]\s*|\s+)([^\s,;，；]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
    re.DOTALL,
)


def redact_sensitive_text(value: str) -> str:
    """Redact common credentials before memory processing or persistence."""

    text = _PRIVATE_KEY_BLOCK.sub("[REDACTED PRIVATE KEY]", str(value))
    text = _BEARER_TOKEN.sub("Bearer [REDACTED]", text)
    text = _OPENAI_STYLE_KEY.sub("[REDACTED]", text)
    return _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )


def contains_sensitive_data(value: str) -> bool:
    """Conservatively reject memory fields that look credential-related."""

    text = str(value)
    return bool(_SENSITIVE_LABEL.search(text)) or redact_sensitive_text(text) != text


def sanitize_json(value: Any, *, max_string_length: int = 500) -> Any:
    """Recursively redact and bound JSON-like tool/trace data."""

    if isinstance(value, str):
        return redact_sensitive_text(value)[:max_string_length]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            clean_key = str(key)[:100]
            if _SENSITIVE_LABEL.search(clean_key):
                sanitized[clean_key] = "[REDACTED]"
            else:
                sanitized[clean_key] = sanitize_json(
                    item,
                    max_string_length=max_string_length,
                )
        return sanitized
    if isinstance(value, (list, tuple)):
        return [
            sanitize_json(item, max_string_length=max_string_length)
            for item in list(value)[:50]
        ]
    return redact_sensitive_text(str(value))[:max_string_length]


def validate_vector(vector: Any) -> list[float]:
    """Return a finite non-empty float vector or raise a precise error."""

    if not isinstance(vector, list) or not vector:
        raise ValueError("embedding 必须是非空浮点数组")
    clean: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("embedding 只能包含数值")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding 不能包含 NaN 或无穷大")
        clean.append(number)
    return clean
