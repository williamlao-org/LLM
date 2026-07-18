"""Shared, deterministic forgetting policy for durable Phase 4 memories."""

from __future__ import annotations

from datetime import datetime, timezone


DEFAULT_IMPORTANCE = 0.5
MAX_RECALL_BONUS_COUNT = 5


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to UTC."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def forgetting_score(
    *,
    fallback_at: str,
    last_recalled_at: str | None,
    importance: float,
    recall_count: int,
    retention_days: int,
    now: datetime | None = None,
) -> float:
    """Return the eviction score; values >= 1 are eligible for deletion.

    ``fallback_at`` is ``updated_at`` for semantic facts and ``created_at`` for
    episodes.  A recent successful recall resets the idle clock, while higher
    importance and repeated recalls extend the retention window.
    """

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    activity_at = parse_timestamp(last_recalled_at or fallback_at)
    idle_days = max(0.0, (current - activity_at).total_seconds() / 86_400)
    protected_days = (
        retention_days
        * (1 + importance)
        * (1 + min(max(recall_count, 0), MAX_RECALL_BONUS_COUNT)
           / MAX_RECALL_BONUS_COUNT)
    )
    return idle_days / protected_days
