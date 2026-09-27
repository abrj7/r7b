"""Keep a trial key inside its monthly call cap and per-minute rate limits.

Every live request is appended to a ledger before it is sent. The ledger is
the source of truth for "how many calls have I used this month", so a crashed
or interrupted run still counts what it spent.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from r7b.config import EMBED_BATCH, Budget


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, budget: Budget, now: Callable[[], datetime] | None = None):
        self.path = path
        self.budget = budget
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()

    def _month(self, ts: datetime) -> str:
        return ts.strftime("%Y-%m")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def used_this_month(self) -> int:
        month = self._month(self._now())
        return sum(1 for e in self.entries() if e["ts"].startswith(month))

    def remaining(self) -> int:
        return self.budget.usable - self.used_this_month()

    def record(self, endpoint: str, model: str, note: str = "") -> None:
        """Reserve one call. Raises before the request is sent if the cap would be crossed."""
        with self._lock:
            if self.remaining() <= 0:
                raise BudgetExceeded(
                    f"monthly call budget spent ({self.budget.usable} usable of "
                    f"{self.budget.monthly_cap}); cached work is kept, rerun next month"
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"ts": self._now().isoformat(), "endpoint": endpoint, "model": model,
                     "note": note}
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")


class RateLimiter:
    """Evenly spaced requests, per key (one key per model for chat)."""

    def __init__(self, per_minute: float, safety: float = 0.9,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / (per_minute * safety)
        self._next: dict[str, float] = {}
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def wait(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(key, now))
            self._next[key] = slot + self.interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)


@dataclass(frozen=True)
class CallPlan:
    queries: int
    open_queries: int
    judge_validation_items: int
    chat_calls: int
    judge_calls: int
    embed_calls: int

    @property
    def total(self) -> int:
        return self.chat_calls + self.judge_calls + self.embed_calls

    def fits(self, budget: Budget) -> bool:
        return self.total <= budget.usable

    def to_dict(self, budget: Budget) -> dict:
        return {
            "queries": self.queries,
            "open_queries": self.open_queries,
            "judge_validation_items": self.judge_validation_items,
            "chat_calls": self.chat_calls,
            "judge_calls": self.judge_calls,
            "embed_calls": self.embed_calls,
            "total_calls": self.total,
            "monthly_cap": budget.monthly_cap,
            "reserve": budget.reserve,
            "headroom": budget.usable - self.total,
        }


def plan_calls(n_queries: int, n_open: int, n_validation: int,
               embed_batch: int = EMBED_BATCH) -> CallPlan:
    """Both models answer every query; one judge call grades both answers to an
    open-ended query; one judge call per hand-labeled validation pair; embeddings
    are batched."""
    return CallPlan(
        queries=n_queries,
        open_queries=n_open,
        judge_validation_items=n_validation,
        chat_calls=2 * n_queries,
        judge_calls=n_open + n_validation,
        embed_calls=math.ceil(n_queries / embed_batch),
    )
