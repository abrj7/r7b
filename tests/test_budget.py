from datetime import UTC, datetime

import pytest

from r7b.budget import BudgetExceeded, Ledger, RateLimiter, plan_calls
from r7b.config import Budget
from r7b.data import SAMPLE_SIZES, load_queries


def test_plan_counts_calls():
    plan = plan_calls(n_queries=380, n_open=80, n_validation=16)
    assert plan.chat_calls == 760
    assert plan.judge_calls == 96
    assert plan.embed_calls == 4  # 380 / 96 rounded up
    assert plan.total == 860


def test_committed_query_set_fits_a_trial_key():
    queries = load_queries()
    assert len(queries) == sum(SAMPLE_SIZES.values()) + 80
    plan = plan_calls(len(queries), sum(q.source == "open" for q in queries), 16)
    assert plan.fits(Budget(monthly_cap=1000, reserve=40))


def test_ledger_enforces_cap(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl", Budget(monthly_cap=3, reserve=1))
    ledger.record("chat", "m")
    ledger.record("chat", "m")
    assert ledger.remaining() == 0
    with pytest.raises(BudgetExceeded):
        ledger.record("chat", "m")
    assert len(ledger.entries()) == 2


def test_ledger_resets_each_month(tmp_path):
    now = [datetime(2026, 9, 30, 23, 59, tzinfo=UTC)]
    ledger = Ledger(tmp_path / "l.jsonl", Budget(monthly_cap=1, reserve=0), now=lambda: now[0])
    ledger.record("chat", "m")
    with pytest.raises(BudgetExceeded):
        ledger.record("chat", "m")
    now[0] = datetime(2026, 10, 1, 0, 1, tzinfo=UTC)
    ledger.record("chat", "m")
    assert ledger.used_this_month() == 1


def test_rate_limiter_spaces_requests_per_key():
    t = [0.0]
    slept = []

    def sleep(d):
        slept.append(d)
        t[0] += d

    limiter = RateLimiter(per_minute=20, safety=1.0, clock=lambda: t[0], sleep=sleep)
    limiter.wait("a")
    limiter.wait("a")
    limiter.wait("b")  # other model has its own budget
    assert slept == [pytest.approx(3.0)]
