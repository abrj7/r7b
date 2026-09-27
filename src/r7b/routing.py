"""Routing decisions, threshold sweeps, baselines, and the cost-quality frontier.

Conventions: `p` is the router's probability that R7B is good enough for a
query. A query goes to R7B when p >= threshold, so threshold 0 sends everything
to R7B and a threshold above 1 sends everything to Command A.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Outcomes:
    """Per-query results for both models on one split."""

    q_cheap: np.ndarray
    q_strong: np.ndarray
    c_cheap: np.ndarray
    c_strong: np.ndarray
    router_cost: np.ndarray

    def __post_init__(self):
        n = len(self.q_cheap)
        for name in ("q_strong", "c_cheap", "c_strong", "router_cost"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has the wrong length")

    @property
    def n(self) -> int:
        return len(self.q_cheap)

    def subset(self, idx: np.ndarray) -> Outcomes:
        return Outcomes(self.q_cheap[idx], self.q_strong[idx], self.c_cheap[idx],
                        self.c_strong[idx], self.router_cost[idx])

    @property
    def strong_quality(self) -> float:
        return float(self.q_strong.mean())

    @property
    def strong_cost(self) -> float:
        return float(self.c_strong.sum())


@dataclass(frozen=True)
class Point:
    threshold: float
    frac_strong: float
    cost: float
    quality: float
    cost_saved: float
    quality_retained: float
    pgr: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_routing(out: Outcomes, to_cheap: np.ndarray, threshold: float = float("nan"),
                     include_router_cost: bool = True) -> Point:
    """Quality and cost of a concrete routing decision (True = send to R7B)."""
    to_cheap = np.asarray(to_cheap, dtype=bool)
    quality = float(np.where(to_cheap, out.q_cheap, out.q_strong).mean())
    cost = float(np.where(to_cheap, out.c_cheap, out.c_strong).sum())
    if include_router_cost:
        cost += float(out.router_cost.sum())
    return _point(out, threshold, float(1 - to_cheap.mean()), cost, quality)


def _point(out: Outcomes, threshold: float, frac_strong: float, cost: float,
           quality: float) -> Point:
    strong_q, strong_c = out.strong_quality, out.strong_cost
    cheap_q = float(out.q_cheap.mean())
    gap = strong_q - cheap_q
    return Point(
        threshold=threshold,
        frac_strong=frac_strong,
        cost=cost,
        quality=quality,
        cost_saved=1 - cost / strong_c if strong_c else 0.0,
        quality_retained=quality / strong_q if strong_q else float("nan"),
        pgr=(quality - cheap_q) / gap if gap > 0 else float("nan"),
    )


def thresholds_for(p: np.ndarray) -> np.ndarray:
    """Every threshold that changes a decision, plus both extremes."""
    return np.unique(np.concatenate([[0.0], np.asarray(p, float), [1.0 + 1e-9]]))


def sweep(out: Outcomes, p: np.ndarray, thresholds: np.ndarray | None = None) -> list[Point]:
    p = np.asarray(p, float)
    ts = thresholds_for(p) if thresholds is None else thresholds
    return [evaluate_routing(out, p >= t, float(t)) for t in ts]


def pareto_frontier(points: list[Point]) -> list[Point]:
    """Points not beaten on both cost (lower) and quality (higher), sorted by cost."""
    ordered = sorted(points, key=lambda pt: (pt.cost, -pt.quality))
    frontier: list[Point] = []
    best = -np.inf
    for pt in ordered:
        if pt.quality > best + 1e-12:
            frontier.append(pt)
            best = pt.quality
    return frontier


def always_cheap(out: Outcomes) -> Point:
    return evaluate_routing(out, np.ones(out.n, bool), include_router_cost=False)


def always_strong(out: Outcomes) -> Point:
    return evaluate_routing(out, np.zeros(out.n, bool), include_router_cost=False)


def oracle(out: Outcomes, labels: np.ndarray) -> Point:
    """Perfect knowledge of the label: R7B exactly when it is good enough."""
    return evaluate_routing(out, np.asarray(labels, bool), include_router_cost=False)


def random_at_cost(out: Outcomes, target_cost: float, reps: int = 2000,
                   seed: int = 0) -> dict:
    """Send a random fraction f of queries to Command A, with f set so the
    expected cost equals `target_cost`. Returns the exact expectation plus a
    spread from simulated draws of the same fraction."""
    lo, hi = float(out.c_cheap.sum()), float(out.c_strong.sum())
    f = 0.0 if hi == lo else float(np.clip((target_cost - lo) / (hi - lo), 0, 1))
    q_exp = (1 - f) * out.q_cheap.mean() + f * out.q_strong.mean()
    cost_exp = (1 - f) * lo + f * hi
    expected = _point(out, float("nan"), f, float(cost_exp), float(q_exp))
    rng = np.random.default_rng(seed)
    k = round(f * out.n)
    draws = []
    for _ in range(reps):
        to_strong = np.zeros(out.n, bool)
        to_strong[rng.choice(out.n, size=k, replace=False)] = True
        draws.append(evaluate_routing(out, ~to_strong, include_router_cost=False).quality)
    lo_q, hi_q = np.quantile(draws, [0.025, 0.975])
    return {"expected": expected, "quality_ci": (float(lo_q), float(hi_q))}


def random_line(out: Outcomes, n: int = 11) -> list[Point]:
    """Expected cost and quality of random routing, from all-R7B to all-Command A."""
    lo, hi = float(out.c_cheap.sum()), float(out.c_strong.sum())
    return [random_at_cost(out, lo + f * (hi - lo), reps=1)["expected"]
            for f in np.linspace(0, 1, n)]


def quality_at_cost_saved(points: list[Point], saved: float) -> float:
    """Best quality achievable with at least `saved` fraction of cost saved."""
    ok = [pt.quality for pt in points if pt.cost_saved >= saved - 1e-12]
    return max(ok) if ok else float("nan")


def apgr(out: Outcomes, p: np.ndarray, grid: int = 21) -> float:
    """Average performance gap recovered over evenly spaced Command A call rates.

    For each fraction of traffic sent to Command A (0%..100%), route the
    queries the router is least confident about, and measure how much of the
    R7B-to-Command-A quality gap is closed. Random routing scores 0.5.
    """
    order = np.argsort(np.asarray(p, float), kind="mergesort")
    pgrs = []
    for f in np.linspace(0, 1, grid):
        k = round(f * out.n)
        to_cheap = np.ones(out.n, bool)
        to_cheap[order[:k]] = False
        pgrs.append(evaluate_routing(out, to_cheap, include_router_cost=False).pgr)
    return float(np.trapezoid(pgrs, dx=1 / (grid - 1)))


def pick_threshold(out: Outcomes, p: np.ndarray, target_retained: float) -> Point:
    """Cheapest threshold on this split that keeps quality >= target share of Command A's."""
    pts = [pt for pt in sweep(out, p) if pt.quality_retained >= target_retained - 1e-12]
    if not pts:
        return evaluate_routing(out, np.zeros(out.n, bool), threshold=1.0 + 1e-9)
    return min(pts, key=lambda pt: (pt.cost, -pt.quality))
