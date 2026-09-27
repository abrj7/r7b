"""Small, dependency-light statistics used across the pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def rankdata(x: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based), ties share the mean of their positions."""
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label sequences")
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(y_true: Sequence[int], p: Sequence[float]) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y_true, float)) ** 2))


def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, reps: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile interval for `stat(indices)` over bootstrap resamples of range(n)."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(reps)])
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
