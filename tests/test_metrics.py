import numpy as np
import pytest

from r7b.metrics import bootstrap_ci, brier, cohen_kappa, rankdata, roc_auc, spearman


def test_rankdata_averages_ties():
    assert list(rankdata([10, 20, 20, 30])) == [1, 2.5, 2.5, 4]


def test_spearman_monotone():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 100]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # observed 0.5, expected 0.5 -> 0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])


def test_auc_matches_definition():
    y = [0, 0, 1, 1]
    assert roc_auc(y, [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc(y, [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc([1, 1], [0.2, 0.3]))


def test_auc_agrees_with_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_brier():
    assert brier([1, 0], [1, 0]) == 0
    assert brier([1, 0], [0.5, 0.5]) == 0.25


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).normal(5, 1, 400)
    lo, hi = bootstrap_ci(lambda idx: float(x[idx].mean()), len(x), reps=500)
    assert lo < x.mean() < hi
    assert hi - lo < 0.5
