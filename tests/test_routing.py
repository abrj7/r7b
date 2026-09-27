import numpy as np
import pytest

from r7b.routing import (
    Outcomes,
    Point,
    always_cheap,
    always_strong,
    apgr,
    evaluate_routing,
    oracle,
    pareto_frontier,
    pick_threshold,
    quality_at_cost_saved,
    random_at_cost,
    random_line,
    sweep,
    thresholds_for,
)


def outcomes(q_cheap, q_strong, c_cheap=None, c_strong=None, router=None):
    n = len(q_cheap)
    return Outcomes(
        q_cheap=np.array(q_cheap, float),
        q_strong=np.array(q_strong, float),
        c_cheap=np.array(c_cheap if c_cheap is not None else [1.0] * n, float),
        c_strong=np.array(c_strong if c_strong is not None else [10.0] * n, float),
        router_cost=np.array(router if router is not None else [0.0] * n, float),
    )


@pytest.fixture
def four():
    # R7B fails queries 2 and 3; Command A gets everything right.
    return outcomes([1, 1, 0, 0], [1, 1, 1, 1])


def test_always_baselines(four):
    cheap, strong = always_cheap(four), always_strong(four)
    assert cheap.quality == 0.5 and cheap.cost == 4 and cheap.frac_strong == 0
    assert strong.quality == 1.0 and strong.cost == 40 and strong.frac_strong == 1
    assert strong.cost_saved == 0 and strong.quality_retained == 1
    assert cheap.cost_saved == pytest.approx(0.9)
    assert cheap.pgr == 0 and strong.pgr == 1


def test_oracle_is_cheapest_perfect_router(four):
    labels = np.array([1, 1, 0, 0])
    o = oracle(four, labels)
    assert o.quality == 1.0
    assert o.cost == 1 + 1 + 10 + 10
    assert o.cost_saved == pytest.approx(1 - 22 / 40)


def test_evaluate_routing_mixes_per_query(four):
    pt = evaluate_routing(four, np.array([True, False, True, False]))
    assert pt.quality == pytest.approx((1 + 1 + 0 + 1) / 4)
    assert pt.cost == 1 + 10 + 1 + 10
    assert pt.frac_strong == 0.5


def test_router_cost_is_added_unless_excluded():
    out = outcomes([1, 1], [1, 1], router=[0.5, 0.5])
    to_cheap = np.array([True, True])
    assert evaluate_routing(out, to_cheap).cost == 3.0
    assert evaluate_routing(out, to_cheap, include_router_cost=False).cost == 2.0


def test_thresholds_include_extremes():
    ts = thresholds_for(np.array([0.3, 0.7, 0.3]))
    assert ts[0] == 0.0 and ts[-1] > 1.0
    assert list(ts[1:-1]) == [0.3, 0.7]


def test_sweep_extremes_match_baselines(four):
    p = np.array([0.9, 0.8, 0.2, 0.1])
    pts = sweep(four, p)
    assert pts[0].frac_strong == 0  # threshold 0: everything to R7B
    assert pts[-1].frac_strong == 1  # threshold > 1: everything to Command A
    assert pts[0].quality == always_cheap(four).quality
    assert pts[-1].quality == always_strong(four).quality


def test_perfect_router_reaches_oracle_on_sweep(four):
    p = np.array([0.9, 0.8, 0.2, 0.1])
    best = max(sweep(four, p), key=lambda pt: (pt.quality, -pt.cost))
    assert best.quality == 1.0 and best.cost == 22


def test_sweep_cost_is_monotone_in_threshold():
    rng = np.random.default_rng(0)
    out = outcomes(rng.integers(0, 2, 50), rng.integers(0, 2, 50),
                   rng.uniform(0.1, 1, 50), rng.uniform(5, 20, 50))
    costs = [pt.cost for pt in sweep(out, rng.uniform(size=50))]
    assert all(b >= a - 1e-12 for a, b in zip(costs, costs[1:], strict=False))


def _pt(cost, quality):
    return Point(threshold=0, frac_strong=0, cost=cost, quality=quality, cost_saved=0,
                 quality_retained=0, pgr=0)


def test_pareto_frontier_drops_dominated_points():
    pts = [_pt(1, 0.5), _pt(2, 0.4), _pt(3, 0.8), _pt(3, 0.7), _pt(5, 0.8), _pt(6, 0.9)]
    front = pareto_frontier(pts)
    assert [(p.cost, p.quality) for p in front] == [(1, 0.5), (3, 0.8), (6, 0.9)]


def test_random_at_cost_interpolates_linearly(four):
    mid = (4 + 40) / 2
    res = random_at_cost(four, mid, reps=200)
    exp = res["expected"]
    assert exp.frac_strong == pytest.approx(0.5)
    assert exp.quality == pytest.approx(0.75)
    lo, hi = res["quality_ci"]
    assert lo <= 0.75 <= hi


def test_random_at_cost_clamps(four):
    assert random_at_cost(four, -5, reps=1)["expected"].frac_strong == 0
    assert random_at_cost(four, 1e9, reps=1)["expected"].frac_strong == 1


def test_random_line_endpoints(four):
    line = random_line(four, n=5)
    assert line[0].quality == pytest.approx(0.5)
    assert line[-1].quality == pytest.approx(1.0)
    assert len(line) == 5


def test_apgr_perfect_vs_random_vs_inverted():
    q_cheap = np.array([1, 0] * 50, float)
    out = outcomes(q_cheap, np.ones(100))
    perfect = apgr(out, q_cheap)  # confident exactly where R7B is right
    inverted = apgr(out, 1 - q_cheap)
    random_scores = np.random.default_rng(1).uniform(size=100)
    rand = apgr(out, random_scores)
    assert perfect > rand > inverted
    assert 0.35 < rand < 0.65
    assert perfect > 0.7


def test_pick_threshold_respects_target(four):
    p = np.array([0.9, 0.8, 0.2, 0.1])
    pt = pick_threshold(four, p, target_retained=0.99)
    assert pt.quality_retained >= 0.99
    assert pt.cost == 22  # the cheapest way to keep full quality


def test_pick_threshold_falls_back_to_strong(four):
    p = np.array([0.1, 0.2, 0.9, 0.8])  # confidently wrong router
    pt = pick_threshold(four, p, target_retained=1.0)
    assert pt.quality_retained == 1.0


def test_quality_at_cost_saved(four):
    p = np.array([0.9, 0.8, 0.2, 0.1])
    pts = sweep(four, p)
    assert quality_at_cost_saved(pts, 0.45) == 1.0
    assert quality_at_cost_saved(pts, 0.9) == 0.5
    assert np.isnan(quality_at_cost_saved(pts, 0.99))


def test_outcomes_length_check():
    with pytest.raises(ValueError):
        Outcomes(np.ones(3), np.ones(2), np.ones(3), np.ones(3), np.zeros(3))
