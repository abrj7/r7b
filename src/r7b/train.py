"""Stage 2: train routers, pick one on validation, and evaluate it once on test."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from r7b.config import SEED
from r7b.features import FEATURE_NAMES, FEATURE_SETS, feature_set
from r7b.metrics import bootstrap_ci, brier, roc_auc
from r7b.routing import (
    Outcomes,
    always_cheap,
    always_strong,
    apgr,
    evaluate_routing,
    oracle,
    pareto_frontier,
    pick_threshold,
    random_at_cost,
    random_line,
    sweep,
)

TARGETS = (0.90, 0.95, 0.98)
HEADLINE_TARGET = 0.95


def _logreg():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000))])


def _gbdt():
    return HistGradientBoostingClassifier(random_state=SEED)


def _mlp():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", MLPClassifier(max_iter=2000, random_state=SEED))])


MODELS = {
    "logreg": (_logreg, {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}),
    "gbdt": (_gbdt, {"learning_rate": [0.03, 0.1], "max_depth": [2, 3],
                     "max_iter": [100, 300], "l2_regularization": [0.0, 1.0]}),
    "mlp": (_mlp, {"clf__hidden_layer_sizes": [(32,), (64,)],
                   "clf__alpha": [0.01, 0.1, 1.0, 3.0]}),
}


@dataclass
class Dataset:
    ids: list[str]
    sources: np.ndarray
    labels: np.ndarray
    hand: np.ndarray
    emb: np.ndarray
    outcomes: Outcomes


def build_dataset(records: list[dict], emb: np.ndarray, router_cost: np.ndarray,
                  hand: np.ndarray) -> Dataset:
    r = [rec["responses"] for rec in records]
    return Dataset(
        ids=[rec["id"] for rec in records],
        sources=np.array([rec["source"] for rec in records]),
        labels=np.array([rec["label"] for rec in records], dtype=int),
        hand=hand,
        emb=emb,
        outcomes=Outcomes(
            q_cheap=np.array([x["cheap"]["quality"] for x in r], float),
            q_strong=np.array([x["strong"]["quality"] for x in r], float),
            c_cheap=np.array([x["cheap"]["cost"] for x in r], float),
            c_strong=np.array([x["strong"]["cost"] for x in r], float),
            router_cost=np.asarray(router_cost, float),
        ),
    )


def split_indices(ds: Dataset, val: float = 0.2, test: float = 0.2) -> dict[str, np.ndarray]:
    """60/20/20, stratified on source x label so every split has the same mix."""
    idx = np.arange(len(ds.ids))
    strata = np.char.add(ds.sources.astype(str), ds.labels.astype(str))
    train_val, test_idx = train_test_split(idx, test_size=test, stratify=strata,
                                           random_state=SEED)
    train_idx, val_idx = train_test_split(train_val, test_size=val / (1 - test),
                                          stratify=strata[train_val], random_state=SEED)
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def fit_one(name: str, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
    make, grid = MODELS[name]
    search = GridSearchCV(make(), grid, cv=_cv(), scoring="roc_auc", n_jobs=-1, refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(X, y)
    return search


def _pts(points):
    return [p.to_dict() for p in points]


def _thin(points, keep: int = 120):
    if len(points) <= keep:
        return points
    idx = np.unique(np.linspace(0, len(points) - 1, keep).round().astype(int))
    return [points[i] for i in idx]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fn(*args, **kwargs)


def train_and_evaluate(ds: Dataset) -> dict:
    """Protocol, with test touched exactly once per model at the very end:

    1. Hyperparameters: 5-fold CV grid search on train.
    2. Model and feature set: best mean CV AUC on train; val AUC/APGR reported
       for the train-fit model as a second opinion.
    3. Threshold: chosen on out-of-fold predictions over train+val (304 queries,
       far steadier than 76 val queries alone) as the cheapest threshold that
       keeps the target share of Command A quality.
    4. Final router: chosen config refit on train+val, scored on test.
    """
    splits = split_indices(ds)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    dev = np.sort(np.concatenate([tr, va]))
    y = ds.labels
    out_val, out_dev, out_test = (ds.outcomes.subset(i) for i in (va, dev, te))

    runs = []
    for fs in FEATURE_SETS:
        X = feature_set(fs, ds.hand, ds.emb)
        for name in MODELS:
            search = fit_one(name, X[tr], y[tr])
            best = search.best_index_
            p_val = search.predict_proba(X[va])[:, 1]
            final = _quiet(clone(search.best_estimator_).fit, X[dev], y[dev])
            p_test = final.predict_proba(X[te])[:, 1]
            runs.append({
                "model": name,
                "features": fs,
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in search.best_params_.items()},
                "cv_auc": float(search.cv_results_["mean_test_score"][best]),
                "cv_auc_std": float(search.cv_results_["std_test_score"][best]),
                "val_auc": roc_auc(y[va], p_val),
                "val_apgr": apgr(out_val, p_val),
                "test_auc": roc_auc(y[te], p_test),
                "test_apgr": apgr(out_test, p_test),
                "test_brier": brier(y[te], p_test),
                "_estimator": search.best_estimator_,
                "_p_test": p_test,
            })

    chosen = max(runs, key=lambda r: (r["cv_auc"], r["val_auc"]))
    X_chosen = feature_set(chosen["features"], ds.hand, ds.emb)
    p_dev = _quiet(cross_val_predict, clone(chosen["_estimator"]), X_chosen[dev], y[dev],
                   cv=_cv(), method="predict_proba")[:, 1]
    p_test = chosen["_p_test"]

    operating_points = []
    for target in TARGETS:
        on_dev = pick_threshold(out_dev, p_dev, target)
        t = on_dev.threshold
        on_test = evaluate_routing(out_test, p_test >= t, t)
        rand = random_at_cost(out_test, on_test.cost)
        operating_points.append({
            "target_retained": target,
            "threshold": t,
            "dev_oof": on_dev.to_dict(),
            "test": on_test.to_dict(),
            "random_at_same_cost": rand["expected"].to_dict(),
            "random_quality_ci": rand["quality_ci"],
            "test_ci": _bootstrap_point(out_test, p_test, t),
        })

    headline = next(op for op in operating_points if op["target_retained"] == HEADLINE_TARGET)

    p_oof = _quiet(cross_val_predict, clone(chosen["_estimator"]), X_chosen, y, cv=_cv(),
                   method="predict_proba")[:, 1]

    curves = {}
    for r in runs:
        if r["features"] != chosen["features"] and r["model"] != chosen["model"]:
            continue
        pts = sweep(out_test, r["_p_test"])
        curves[f"{r['model']}:{r['features']}"] = {
            "model": r["model"], "features": r["features"],
            "points": _pts(_thin(pts)), "frontier": _pts(pareto_frontier(pts)),
        }

    baselines = {
        "always_r7b": always_cheap(out_test).to_dict(),
        "always_command_a": always_strong(out_test).to_dict(),
        "oracle": oracle(out_test, y[te]).to_dict(),
        "random_line": _pts(random_line(out_test)),
    }

    return {
        "splits": {k: [ds.ids[i] for i in v] for k, v in splits.items()},
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "label_rate": {k: float(y[v].mean()) for k, v in splits.items()},
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
        "chosen": {"model": chosen["model"], "features": chosen["features"],
                   "params": chosen["params"]},
        "operating_points": operating_points,
        "headline": headline,
        "baselines": baselines,
        "curves": curves,
        "p_test": {ds.ids[i]: float(p) for i, p in zip(te, p_test, strict=True)},
        "p_oof": {ds.ids[i]: float(p) for i, p in enumerate(p_oof)},
        "browser_router": browser_router(ds, splits),
    }


def _bootstrap_point(out: Outcomes, p: np.ndarray, t: float) -> dict:
    def stat(field):
        def f(idx):
            sub = out.subset(idx)
            return getattr(evaluate_routing(sub, p[idx] >= t), field)
        return f

    return {
        "cost_saved": bootstrap_ci(stat("cost_saved"), out.n, reps=1000, seed=SEED),
        "quality_retained": bootstrap_ci(stat("quality_retained"), out.n, reps=1000, seed=SEED),
    }


def browser_router(ds: Dataset, splits: dict[str, np.ndarray]) -> dict:
    """A hand-crafted-features-only logistic regression small enough to run in the
    browser on text the visitor types, where no embedding call is possible."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    dev = np.sort(np.concatenate([tr, va]))
    y = ds.labels
    search = fit_one("logreg", ds.hand[tr], y[tr])
    p_dev = cross_val_predict(clone(search.best_estimator_), ds.hand[dev], y[dev], cv=_cv(),
                              method="predict_proba")[:, 1]
    op = pick_threshold(ds.outcomes.subset(dev), p_dev, HEADLINE_TARGET)
    pipe = clone(search.best_estimator_).fit(ds.hand[dev], y[dev])
    scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    return {
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold": op.threshold,
        "cv_auc": float(search.best_score_),
        "test_auc": roc_auc(y[te], pipe.predict_proba(ds.hand[te])[:, 1]),
        "C": float(search.best_params_["clf__C"]),
    }
