"""Stage 3: write the JSON the static site reads from site/public/data/."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from r7b.budget import CallPlan
from r7b.config import (
    CHEAP_MODEL,
    EMBED_DIM,
    EMBED_MODEL,
    JUDGE_MODEL,
    SITE_DATA_DIR,
    STRONG_MODEL,
    Budget,
)
from r7b.costs import pricing_table


def _per_source(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["source"]].append(rec)
    groups["all"] = records
    out = {}
    for source, recs in groups.items():
        row = {"n": len(recs), "r7b_good_enough": float(np.mean([r["label"] for r in recs]))}
        for role in ("cheap", "strong"):
            rs = [r["responses"][role] for r in recs]
            row[role] = {
                "quality": float(np.mean([x["quality"] for x in rs])),
                "input_tokens": float(np.mean([x["input_tokens"] for x in rs])),
                "output_tokens": float(np.mean([x["output_tokens"] for x in rs])),
                "cost_per_1k_queries": float(np.mean([x["cost"] for x in rs]) * 1000),
            }
        out[source] = row
    return out


def _experiment_cost(records: list[dict], router_cost: np.ndarray, validation: dict) -> dict:
    answers = sum(r["responses"][role]["cost"] for r in records for role in ("cheap", "strong"))
    judge = sum(r.get("judge", {}).get("cost", 0.0) for r in records)
    return {
        "answers_usd": float(answers),
        "judge_usd": float(judge),
        "embed_usd": float(router_cost.sum()),
        "total_usd": float(answers + judge + router_cost.sum()),
        "note": "what the experiment would cost on a production key; trial calls are free",
        "judge_validation_pairs": validation["n_pairs"],
    }


def build_queries_payload(records: list[dict], results: dict,
                          router_cost: np.ndarray) -> list[dict]:
    """One row per query. `p` is always an out-of-sample confidence: the final
    router's prediction for test queries, and a 5-fold out-of-fold prediction
    for train and val queries."""
    split_of = {qid: name for name, ids in results["splits"].items() for qid in ids}
    threshold = results["headline"]["threshold"]
    rows = []
    for i, rec in enumerate(records):
        held_out = results["p_test"].get(rec["id"])
        p = held_out if held_out is not None else results["p_oof"][rec["id"]]
        row = {
            "id": rec["id"],
            "source": rec["source"],
            "split": split_of[rec["id"]],
            "text": rec["text"],
            "gold": rec["gold"],
            "meta": rec["meta"],
            "label": rec["label"],
            "p": p,
            "route": "r7b" if p >= threshold else "command-a",
            "router_cost": float(router_cost[i]),
        }
        for role, key in (("cheap", "r7b"), ("strong", "command_a")):
            r = rec["responses"][role]
            row[key] = {
                "text": r["text"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "cost": r["cost"],
                "quality": r["quality"],
                "judge_score": r.get("judge_score"),
            }
        rows.append(row)
    return rows


def export_site(mode: str, records: list[dict], results: dict, validation: dict,
                plan: CallPlan, budget: Budget, router_cost: np.ndarray,
                calls_used: int | None, out_dir: Path = SITE_DATA_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": {"cheap": CHEAP_MODEL, "strong": STRONG_MODEL, "judge": JUDGE_MODEL,
                   "embed": EMBED_MODEL, "embed_dim": EMBED_DIM},
        "pricing": pricing_table(),
        "budget": {**plan.to_dict(budget), "calls_used_this_month": calls_used},
        "experiment_cost": _experiment_cost(records, router_cost, validation),
        "per_source": _per_source(records),
        "split_sizes": results["split_sizes"],
        "label_rate": results["label_rate"],
        "runs": results["runs"],
        "chosen": results["chosen"],
        "headline": results["headline"],
        "operating_points": results["operating_points"],
        "baselines": {k: v for k, v in results["baselines"].items() if k != "random_line"},
        "judge_validation": validation,
    }
    curves = {
        "curves": results["curves"],
        "random_line": results["baselines"]["random_line"],
        "baselines": {k: v for k, v in results["baselines"].items() if k != "random_line"},
        "chosen": f"{results['chosen']['model']}:{results['chosen']['features']}",
    }
    payloads = {
        "summary.json": summary,
        "curves.json": curves,
        "queries.json": build_queries_payload(records, results, router_cost),
        "browser_router.json": results["browser_router"],
    }
    paths = []
    for name, data in payloads.items():
        path = out_dir / name
        path.write_text(json.dumps(clean(data), ensure_ascii=False, allow_nan=False))
        paths.append(path)
    return paths


def clean(obj):
    """JSON-safe copy: numpy scalars to Python, tuples to lists, NaN/inf to null
    (browsers reject bare NaN in JSON)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj
