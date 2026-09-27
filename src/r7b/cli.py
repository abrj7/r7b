"""Command-line entry point: `uv run r7b --help`."""

from __future__ import annotations

import argparse
import json
import sys
import time

from r7b.budget import Ledger, plan_calls
from r7b.config import api_key, budget_from_env, cache_dir, run_dir


def _mode(args) -> str:
    if args.mock:
        return "mock"
    if args.live:
        return "live"
    if api_key():
        return "live"
    print("no COHERE_API_KEY in .env, running in mock mode (pass --live to insist)",
          file=sys.stderr)
    return "mock"


def _plan():
    from r7b.data import load_queries
    from r7b.judge import load_validation

    queries = load_queries()
    n_open = sum(q.source == "open" for q in queries)
    return queries, plan_calls(len(queries), n_open, len(load_validation()))


def cmd_data(args) -> None:
    from collections import Counter

    from r7b.data import build_query_set

    queries = build_query_set()
    print(f"wrote {len(queries)} queries: {dict(Counter(q.source for q in queries))}")


def cmd_budget(args) -> None:
    budget = budget_from_env()
    _, plan = _plan()
    report = plan.to_dict(budget)
    ledger = Ledger(cache_dir("live") / "ledger.jsonl", budget)
    report["live_calls_used_this_month"] = ledger.used_this_month()
    print(json.dumps(report, indent=2))
    if not plan.fits(budget):
        sys.exit("plan does not fit the monthly budget; shrink SAMPLE_SIZES in r7b/data.py")


def cmd_run(args) -> None:
    import numpy as np

    from r7b.clients import make_client
    from r7b.collect import collect, save_collected
    from r7b.export import clean, export_site
    from r7b.features import embed_matrix, handcrafted_matrix
    from r7b.judge import validate_judge
    from r7b.train import build_dataset, train_and_evaluate

    mode = _mode(args)
    budget = budget_from_env()
    queries, plan = _plan()
    if mode == "live" and not plan.fits(budget):
        sys.exit(f"plan needs {plan.total} calls but only {budget.usable} are usable")
    client = make_client(mode)
    t0 = time.time()

    print(f"[{mode}] collecting answers from both models for {len(queries)} queries")
    records = collect(client, queries, workers=args.workers)
    save_collected(records, mode)

    print(f"[{mode}] validating the judge on hand-labeled pairs")
    prompts = {q.id: q.text for q in queries if q.source == "open"}
    validation = validate_judge(client, prompts)

    print(f"[{mode}] embedding queries with Embed v4 and training routers")
    texts = [r["text"] for r in records]
    emb, router_cost = embed_matrix(client, texts)
    hand = handcrafted_matrix(texts)
    ds = build_dataset(records, emb, router_cost, hand)
    results = train_and_evaluate(ds)

    out = run_dir(mode)
    (out / "results.json").write_text(json.dumps(clean(
        {k: v for k, v in results.items() if k not in ("p_oof", "p_test")}), indent=1))
    (out / "judge_validation.json").write_text(json.dumps(clean(validation), indent=1))
    np.save(out / "embeddings.npy", emb.astype(np.float32))

    calls_used = None
    if mode == "live":
        calls_used = Ledger(cache_dir("live") / "ledger.jsonl", budget).used_this_month()
    paths = export_site(mode, records, results, validation, plan, budget, router_cost,
                        calls_used)
    h = results["headline"]["test"]
    print(f"[{mode}] done in {time.time() - t0:.1f}s; client stats {client.stats}")
    print(f"headline (test; threshold picked on train+val out-of-fold predictions for "
          f">=95% of Command A quality): "
          f"{h['cost_saved']:.1%} cost saved at {h['quality_retained']:.1%} of Command A "
          f"quality, {h['frac_strong']:.0%} of queries escalated")
    for p in paths:
        print(f"wrote {p}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="r7b", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("data", help="rebuild data/queries.jsonl from the source datasets")
    sub.add_parser("budget", help="show the API call plan against the monthly cap")
    run = sub.add_parser("run", help="collect, judge, embed, train, evaluate, export")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="deterministic offline mode")
    group.add_argument("--live", action="store_true", help="call Cohere (needs a key)")
    run.add_argument("--workers", type=int, default=4)

    args = parser.parse_args(argv)
    {"data": cmd_data, "budget": cmd_budget, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
