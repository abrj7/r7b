"""Stage 1: ask both models every query, score the answers, and label each query."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from r7b.clients import ChatResult, Client
from r7b.config import CHEAP_MODEL, STRONG_MODEL, run_dir
from r7b.costs import chat_cost
from r7b.data import Query
from r7b.grading import grade, judge_to_quality, r7b_good_enough
from r7b.judge import judge_pair
from r7b.prompts import max_tokens, user_message

MODELS = {"cheap": CHEAP_MODEL, "strong": STRONG_MODEL}


def _answer_all(client: Client, queries: list[Query], workers: int) -> dict:
    jobs = [(q, role) for q in queries for role in MODELS]

    def run(job: tuple[Query, str]) -> ChatResult:
        q, role = job
        return client.chat(MODELS[role], user_message(q), max_tokens(q))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    return {(q.id, role): r for (q, role), r in zip(jobs, results, strict=True)}


def collect(client: Client, queries: list[Query], workers: int = 4) -> list[dict]:
    answers = _answer_all(client, queries, workers)

    open_qs = [q for q in queries if q.source == "open"]

    def run_judge(q: Query):
        return judge_pair(client, q.id, q.text, answers[(q.id, "cheap")].text,
                          answers[(q.id, "strong")].text)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = dict(zip([q.id for q in open_qs], pool.map(run_judge, open_qs), strict=True))

    records = []
    for q in queries:
        rec = {"id": q.id, "source": q.source, "text": q.text, "gold": q.gold, "meta": q.meta,
               "responses": {}}
        for role, model in MODELS.items():
            r = answers[(q.id, role)]
            rec["responses"][role] = {
                "model": model,
                "text": r.text,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "finish_reason": r.finish_reason,
                "cost": chat_cost(model, r.input_tokens, r.output_tokens),
            }
        if q.source == "open":
            v = verdicts[q.id]
            rec["responses"]["cheap"]["judge_score"] = v.score_a
            rec["responses"]["strong"]["judge_score"] = v.score_b
            rec["responses"]["cheap"]["quality"] = judge_to_quality(v.score_a)
            rec["responses"]["strong"]["quality"] = judge_to_quality(v.score_b)
            rec["judge"] = {"shown_swapped": v.swapped, "raw": v.raw,
                            "input_tokens": v.input_tokens, "output_tokens": v.output_tokens,
                            "cost": chat_cost(STRONG_MODEL, v.input_tokens, v.output_tokens)}
        else:
            for role in MODELS:
                resp = rec["responses"][role]
                resp["quality"] = grade(q.source, resp["text"], q.gold)
        rec["label"] = int(r7b_good_enough(q.source, rec["responses"]["cheap"]["quality"],
                                           rec["responses"]["strong"]["quality"]))
        records.append(rec)
    return records


def collected_path(mode: str) -> Path:
    return run_dir(mode) / "collected.jsonl"


def save_collected(records: list[dict], mode: str) -> Path:
    path = collected_path(mode)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_collected(mode: str) -> list[dict]:
    path = collected_path(mode)
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
