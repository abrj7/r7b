"""Command A as a judge for open-ended answers, and its validation against hand labels."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from r7b.clients import Client
from r7b.config import DATA_DIR, JUDGE_MODEL
from r7b.grading import OPEN_TOLERANCE, judge_to_quality, parse_judge
from r7b.hashing import h01
from r7b.metrics import cohen_kappa, spearman
from r7b.prompts import judge_message

JUDGE_MAX_TOKENS = 200
VALIDATION_PATH = DATA_DIR / "judge_validation.jsonl"


@dataclass
class JudgeVerdict:
    score_a: int
    score_b: int
    swapped: bool
    raw: str
    input_tokens: int
    output_tokens: int


def judge_pair(client: Client, key: str, request: str, answer_a: str,
               answer_b: str) -> JudgeVerdict:
    """Score two answers in one call. Which one is shown first is decided by a
    hash of `key`, so position bias cannot systematically favor either model."""
    swapped = h01(key, "judge-order") < 0.5
    first, second = (answer_b, answer_a) if swapped else (answer_a, answer_b)
    result = client.chat(JUDGE_MODEL, judge_message(request, first, second), JUDGE_MAX_TOKENS)
    s1, s2 = parse_judge(result.text)
    a, b = (s2, s1) if swapped else (s1, s2)
    return JudgeVerdict(a, b, swapped, result.text, result.input_tokens, result.output_tokens)


def _preference(a: float, b: float, tie: float) -> int:
    if abs(a - b) <= tie:
        return 0
    return 1 if a > b else -1


def load_validation() -> list[dict]:
    return [json.loads(ln) for ln in VALIDATION_PATH.read_text().splitlines() if ln.strip()]


def validate_judge(client: Client, prompts: dict[str, str]) -> dict:
    """Run the judge on hand-labeled pairs and measure agreement with the human scores.

    The number that matters most for routing is `routing_label_agreement`: does
    the judge reach the same "is answer 1 within a point of answer 2" verdict
    as the human, which is exactly how open-ended labels are built.
    """
    rows = load_validation()
    human, judged, items = [], [], []
    pref_h, pref_j, label_h, label_j = [], [], [], []
    for row in rows:
        v = judge_pair(client, row["id"], prompts[row["prompt_id"]],
                       row["response_1"], row["response_2"])
        h1, h2 = row["human_1"], row["human_2"]
        human += [h1, h2]
        judged += [v.score_a, v.score_b]
        pref_h.append(_preference(h1, h2, 0.5))
        pref_j.append(_preference(v.score_a, v.score_b, 0.5))
        qh1, qh2 = judge_to_quality(h1), judge_to_quality(h2)
        qj1, qj2 = judge_to_quality(v.score_a), judge_to_quality(v.score_b)
        label_h.append(qh1 >= qh2 - OPEN_TOLERANCE - 1e-9)
        label_j.append(qj1 >= qj2 - OPEN_TOLERANCE - 1e-9)
        items.append({"id": row["id"], "prompt_id": row["prompt_id"], "human": [h1, h2],
                      "judge": [v.score_a, v.score_b], "shown_swapped": v.swapped,
                      "note": row.get("note", "")})
    human_a, judged_a = np.array(human, float), np.array(judged, float)
    return {
        "n_pairs": len(rows),
        "n_scores": len(human),
        "mae": float(np.mean(np.abs(human_a - judged_a))),
        "within_1": float(np.mean(np.abs(human_a - judged_a) <= 1)),
        "spearman": spearman(human_a, judged_a),
        "preference_agreement": float(np.mean(np.array(pref_h) == np.array(pref_j))),
        "preference_kappa": cohen_kappa(pref_h, pref_j),
        "routing_label_agreement": float(np.mean(np.array(label_h) == np.array(label_j))),
        "items": items,
    }
