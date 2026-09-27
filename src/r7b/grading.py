"""Turn raw model responses into quality scores in [0, 1]."""

from __future__ import annotations

import json
import re
import string

ANSWER_LINE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
ARTICLES = re.compile(r"\b(a|an|the)\b")

# For open-ended queries: R7B is "good enough" if its judge score is within
# this much of Command A's (on the 0-1 scale, 1/9 is one rubric point).
OPEN_TOLERANCE = 1 / 9


def final_answer(text: str) -> str:
    """The text after the last `Answer:` marker, or the last non-empty line."""
    matches = ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].strip().strip("*`_ ").strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _to_number(s: str) -> float | None:
    found = NUMBER.findall(s)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def grade_gsm8k(response: str, gold: str) -> float:
    pred = _to_number(final_answer(response))
    if pred is None:
        pred = _to_number(response)
    if pred is None:
        return 0.0
    return float(abs(pred - float(gold)) < 1e-6)


def extract_choice(response: str) -> str | None:
    ans = final_answer(response).upper()
    m = re.search(r"\b([ABCD])\b", ans) or re.match(r"\(?([ABCD])[\).:]?", ans)
    if m:
        return m.group(1)
    tail = re.findall(r"\b([ABCD])\b", response.upper())
    return tail[-1] if tail else None


def grade_mmlu(response: str, gold: str) -> float:
    return float(extract_choice(response) == gold)


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def grade_nq(response: str, gold: list[str]) -> float:
    """Correct if any gold alias appears as a whole-token span of the final answer."""
    pred = f" {normalize(final_answer(response))} "
    for alias in gold:
        a = normalize(alias)
        if a and f" {a} " in pred:
            return 1.0
    return 0.0


def grade(source: str, response: str, gold) -> float:
    if source == "gsm8k":
        return grade_gsm8k(response, gold)
    if source == "mmlu":
        return grade_mmlu(response, gold)
    if source == "nq":
        return grade_nq(response, gold)
    raise ValueError(f"{source} has no ground truth; use the judge")


def parse_judge(text: str) -> tuple[int, int]:
    """Pull the two 1-10 scores out of the judge's JSON, tolerating code fences."""
    body = text.strip()
    fence = re.search(r"\{.*\}", body, re.DOTALL)
    if fence:
        body = fence.group(0)
    try:
        data = json.loads(body)
        s1 = int(data["response_1"]["score"])
        s2 = int(data["response_2"]["score"])
    except (ValueError, KeyError, TypeError) as exc:
        nums = re.findall(r'"score"\s*:\s*(\d+)', text)
        if len(nums) < 2:
            raise ValueError(f"could not parse judge output: {text[:200]!r}") from exc
        s1, s2 = int(nums[0]), int(nums[1])
    return _clamp(s1), _clamp(s2)


def _clamp(score: int) -> int:
    return max(1, min(10, score))


def judge_to_quality(score: int) -> float:
    return (score - 1) / 9


def r7b_good_enough(source: str, q_cheap: float, q_strong: float) -> bool:
    """The routing label: does sending this query to R7B lose nothing that matters?

    Ground-truth tasks: R7B is at least as right as Command A (so both-wrong
    counts as good enough, since escalating would not have helped).
    Open-ended: R7B's judge score is within one rubric point of Command A's.
    """
    tol = OPEN_TOLERANCE if source == "open" else 0.0
    return q_cheap >= q_strong - tol - 1e-9
