"""Prompt templates. Both models see exactly the same message for a query."""

from __future__ import annotations

from r7b.data import Query

TASK_INSTRUCTIONS = {
    "gsm8k": (
        "Solve the math word problem. Reason briefly, then give the final numeric answer "
        "on its own last line in the form `Answer: <number>`."
    ),
    "mmlu": (
        "Answer the multiple-choice question. Reason briefly, then give the letter of the "
        "correct option on its own last line in the form `Answer: <letter>`."
    ),
    "nq": (
        "Answer the question with a short factual phrase (a name, date, number, or a few "
        "words). Put it on its own last line in the form `Answer: <answer>`."
    ),
}

MAX_TOKENS = {"gsm8k": 400, "mmlu": 300, "nq": 80, "open": 600}


def user_message(q: Query) -> str:
    instruction = TASK_INSTRUCTIONS.get(q.source)
    if instruction is None:
        return q.text
    return f"{instruction}\n\n{q.text}"


def max_tokens(q: Query) -> int:
    return MAX_TOKENS[q.source]


JUDGE_RUBRIC = """\
Score each response from 1 to 10 against this rubric.

- Correctness: facts, math, and code are right. A single material error caps the score at 5.
- Instruction following: every explicit constraint in the request is met (length, format,
  word or letter restrictions, language). Missing a stated constraint caps the score at 6.
- Completeness: the request is fully addressed, including every sub-question.
- Clarity: well organized and no longer than it needs to be.

Anchors:
10 = fully correct, meets every constraint, nothing an expert would add.
8  = correct and useful, minor omissions or slightly verbose.
6  = mostly right but misses a constraint or a meaningful part of the request.
4  = partially correct, contains a material error, or is badly incomplete.
2  = mostly wrong or unhelpful.
1  = wrong, off-topic, or refuses.

Judge each response on its own merits against the request, not against the other response.
Do not reward length for its own sake."""

JUDGE_TEMPLATE = """\
You are grading two anonymous AI responses to the same request.

{rubric}

<request>
{request}
</request>

<response_1>
{response_1}
</response_1>

<response_2>
{response_2}
</response_2>

Return only JSON, no prose, in exactly this shape:
{{"response_1": {{"score": <int 1-10>, "reason": "<one sentence>"}},
 "response_2": {{"score": <int 1-10>, "reason": "<one sentence>"}}}}"""


def judge_message(request: str, response_1: str, response_2: str) -> str:
    return JUDGE_TEMPLATE.format(
        rubric=JUDGE_RUBRIC, request=request, response_1=response_1, response_2=response_2
    )
