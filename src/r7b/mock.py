"""A deterministic stand-in for the Cohere API, so the pipeline runs without a key.

It is a simulation, not a model: each query gets a latent difficulty from cheap
signals in its text (how many numbers a word problem has, which MMLU subject
it is from, the hand-assigned tier of an open-ended prompt) plus hash noise.
Each model then succeeds with probability sigmoid(ability - difficulty), with
correlated luck so that when R7B gets something right, Command A usually does
too. Abilities are set so per-source accuracies land near published numbers
for the two models, and output lengths reflect Command A being wordier.

Everything is a pure function of the request text, so results are identical
across runs and machines, and the cache is optional.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import cached_property
from pathlib import Path

import numpy as np

from r7b.clients import ChatResult, EmbedResult
from r7b.config import CHEAP_MODEL, DATA_DIR, EMBED_DIM, STRONG_MODEL
from r7b.data import Query, load_queries
from r7b.hashing import h01, hnorm
from r7b.prompts import max_tokens, user_message

VALIDATION_PATH = DATA_DIR / "judge_validation.jsonl"

HARD_SUBJECTS = {
    "abstract_algebra", "college_mathematics", "college_physics", "college_chemistry",
    "formal_logic", "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "professional_law", "professional_accounting", "moral_scenarios", "econometrics",
    "college_computer_science", "machine_learning", "elementary_mathematics",
}
EASY_SUBJECTS = {
    "marketing", "high_school_government_and_politics", "sociology", "us_foreign_policy",
    "high_school_psychology", "management", "miscellaneous", "world_religions",
    "high_school_geography", "public_relations", "nutrition", "human_aging",
}

# Latent ability per model and source, on the same scale as difficulty.
ABILITY = {
    CHEAP_MODEL: {"gsm8k": 0.85, "mmlu": 0.9, "nq": -0.75, "open": 0.75},
    STRONG_MODEL: {"gsm8k": 2.9, "mmlu": 1.85, "nq": -0.1, "open": 1.75},
}
# Mean billed output tokens per model and source.
OUTPUT_TOKENS = {
    CHEAP_MODEL: {"gsm8k": 120, "mmlu": 75, "nq": 11, "open": 270},
    STRONG_MODEL: {"gsm8k": 185, "mmlu": 125, "nq": 16, "open": 430},
}
LUCK_CORRELATION = 0.65

MOCK_BANNER = "[simulated response: mock mode, no API call was made]"


def simulator_version() -> str:
    """Changes whenever this file changes, so cached mock responses never go stale."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def estimate_tokens(text: str) -> int:
    return max(1, round(len(re.findall(r"\w+|[^\w\s]", text)) * 1.1))


def difficulty(q: Query) -> float:
    """Mostly driven by what is visible in the text, plus a hidden idiosyncratic part."""
    text = q.text.lower()
    words = len(q.text.split())
    hidden = hnorm(q.id, "diff")
    if q.source == "gsm8k":
        numbers = len(re.findall(r"\d+(?:\.\d+)?", q.text))
        tricky = sum(w in text for w in ("percent", "%", "half", "twice", "fraction", "ratio",
                                         "average", "remaining", "each week", "profit"))
        return (0.38 * (numbers - 4) + 0.018 * (words - 50) + 0.45 * tricky
                + 0.5 * hidden)
    if q.source == "mmlu":
        subject = q.meta.get("subject", "")
        base = 1.1 if subject in HARD_SUBJECTS else -0.8 if subject in EASY_SUBJECTS else 0.0
        statements = 0.6 if re.search(r"\b(i{1,3}|iv)\.", text) or "statement" in text else 0.0
        mathy = 0.5 if re.search(r"[=^√∫]|\d+\s*[x×*/]\s*\d+", text) else 0.0
        return base + statements + mathy + 0.008 * (words - 60) + 0.55 * hidden
    if q.source == "nq":
        rare = 0.6 if re.search(r"\d", q.text) else 0.0
        kind = 0.5 if text.startswith(("when", "how many", "how much", "what year")) else (
            -0.4 if text.startswith("who") else 0.0)
        return rare + kind + 0.07 * (words - 9) + 0.7 * hidden
    tier = q.meta.get("difficulty", 2)
    bump = 0.35 if q.meta.get("category") in {"code", "reasoning"} else 0.0
    return 1.05 * (tier - 2) + bump + 0.4 * hidden


def luck(q: Query, model: str) -> float:
    """Correlated standard-normal luck: shared across models plus a private part."""
    shared = hnorm(q.id, "luck")
    own = hnorm(q.id, model, "luck")
    return LUCK_CORRELATION * shared + math.sqrt(1 - LUCK_CORRELATION**2) * own


def is_correct(q: Query, model: str) -> bool:
    p = sigmoid(2.2 * (ABILITY[model][q.source] - difficulty(q)))
    return phi(luck(q, model)) < p


def open_score(q: Query, model: str) -> int:
    """True 1-10 quality of a simulated open-ended answer."""
    center = 5.5 + 1.9 * (ABILITY[model]["open"] - difficulty(q))
    return int(np.clip(round(center + 1.1 * luck(q, model)), 1, 10))


def output_tokens(q: Query, model: str) -> int:
    mean = OUTPUT_TOKENS[model][q.source]
    scale = 1 + 0.25 * hnorm(q.id, model, "len")
    if q.source == "open":
        scale *= 0.75 + 0.2 * q.meta.get("difficulty", 2) / 2
    return int(np.clip(round(mean * scale), 4, max_tokens(q)))


def _wrong_number(gold: str, q: Query, model: str) -> str:
    g = float(gold)
    options = [g + 1, g - 1, g * 2, g / 2, g + 10, g * 1.5, g - 5]
    pick = options[int(h01(q.id, model, "wrong") * len(options))]
    if pick == g:
        pick = g + 2
    return str(int(pick)) if float(pick).is_integer() else f"{pick:.2f}"


def _model_label(model: str) -> str:
    return "Command R7B" if model == CHEAP_MODEL else "Command A"


class MockBackend:
    name = "mock"

    def __init__(self, queries: list[Query] | None = None):
        self._queries = queries

    @cached_property
    def queries(self) -> list[Query]:
        return self._queries if self._queries is not None else load_queries()

    @cached_property
    def by_message(self) -> dict[str, Query]:
        return {user_message(q): q for q in self.queries}

    @cached_property
    def by_request(self) -> dict[str, Query]:
        return {q.text: q for q in self.queries}

    @cached_property
    def nq_aliases(self) -> list[str]:
        return [q.gold[0] for q in self.queries if q.source == "nq" and q.gold]

    @cached_property
    def validation_scores(self) -> dict[str, int]:
        scores: dict[str, int] = {}
        if VALIDATION_PATH.exists():
            for line in VALIDATION_PATH.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    scores[rec["response_1"].strip()] = rec["human_1"]
                    scores[rec["response_2"].strip()] = rec["human_2"]
        return scores

    def respond(self, q: Query, model: str) -> str:
        label = _model_label(model)
        correct = is_correct(q, model)
        if q.source == "gsm8k":
            ans = q.gold if correct else _wrong_number(q.gold, q, model)
            nums = re.findall(r"\d+(?:\.\d+)?", q.text)[:6]
            steps = "\n".join(f"- Use {n} from the problem." for n in nums)
            return f"{MOCK_BANNER}\n{label} works through the quantities:\n{steps}\n\nAnswer: {ans}"
        if q.source == "mmlu":
            letters = "ABCD"
            if correct:
                ans = q.gold
            else:
                others = [c for c in letters if c != q.gold]
                ans = others[int(h01(q.id, model, "wrong") * 3)]
            subject = q.meta.get("subject", "").replace("_", " ")
            return (f"{MOCK_BANNER}\n{label} weighs the options for this {subject} question "
                    f"and picks {ans}.\n\nAnswer: {ans}")
        if q.source == "nq":
            if correct:
                ans = q.gold[0]
            else:
                pool = [a for a in self.nq_aliases if a not in q.gold]
                ans = pool[int(h01(q.id, model, "wrong") * len(pool))]
            return f"{MOCK_BANNER}\nAnswer: {ans}"
        score = open_score(q, model)
        return (f"{MOCK_BANNER}\nThis stands in for {label}'s answer to a "
                f"{q.meta.get('category', 'general')} request. In the simulation it earns "
                f"about {score}/10 on the rubric, and its billed length is drawn from what "
                f"{label} typically writes for prompts like this.")

    def chat(self, model: str, message: str, max_tokens: int) -> ChatResult:
        if message.startswith("You are grading two anonymous AI responses"):
            return self._judge(message)
        q = self.by_message.get(message)
        if q is None:
            raise KeyError("mock backend only answers queries from data/queries.jsonl")
        text = self.respond(q, model)
        return ChatResult(text=text, input_tokens=estimate_tokens(message),
                          output_tokens=output_tokens(q, model), finish_reason="COMPLETE")

    def _true_score(self, request: str, response: str) -> int:
        response = response.strip()
        if response in self.validation_scores:
            return self.validation_scores[response]
        q = self.by_request.get(request.strip())
        if q is not None:
            for model in (CHEAP_MODEL, STRONG_MODEL):
                if self.respond(q, model).strip() == response:
                    return open_score(q, model)
        return 5

    def _judge(self, message: str) -> ChatResult:
        def section(tag: str) -> str:
            m = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", message, re.DOTALL)
            return m.group(1) if m else ""

        request = section("request")
        scores = []
        for tag in ("response_1", "response_2"):
            response = section(tag)
            true = self._true_score(request, response)
            noise = round(0.7 * hnorm(response, "judge"))
            scores.append(int(np.clip(true + noise, 1, 10)))
        body = {
            "response_1": {"score": scores[0], "reason": "simulated judge score"},
            "response_2": {"score": scores[1], "reason": "simulated judge score"},
        }
        return ChatResult(text=json.dumps(body), input_tokens=estimate_tokens(message),
                          output_tokens=48, finish_reason="COMPLETE")

    @cached_property
    def lsa(self):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        corpus = [q.text for q in self.queries]
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
            TruncatedSVD(n_components=min(LSA_DIM, len(corpus) - 1), random_state=0),
        )
        return pipe.fit(corpus)

    def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(vectors=[lsa_embedding(self.lsa, t) for t in texts],
                           input_tokens=sum(estimate_tokens(t) for t in texts))


LSA_DIM = 64


def lsa_embedding(lsa, text: str, dim: int = EMBED_DIM) -> list[float]:
    """A topical stand-in for Embed v4: TF-IDF + SVD fitted on the query corpus,
    zero-padded to the real embedding width and L2-normalized. It captures
    subject matter (which is what makes real embeddings useful for routing)
    far better than raw hashed n-grams do on a few hundred training examples."""
    vec = np.zeros(dim)
    z = lsa.transform([text])[0]
    vec[: len(z)] = z
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).round(6).tolist()


def hashed_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Signed feature hashing of unigrams and bigrams, L2-normalized.

    A lexical stand-in for Embed v4: similar wording lands nearby, which is
    enough signal for the router to learn from in mock mode.
    """
    tokens = re.findall(r"[a-z]+|\d+", text.lower())
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    vec = np.zeros(dim)
    for g in grams:
        digest = hashlib.md5(g.encode()).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).round(6).tolist()
