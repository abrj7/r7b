"""Router features: Embed v4 vectors plus cheap hand-crafted text statistics.

The hand-crafted features are deliberately simple regex counts so the site can
recompute them in the browser (site/src/lib/features.ts mirrors this file and
tests/test_features.py pins shared examples).
"""

from __future__ import annotations

import math
import re

import numpy as np

from r7b.clients import Client
from r7b.config import EMBED_MODEL
from r7b.costs import embed_cost

REASONING_WORDS = {"why", "explain", "prove", "derive", "compare", "analyze", "analyse",
                   "calculate", "estimate", "design", "implement", "critique", "justify"}
CONSTRAINT_WORDS = {"exactly", "only", "must", "without", "under", "least", "most", "words",
                    "sentences", "lines", "rhyme", "syllable", "letter", "format", "formal"}
SIMPLE_WORDS = {"what", "who", "when", "where", "name", "list", "which"}
WH_WORDS = {"who", "what", "when", "where", "which", "how", "why"}
CODE_CHARS = set("{}[]();=<>_")

WORD = re.compile(r"[A-Za-z']+")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
CHOICE_LINE = re.compile(r"^[A-D]\. ", re.MULTILINE)
MATH_CHARS = re.compile(r"[+\-*/=%^$×÷]")
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

FEATURE_NAMES = [
    "log_chars", "log_words", "log_lines", "log_numbers", "choice_lines", "question_marks",
    "avg_word_len", "capital_ratio", "log_math_chars", "code_char_ratio", "reasoning_words",
    "constraint_words", "simple_words", "starts_wh", "log_sentences", "quotes", "has_currency",
]


def handcrafted(text: str) -> list[float]:
    words = WORD.findall(text)
    lower = [w.lower() for w in words]
    n_chars = len(text)
    n_words = len(words)
    first = lower[0] if lower else ""
    return [
        math.log1p(n_chars),
        math.log1p(n_words),
        math.log1p(text.count("\n") + 1),
        math.log1p(len(NUMBER.findall(text))),
        float(len(CHOICE_LINE.findall(text))),
        float(text.count("?")),
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        (sum(1 for w in words if w[0].isupper()) / n_words) if n_words else 0.0,
        math.log1p(len(MATH_CHARS.findall(text))),
        (sum(1 for c in text if c in CODE_CHARS) / n_chars) if n_chars else 0.0,
        float(sum(1 for w in lower if w in REASONING_WORDS)),
        float(sum(1 for w in lower if w in CONSTRAINT_WORDS)),
        float(sum(1 for w in lower if w in SIMPLE_WORDS)),
        float(first in WH_WORDS),
        math.log1p(len(SENTENCE_END.findall(text))),
        float(text.count('"') + text.count("\u201c") + text.count("\u201d")),
        float("$" in text),
    ]


def handcrafted_matrix(texts: list[str]) -> np.ndarray:
    return np.array([handcrafted(t) for t in texts], dtype=float)


def embed_matrix(client: Client, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings and the per-query cost of computing them (the router's overhead)."""
    vectors, tokens = client.embed(texts)
    costs = np.array([embed_cost(EMBED_MODEL, t) for t in tokens])
    return np.array(vectors, dtype=float), costs


FEATURE_SETS = ("handcrafted", "embed", "embed+handcrafted")


def feature_set(name: str, hand: np.ndarray, emb: np.ndarray) -> np.ndarray:
    if name == "handcrafted":
        return hand
    if name == "embed":
        return emb
    if name == "embed+handcrafted":
        return np.hstack([emb, hand])
    raise ValueError(name)
