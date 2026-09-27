"""Build the query set: ground-truth subsets plus hand-written open-ended prompts.

The sampled set is committed as data/queries.jsonl, so a fresh clone never
needs to re-download anything. Rebuilding is deterministic given SEED.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from r7b.config import DATA_DIR, RAW_DIR, SEED

QUERIES_PATH = DATA_DIR / "queries.jsonl"
OPEN_PROMPTS_PATH = DATA_DIR / "open_prompts.jsonl"

# Sized so every query can be answered by both models and every open-ended
# pair judged inside one month of a trial key. See r7b.budget.
SAMPLE_SIZES = {"gsm8k": 100, "mmlu": 120, "nq": 80}

SOURCES = {
    "gsm8k": "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet",
    "mmlu": "https://huggingface.co/api/datasets/cais/mmlu/parquet/all/test/0.parquet",
    "nq": "https://huggingface.co/api/datasets/google-research-datasets/nq_open/parquet/nq_open/validation/0.parquet",
}

LETTERS = "ABCD"


@dataclass
class Query:
    id: str
    source: str
    text: str
    gold: str | list[str] | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _download(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        with httpx.stream("GET", SOURCES[name], follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return path


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)]


def gsm8k_queries(n: int, rng: np.random.Generator) -> list[Query]:
    df = _sample(pd.read_parquet(_download("gsm8k")), n, rng)
    out = []
    for i, row in df.iterrows():
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(Query(id=f"gsm8k-{i:04d}", source="gsm8k", text=row["question"].strip(),
                         gold=gold))
    return out


def mmlu_queries(n: int, rng: np.random.Generator) -> list[Query]:
    df = _sample(pd.read_parquet(_download("mmlu")), n, rng)
    out = []
    for i, row in df.iterrows():
        choices = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(row["choices"]))
        text = f"{row['question'].strip()}\n\n{choices}"
        out.append(Query(id=f"mmlu-{i:05d}", source="mmlu", text=text,
                         gold=LETTERS[int(row["answer"])], meta={"subject": row["subject"]}))
    return out


def nq_queries(n: int, rng: np.random.Generator) -> list[Query]:
    df = _sample(pd.read_parquet(_download("nq")), n, rng)
    out = []
    for i, row in df.iterrows():
        question = row["question"].strip()
        question = question[0].upper() + question[1:]
        if not question.endswith("?"):
            question += "?"
        out.append(Query(id=f"nq-{i:04d}", source="nq", text=question,
                         gold=[str(a) for a in row["answer"]]))
    return out


def open_queries() -> list[Query]:
    out = []
    for line in OPEN_PROMPTS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append(Query(id=rec["id"], source="open", text=rec["prompt"],
                         meta={"category": rec["category"], "difficulty": rec["difficulty"]}))
    return out


def build_query_set(path: Path = QUERIES_PATH) -> list[Query]:
    rng = np.random.default_rng(SEED)
    queries = (
        gsm8k_queries(SAMPLE_SIZES["gsm8k"], rng)
        + mmlu_queries(SAMPLE_SIZES["mmlu"], rng)
        + nq_queries(SAMPLE_SIZES["nq"], rng)
        + open_queries()
    )
    write_queries(queries, path)
    return queries


def write_queries(queries: list[Query], path: Path = QUERIES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    return [Query(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
