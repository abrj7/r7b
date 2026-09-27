"""Models, paths, and the trial-key call budget."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = ROOT / "cache"
RUNS_DIR = ROOT / "runs"
SITE_DATA_DIR = ROOT / "site" / "public" / "data"

CHEAP_MODEL = "command-r7b-12-2024"
STRONG_MODEL = "command-a-03-2025"
JUDGE_MODEL = STRONG_MODEL
EMBED_MODEL = "embed-v4.0"
EMBED_DIM = 256
# /v2/embed accepts at most 96 texts per request.
EMBED_BATCH = 96

SEED = 7

# Trial-key limits, verified against docs.cohere.com/docs/rate-limits on 2026-09-27:
# 1,000 calls per month across the key, 20 chat requests/min per model,
# 2,000 embed inputs/min.
TRIAL_CHAT_RPM = 20
TRIAL_EMBED_INPUTS_PER_MIN = 2000


@dataclass(frozen=True)
class Budget:
    monthly_cap: int
    reserve: int

    @property
    def usable(self) -> int:
        return self.monthly_cap - self.reserve


def budget_from_env() -> Budget:
    return Budget(
        monthly_cap=int(os.getenv("R7B_MONTHLY_CALL_CAP", "1000")),
        reserve=int(os.getenv("R7B_CALL_RESERVE", "40")),
    )


def api_key() -> str | None:
    key = os.getenv("COHERE_API_KEY", "").strip()
    return key or None


def run_dir(mode: str) -> Path:
    path = RUNS_DIR / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(mode: str) -> Path:
    path = CACHE_DIR / mode
    path.mkdir(parents=True, exist_ok=True)
    return path
