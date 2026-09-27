"""Chat and embed clients. The live backend talks to Cohere; the mock backend
(r7b.mock) is a deterministic stand-in with the same interface. Both sit behind
`Client`, which checks the disk cache before touching the backend."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol

from r7b.budget import Ledger, RateLimiter
from r7b.cache import DiskCache
from r7b.config import (
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_MODEL,
    SEED,
    TRIAL_CHAT_RPM,
    TRIAL_EMBED_INPUTS_PER_MIN,
    api_key,
    budget_from_env,
    cache_dir,
)

EMBED_INPUT_TYPE = "classification"


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    input_tokens: int


class Backend(Protocol):
    name: str

    def chat(self, model: str, message: str, max_tokens: int) -> ChatResult: ...

    def embed(self, texts: list[str]) -> EmbedResult: ...


class LiveBackend:
    name = "live"

    def __init__(self, key: str, ledger: Ledger, max_retries: int = 5):
        import cohere

        self._errors = cohere.errors
        self.co = cohere.ClientV2(api_key=key, timeout=120)
        self.ledger = ledger
        self.max_retries = max_retries
        self.chat_limiter = RateLimiter(TRIAL_CHAT_RPM)
        # One embed request per 96 inputs; 2,000 inputs/min is ~20 requests/min.
        self.embed_limiter = RateLimiter(TRIAL_EMBED_INPUTS_PER_MIN / EMBED_BATCH)

    def _with_retries(self, fn, what: str):
        retryable = (
            self._errors.TooManyRequestsError,
            self._errors.ServiceUnavailableError,
            self._errors.InternalServerError,
            self._errors.GatewayTimeoutError,
        )
        for attempt in range(self.max_retries):
            try:
                return fn()
            except retryable:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(min(60, 4 * 2**attempt))
        raise RuntimeError(f"unreachable: {what}")

    def chat(self, model: str, message: str, max_tokens: int) -> ChatResult:
        def call():
            self.ledger.record("chat", model)
            self.chat_limiter.wait(model)
            return self.co.chat(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
                temperature=0.0,
                seed=SEED,
            )

        r = self._with_retries(call, f"chat {model}")
        text = "".join(c.text for c in (r.message.content or []) if c.type == "text")
        billed = r.usage.billed_units
        return ChatResult(
            text=text,
            input_tokens=int(billed.input_tokens or 0),
            output_tokens=int(billed.output_tokens or 0),
            finish_reason=r.finish_reason,
        )

    def embed(self, texts: list[str]) -> EmbedResult:
        if len(texts) > EMBED_BATCH:
            raise ValueError(f"at most {EMBED_BATCH} texts per embed request")

        def call():
            self.ledger.record("embed", EMBED_MODEL, note=f"{len(texts)} texts")
            self.embed_limiter.wait(EMBED_MODEL)
            return self.co.embed(
                model=EMBED_MODEL,
                texts=texts,
                input_type=EMBED_INPUT_TYPE,
                embedding_types=["float"],
                output_dimension=EMBED_DIM,
            )

        r = self._with_retries(call, "embed")
        tokens = int((r.meta.billed_units.input_tokens if r.meta and r.meta.billed_units
                      else 0) or 0)
        return EmbedResult(vectors=[list(v) for v in r.embeddings.float_], input_tokens=tokens)


class Client:
    """Cache-first access to a backend. Identical requests are never sent twice."""

    def __init__(self, backend: Backend, cache: DiskCache):
        self.backend = backend
        self.cache = cache
        self.stats = {"chat_hits": 0, "chat_calls": 0, "embed_hits": 0, "embed_calls": 0}

    @property
    def mode(self) -> str:
        return self.backend.name

    def chat(self, model: str, message: str, max_tokens: int) -> ChatResult:
        payload = {"model": model, "message": message, "max_tokens": max_tokens,
                   "temperature": 0.0, "seed": SEED}
        hit = self.cache.get("chat", payload)
        if hit is not None:
            self.stats["chat_hits"] += 1
            return ChatResult(**hit)
        result = self.backend.chat(model, message, max_tokens)
        self.stats["chat_calls"] += 1
        self.cache.put("chat", payload, result.to_dict())
        return result

    def embed(self, texts: list[str]) -> tuple[list[list[float]], list[int]]:
        """Embed texts, cached per text, sending only misses in batches of 96.

        Billed tokens come back per request; they are split across the batch
        in proportion to text length so each text carries its share.
        """
        payloads = [{"model": EMBED_MODEL, "text": t, "input_type": EMBED_INPUT_TYPE,
                     "dim": EMBED_DIM} for t in texts]
        vectors: list[list[float] | None] = [None] * len(texts)
        tokens: list[int] = [0] * len(texts)
        missing = []
        for i, p in enumerate(payloads):
            hit = self.cache.get("embed", p)
            if hit is None:
                missing.append(i)
            else:
                vectors[i], tokens[i] = hit["vector"], hit["input_tokens"]
                self.stats["embed_hits"] += 1
        for start in range(0, len(missing), EMBED_BATCH):
            idx = missing[start:start + EMBED_BATCH]
            batch = [texts[i] for i in idx]
            result = self.backend.embed(batch)
            self.stats["embed_calls"] += 1
            lengths = [max(1, len(t)) for t in batch]
            total = sum(lengths)
            for j, i in enumerate(idx):
                share = round(result.input_tokens * lengths[j] / total)
                vectors[i], tokens[i] = result.vectors[j], share
                self.cache.put("embed", payloads[i],
                               {"vector": result.vectors[j], "input_tokens": share})
        return [v for v in vectors if v is not None], tokens


def make_client(mode: str) -> Client:
    if mode == "mock":
        from r7b.mock import MockBackend, simulator_version

        return Client(MockBackend(), DiskCache(cache_dir("mock") / simulator_version()))
    cache = DiskCache(cache_dir(mode))
    if mode == "live":
        key = api_key()
        if key is None:
            raise SystemExit("COHERE_API_KEY is not set. Add it to .env, or run with --mock.")
        ledger = Ledger(cache_dir("live") / "ledger.jsonl", budget_from_env())
        return Client(LiveBackend(key, ledger), cache)
    raise ValueError(f"unknown mode {mode!r}")
