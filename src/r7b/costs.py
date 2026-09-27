"""Per-token pricing and cost arithmetic.

Prices are USD per 1M tokens. Checked 2026-09-27:
- Command R7B and Embed 4: first-party, from the pricing cards on cohere.com/pricing.
- Command A: no longer listed on cohere.com/pricing (the page now leads with
  Command A+). $2.50 / $10.00 is Cohere's list price, and it is what Cohere
  charges as the provider on OpenRouter, where its R7B price matches the
  first-party card exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from r7b.config import CHEAP_MODEL, EMBED_MODEL, STRONG_MODEL

PRICES_CHECKED = "2026-09-27"


@dataclass(frozen=True)
class ChatPrice:
    input_per_m: float
    output_per_m: float


CHAT_PRICES: dict[str, ChatPrice] = {
    CHEAP_MODEL: ChatPrice(input_per_m=0.0375, output_per_m=0.15),
    STRONG_MODEL: ChatPrice(input_per_m=2.50, output_per_m=10.00),
}

EMBED_PRICE_PER_M: dict[str, float] = {EMBED_MODEL: 0.12}


def chat_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    price = CHAT_PRICES[model]
    return (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000


def embed_cost(model: str, input_tokens: int) -> float:
    if input_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return input_tokens * EMBED_PRICE_PER_M[model] / 1_000_000


def price_ratio(cheap: str = CHEAP_MODEL, strong: str = STRONG_MODEL) -> tuple[float, float]:
    """How many times more the strong model costs per input and per output token."""
    c, s = CHAT_PRICES[cheap], CHAT_PRICES[strong]
    return s.input_per_m / c.input_per_m, s.output_per_m / c.output_per_m


def pricing_table() -> dict:
    return {
        "checked": PRICES_CHECKED,
        "chat": {
            m: {"input_per_m": p.input_per_m, "output_per_m": p.output_per_m}
            for m, p in CHAT_PRICES.items()
        },
        "embed": {m: {"input_per_m": p} for m, p in EMBED_PRICE_PER_M.items()},
    }
