import pytest

from r7b.config import CHEAP_MODEL, EMBED_MODEL, STRONG_MODEL
from r7b.costs import CHAT_PRICES, chat_cost, embed_cost, price_ratio


def test_chat_cost_uses_per_million_prices():
    assert chat_cost(STRONG_MODEL, 1_000_000, 0) == pytest.approx(2.50)
    assert chat_cost(STRONG_MODEL, 0, 1_000_000) == pytest.approx(10.00)
    assert chat_cost(CHEAP_MODEL, 1_000_000, 1_000_000) == pytest.approx(0.0375 + 0.15)


def test_chat_cost_small_request():
    # 200 input + 150 output tokens on Command A
    expected = (200 * 2.50 + 150 * 10.00) / 1e6
    assert chat_cost(STRONG_MODEL, 200, 150) == pytest.approx(expected)


def test_zero_tokens_cost_nothing():
    assert chat_cost(CHEAP_MODEL, 0, 0) == 0
    assert embed_cost(EMBED_MODEL, 0) == 0


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        chat_cost(CHEAP_MODEL, -1, 0)
    with pytest.raises(ValueError):
        embed_cost(EMBED_MODEL, -5)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        chat_cost("command-nonexistent", 1, 1)


def test_embed_cost():
    assert embed_cost(EMBED_MODEL, 1_000_000) == pytest.approx(0.12)


def test_command_a_is_about_67x_r7b():
    inp, out = price_ratio()
    assert inp == pytest.approx(2.50 / 0.0375)
    assert out == pytest.approx(10.00 / 0.15)
    assert CHAT_PRICES[STRONG_MODEL].output_per_m > CHAT_PRICES[CHEAP_MODEL].output_per_m
