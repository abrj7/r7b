import json
from pathlib import Path

import pytest

from r7b.features import FEATURE_NAMES, handcrafted

FIXTURE = Path(__file__).parent / "fixtures" / "feature_examples.json"


def test_feature_vector_shape():
    assert len(handcrafted("What is 2 + 2?")) == len(FEATURE_NAMES)
    assert len(handcrafted("")) == len(FEATURE_NAMES)


def test_choice_lines_detected():
    names = FEATURE_NAMES
    v = dict(zip(names, handcrafted("Pick one.\n\nA. x\nB. y\nC. z\nD. w"), strict=True))
    assert v["choice_lines"] == 4
    assert v["starts_wh"] == 0


def test_keywords():
    v = dict(zip(FEATURE_NAMES, handcrafted("Explain why, in exactly 50 words."), strict=True))
    assert v["reasoning_words"] == 2
    assert v["constraint_words"] == 2
    assert v["log_numbers"] > 0


def test_matches_shared_fixture():
    """site/src/lib/features.ts is checked against the same fixture, so the
    in-browser router sees exactly the features it was trained on."""
    cases = json.loads(FIXTURE.read_text())
    for case in cases:
        assert handcrafted(case["text"]) == pytest.approx(case["features"], abs=1e-9)
