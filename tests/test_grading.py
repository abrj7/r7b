import pytest

from r7b.grading import (
    final_answer,
    grade,
    grade_gsm8k,
    grade_mmlu,
    grade_nq,
    judge_to_quality,
    normalize,
    parse_judge,
    r7b_good_enough,
)


def test_final_answer_prefers_last_marker():
    assert final_answer("Answer: 3\nwait\nAnswer: 5") == "5"
    assert final_answer("no marker\nlast line") == "last line"
    assert final_answer("**Answer:** 42") == "42"


def test_gsm8k():
    assert grade_gsm8k("so 18 eggs\nAnswer: 18", "18") == 1.0
    assert grade_gsm8k("Answer: $1,250", "1250") == 1.0
    assert grade_gsm8k("Answer: 18.0", "18") == 1.0
    assert grade_gsm8k("Answer: 17", "18") == 0.0
    assert grade_gsm8k("I think it's 18 in total", "18") == 1.0
    assert grade_gsm8k("no idea", "18") == 0.0


def test_mmlu():
    assert grade_mmlu("Reasoning...\nAnswer: B", "B") == 1.0
    assert grade_mmlu("Answer: (C)", "C") == 1.0
    assert grade_mmlu("Answer: D. Paris", "D") == 1.0
    assert grade_mmlu("Answer: A", "B") == 0.0


def test_nq_alias_span_match():
    assert grade_nq("Answer: George Washington", ["George Washington"]) == 1.0
    assert grade_nq("Answer: the leg", ["leg"]) == 1.0
    assert grade_nq("Answer: Washington", ["George Washington"]) == 0.0
    assert grade_nq("Answer: legend", ["leg"]) == 0.0
    assert grade_nq("Answer: 1945.", ["1945"]) == 1.0


def test_normalize():
    assert normalize("The  Beatles!") == "beatles"


def test_grade_dispatch_rejects_open():
    with pytest.raises(ValueError):
        grade("open", "x", None)


def test_parse_judge_json_and_fences():
    assert parse_judge('{"response_1": {"score": 7}, "response_2": {"score": 9}}') == (7, 9)
    fenced = '```json\n{"response_1": {"score": 3, "reason": "x"}, ' \
             '"response_2": {"score": 11, "reason": "y"}}\n```'
    assert parse_judge(fenced) == (3, 10)


def test_parse_judge_fallback_and_failure():
    assert parse_judge('response_1 "score": 4 ... "score": 6 trailing') == (4, 6)
    with pytest.raises(ValueError):
        parse_judge("I refuse")


def test_judge_to_quality_range():
    assert judge_to_quality(1) == 0
    assert judge_to_quality(10) == 1


def test_good_enough_label():
    assert r7b_good_enough("gsm8k", 1, 1)
    assert r7b_good_enough("gsm8k", 0, 0)  # escalating would not have helped
    assert not r7b_good_enough("gsm8k", 0, 1)
    assert r7b_good_enough("open", judge_to_quality(7), judge_to_quality(8))
    assert not r7b_good_enough("open", judge_to_quality(6), judge_to_quality(8))
