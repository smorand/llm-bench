"""Unit tests for the unified quality_score (judge 'score' rubric + verdict map)."""

from __future__ import annotations

from llm_bench.evaluation import (
    EvalRecord,
    _judge_payload,
    _parse_judge_score,
    _verdict_to_score,
)


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def test_parse_judge_score_reads_and_clamps() -> None:
    """A 0..1 score is parsed; out-of-range values clamp; junk falls back to 0."""
    assert _parse_judge_score(_reply('{"score": 0.8, "reason": "good"}')) == (0.8, "good")
    assert _parse_judge_score(_reply('{"score": 1.7}'))[0] == 1.0  # clamped high
    assert _parse_judge_score(_reply('{"score": -0.5}'))[0] == 0.0  # clamped low
    assert _parse_judge_score(_reply("not json"))[0] == 0.0  # non-numeric fallback


def test_verdict_to_score_mapping() -> None:
    """Categorical verdicts map to a 0..1 quality score."""
    assert _verdict_to_score("pass", "binary") == 1.0
    assert _verdict_to_score("fail", "binary") == 0.0
    assert _verdict_to_score("correct", "three_level") == 1.0
    assert _verdict_to_score("partial", "three_level") == 0.5
    assert _verdict_to_score("incorrect", "three_level") == 0.0


def test_judge_payload_score_rubric_asks_for_a_number() -> None:
    """The 'score' rubric prompts the model for a 0..1 compliance number."""
    record = EvalRecord(request_id="r1", expected="a", actual="b")
    payload = _judge_payload("m", None, record, "score")
    system = payload["messages"][0]["content"]
    assert '"score"' in system
    assert "between 0 and 1" in system
    # the categorical rubrics still forbid a numeric score
    cat = _judge_payload("m", None, record, "binary")["messages"][0]["content"]
    assert "Do not use any numeric score" in cat
