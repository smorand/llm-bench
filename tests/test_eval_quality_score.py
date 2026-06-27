"""Unit tests for the unified quality_score (judge 'score' rubric + verdict map)."""

from __future__ import annotations

import pytest
import typer

from llm_bench.config import BenchConfig, EmbeddingConfig, EvaluationConfig, ModelRegistryEntry
from llm_bench.evaluation import (
    EvalPipeline,
    EvalRecord,
    _judge_payload,
    _parse_judge_score,
    _verdict_to_score,
)
from llm_bench.llm_bench import _apply_eval_overrides_or_exit


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


def _two_model_config() -> BenchConfig:
    return BenchConfig(
        models=[
            ModelRegistryEntry(name="sut", base_url="http://sut/v1", model="m-sut"),
            ModelRegistryEntry(name="grader", base_url="http://grader/v1", model="m-grader", api_key="k"),
        ]
    )


def test_judge_model_override_builds_judge_from_registry() -> None:
    """--judge-model points the judge at a registry entry's endpoint/model/key."""
    cfg = _two_model_config()
    _apply_eval_overrides_or_exit(cfg, judge_model="grader", judge_rubric="score", embedding_model=None)
    assert cfg.evaluation is not None and cfg.evaluation.judge is not None
    judge = cfg.evaluation.judge
    assert judge.model.url == "http://grader/v1"
    assert judge.model.model == "m-grader"
    assert judge.model.api_key == "k"
    assert judge.rubric == "score"


def test_embedding_model_override_builds_embedding_with_default_threshold() -> None:
    """--embedding-model synthesises an embedding block with a default threshold."""
    cfg = _two_model_config()
    _apply_eval_overrides_or_exit(cfg, judge_model=None, judge_rubric=None, embedding_model="grader")
    assert cfg.evaluation is not None and cfg.evaluation.embedding is not None
    emb = cfg.evaluation.embedding
    assert emb.url == "http://grader/v1"
    assert emb.threshold == 0.8


def test_eval_override_rejects_unknown_model_and_rubric() -> None:
    """An unknown model name or rubric exits non-zero."""
    cfg = _two_model_config()
    with pytest.raises(typer.Exit):
        _apply_eval_overrides_or_exit(cfg, judge_model="ghost", judge_rubric=None, embedding_model=None)
    with pytest.raises(typer.Exit):
        _apply_eval_overrides_or_exit(cfg, judge_model="grader", judge_rubric="bogus", embedding_model=None)


def test_embedding_local_override_builds_local_block() -> None:
    """--embedding-model local:cpu uses the built-in embedder (no url, default threshold)."""
    cfg = _two_model_config()
    _apply_eval_overrides_or_exit(cfg, judge_model=None, judge_rubric=None, embedding_model="local:cpu")
    assert cfg.evaluation is not None and cfg.evaluation.embedding is not None
    emb = cfg.evaluation.embedding
    assert emb.local == "cpu"
    assert emb.url is None
    assert emb.threshold == 0.8


def test_embedding_local_rejects_bad_preset() -> None:
    """An unknown local preset exits non-zero."""
    cfg = _two_model_config()
    with pytest.raises(typer.Exit):
        _apply_eval_overrides_or_exit(cfg, judge_model=None, judge_rubric=None, embedding_model="local:bogus")


def test_pipeline_progress_counts_enqueued_not_drops() -> None:
    """progress() reports (scored, enqueued); a drop on a full queue is not counted."""
    evaluation = EvaluationConfig(method="embedding", embedding=EmbeddingConfig(local="cpu", threshold=0.8))
    pipeline = EvalPipeline(evaluation, queue_maxsize=2, global_timeout=None)
    assert pipeline.progress() == (0, 0)

    pipeline.enqueue(EvalRecord(request_id="a", expected="x", actual="y"))
    pipeline.enqueue(EvalRecord(request_id="b", expected="x", actual="y"))
    assert pipeline.progress() == (0, 2)

    # The queue is full (maxsize 2): this record is dropped, so enqueued stays at 2.
    pipeline.enqueue(EvalRecord(request_id="c", expected="x", actual="y"))
    assert pipeline.dropped == 1
    assert pipeline.progress() == (0, 2)
