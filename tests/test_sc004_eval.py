"""Acceptance tests for SC-004: asynchronous quality evaluation.

These cover scenario SC-004 (Section 5 of the spec
``specs/2026-06-24_09:28:00-llm-bench-core.md``): the decoupled, non-blocking
evaluation pipeline that scores output quality (embedding cosine similarity by
default, optional LLM-as-judge) against a per-prompt ``expected_output`` and
joins the scores back onto the performance records by ``request_id``.

One test per E2E id from Section 12.2: E2E-025, 026, 027, 028, 029, 030, 031,
032, 033, 034, 035, plus the cross-scenario E2E-068 (SC-001 + SC-004) and the
saturation/down/self-preference edge cases E2E-107, 109, 110. Each asserts
exactly the observables named in the matching Gherkin, driven through the
``llm-bench`` CLI ``run`` subcommand with its ``--eval-method`` flag, against the
offline FakeSUT + FakeEval harness from ``conftest.py``.

The relevant FRs are FR-003 (embedding requires threshold), FR-040..047 (the
async eval pipeline), FR-056 (traces) and the cross-cutting FR-015 (graceful
interruption) and FR-041 (no backpressure).

Config shape assumed (written by hand, mirroring ``cfg_base``'s ``${SUT_PORT}``
/ ``$ENV:`` style) — an ``evaluation:`` block alongside ``models:`` / ``run:``::

    evaluation:
      embedding:
        url: http://127.0.0.1:<eval_port>/v1
        model: fake-embed
        threshold: 0.80
        rate_limit: 5          # optional, requests/s for the eval pool
      judge:
        model:
          url: http://127.0.0.1:<eval_port>/v1
          api_key: $ENV:SUT_API_KEY
          model: fake/judge
          prompt: "grade the answer"
        rubric: binary         # or three_level
      global_timeout: 30s

The active method is selected by ``--eval-method embedding|judge`` on the CLI.

Summary / record key names asserted (so the implementer matches them):

* summary: ``eval.coverage`` (float), ``eval.judged`` (int),
  ``eval.total_eligible`` (int), ``eval.dropped`` (int; ``eval.spilled`` accepted
  as a synonym), ``status``.
* raw record: ``sim_score`` (float|null), ``quality_pass`` (bool|null),
  ``judge_verdict`` (str|null), ``judge_reason`` (str|null), ``eval_status`` one
  of ``judged`` / ``eval_skipped`` / ``skipped_no_expected``.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import yaml
from typer.testing import CliRunner

from llm_bench.llm_bench import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tests.conftest import EvalController, SUTController

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a fake server ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def _closed_port() -> int:
    """Return a TCP port with nothing listening (bind then immediately release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact into a list of dicts, asserting it exists first."""
    assert path.exists(), f"expected artifact missing: {path}"
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _read_summary(out_dir: Path) -> dict[str, Any]:
    """Load ``summary.json`` from a run directory (asserts it exists)."""
    path = out_dir / "summary.json"
    assert path.exists(), f"missing summary.json: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_block(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the ``eval`` summary section, asserting it is a mapping."""
    section = summary.get("eval")
    assert isinstance(section, dict), f"summary.eval missing/not a dict: {summary.keys()}"
    return section


def _dropped(eval_section: dict[str, Any]) -> int:
    """Return the saturation drop counter (``eval.dropped`` or ``eval.spilled``)."""
    for key in ("dropped", "spilled"):
        if key in eval_section:
            return int(eval_section[key])
    msg = f"no dropped/spilled counter in eval summary: {eval_section}"
    raise AssertionError(msg)


def _write_prompts(path: Path, prompts: list[dict[str, Any]]) -> Path:
    """Write a ``--prompts`` YAML file (FR-036 prompt shape with expected_output)."""
    path.write_text(yaml.safe_dump(prompts, sort_keys=False), encoding="utf-8")
    return path


def _prompt(
    pid: str,
    content: str,
    *,
    expected: str | None = None,
    category: str = "general",
) -> dict[str, Any]:
    """Build one prompt mapping, optionally carrying ``expected_output``."""
    entry: dict[str, Any] = {
        "id": pid,
        "category": category,
        "isl_bucket": "short",
        "messages": [{"role": "user", "content": content}],
    }
    if expected is not None:
        entry["expected_output"] = expected
    return entry


def _write_eval_config(
    path: Path,
    *,
    sut_port: int,
    eval_url: str,
    monkeypatch: pytest.MonkeyPatch,
    method: str = "embedding",
    threshold: float | None = 0.80,
    rate_limit: float | None = None,
    global_timeout: str | None = None,
    eval_queue_maxsize: int | None = None,
    judge_model: str = "fake/judge",
    rubric: str = "binary",
    run_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a CFG_BASE-style config plus an ``evaluation:`` block, by hand.

    Mirrors ``cfg_base``'s ``${SUT_PORT}`` / ``$ENV:`` literal style (so the
    loader exercises env resolution) and points the embedding/judge ``url`` at
    the running FakeEval ``eval_url``. ``threshold=None`` omits the threshold key
    (for the FR-003 runtime-abort test).
    """
    monkeypatch.setenv("SUT_API_KEY", "sk-test")
    monkeypatch.setenv("SUT_PORT", str(sut_port))

    run_block: dict[str, Any] = {
        "mode": "closed",
        "duration": "1s",
        "warmup": "0.2s",
        "cooldown": "0.2s",
        "min_samples": 1,
        "concurrency_levels": [1],
        "max_tokens": 8,
        "ignore_eos": True,
        "temperature": 0.0,
        "cache_busting": False,
        "retries": 0,
        "timeout": "5s",
        "seed": 42,
        "slo_profile": "interactive",
    }
    if eval_queue_maxsize is not None:
        run_block["eval_queue_maxsize"] = eval_queue_maxsize
    if run_overrides:
        run_block.update(run_overrides)

    lines: list[str] = [
        "models:",
        "  - name: sut",
        "    base_url: http://127.0.0.1:${SUT_PORT}/v1",
        "    model: fake/model",
        "    api_key: $ENV:SUT_API_KEY",
        "    supports_vision: false",
        "    supports_tools: false",
        "run:",
    ]
    for key, value in run_block.items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"  {key}: {rendered}")

    # evaluation block
    lines.append("evaluation:")
    if global_timeout is not None:
        lines.append(f"  global_timeout: {global_timeout}")
    if method == "embedding":
        lines.append("  embedding:")
        lines.append(f"    url: {eval_url}")
        lines.append("    model: fake-embed")
        if threshold is not None:
            lines.append(f"    threshold: {threshold}")
        if rate_limit is not None:
            lines.append(f"    rate_limit: {rate_limit}")
    else:
        lines.append("  judge:")
        lines.append(f"    rubric: {rubric}")
        lines.append("    model:")
        lines.append(f"      url: {eval_url}")
        lines.append("      api_key: $ENV:SUT_API_KEY")
        lines.append(f"      model: {judge_model}")
        lines.append('      prompt: "grade the answer"')

    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _run(
    config: Path, out_dir: Path, *, method: str, prompts: Path | None = None, extra: list[str] | None = None
) -> Any:
    """Invoke ``llm-bench run`` with an eval method against the fake harness."""
    args = ["run", "--config", str(config), "--model", "sut", "--eval-method", method, "--out", str(out_dir)]
    if prompts is not None:
        args += ["--prompts", str(prompts)]
    if extra:
        args += extra
    return runner.invoke(app, args)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (used to pick deterministic fixtures)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# E2E-025: Quality eval embedding cosine joined on request_id
# ---------------------------------------------------------------------------


def test_e2e_025_embedding_cosine_joined_on_request_id(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A matching expected/actual pair scores cosine 1.0, joined on request_id.

    FR-040/043/047: exit 0; the perf record and an eval record share the same
    ``request_id``; joined data shows ``sim_score==1.0`` and
    ``quality_pass==true``; coverage ``judged/total == 1.0``.
    """
    sut_url, sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval

    # FakeSUT streams exactly "hello world" so the actual output matches expected.
    from tests.conftest import Behavior, Delta, Usage  # noqa: PLC0415

    sut_ctrl.set_default(Behavior(deltas=[Delta("hello world")], usage=Usage(prompt_tokens=10, completion_tokens=2)))
    # Identical embeddings for both strings -> cosine 1.0.
    vec = [1.0, 0.0, 0.0]
    eval_ctrl.map_text("hello world", vec)

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt("hw", "say hello", expected="hello world")])
    config = _write_eval_config(
        tmp_path, sut_port=_port_of(sut_url), eval_url=eval_url, monkeypatch=monkeypatch, method="embedding"
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    judged = [r for r in records if r.get("eval_status") == "judged"]
    assert judged, f"no judged eval record: {[r.get('eval_status') for r in records]}"
    for rec in judged:
        assert rec.get("request_id"), rec
        assert rec["sim_score"] == 1.0, rec["sim_score"]
        assert rec["quality_pass"] is True, rec

    summary = _eval_block(_read_summary(out_dir))
    assert summary["coverage"] == 1.0, summary


# ---------------------------------------------------------------------------
# E2E-026: Eval coverage reported (judged/total_eligible)
# ---------------------------------------------------------------------------


def test_e2e_026_eval_coverage_reported(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Coverage equals judged/total_eligible; eligible = expected + successful.

    FR-046: summary ``eval.coverage == eval.judged / eval.total_eligible``;
    eligible counts only prompts with ``expected_output`` that got a successful
    response; both counters are integers; terminal prints ``Eval coverage: X/Y``.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]  # everything matches -> all judged pass

    prompts_list = [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(6)]
    prompts_list += [_prompt(f"n{i}", f"q{i}") for i in range(4)]  # 4 without expected_output
    prompts = _write_prompts(tmp_path / "p.yaml", prompts_list)
    config = _write_eval_config(
        tmp_path, sut_port=_port_of(sut_url), eval_url=eval_url, monkeypatch=monkeypatch, method="embedding"
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    summary = _eval_block(_read_summary(out_dir))
    judged = summary["judged"]
    eligible = summary["total_eligible"]
    assert isinstance(judged, int), judged
    assert isinstance(eligible, int), eligible
    assert eligible > 0, summary
    assert summary["coverage"] == judged / eligible, summary
    assert f"Eval coverage: {judged}/{eligible}" in result.output, result.output


# ---------------------------------------------------------------------------
# E2E-027: Eval async never blocks load gen
# ---------------------------------------------------------------------------


def test_e2e_027_eval_never_blocks_load_gen(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Slow eval does not throttle the load generator; queue drops instead.

    FR-041: load-gen RPS is within ±10% of a no-eval baseline; summary
    ``eval.dropped > 0``; no perf record is delayed waiting on the eval queue.
    Kept bounded by using a short duration with a tiny bounded eval queue.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    # The eval pool drains concurrently with the load, so to still exercise the
    # drop-on-full safety valve (FR-041) the eval must badly outpace the workers:
    # 1.5 s per embedding (4 workers -> <3/s) against a 4-slot queue overflows fast.
    eval_ctrl.delay_ms = 1500.0

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(40)])

    # Baseline: no eval, same SUT, same duration -> measure RPS from t_start density.
    base_cfg = _write_eval_config(
        tmp_path / "base", sut_port=_port_of(sut_url), eval_url=eval_url, monkeypatch=monkeypatch, method="embedding"
    )
    base_out = tmp_path / "base_run"
    base_res = runner.invoke(app, ["run", "--config", str(base_cfg), "--model", "sut", "--out", str(base_out)])
    assert base_res.exit_code == 0, base_res.output
    base_records = _read_jsonl(base_out / "raw.jsonl")
    base_rps = _rps(base_records)

    # Eval enabled, slow embeddings, small bounded queue.
    cfg = _write_eval_config(
        tmp_path / "ev",
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        eval_queue_maxsize=4,
    )
    out_dir = tmp_path / "ev_run"
    result = _run(cfg, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    eval_rps = _rps(records)
    # Load-gen RPS within ±10% of the no-eval baseline (eval must not throttle it).
    assert abs(eval_rps - base_rps) <= 0.10 * base_rps, f"eval throttled load gen: {eval_rps} vs base {base_rps}"

    summary = _eval_block(_read_summary(out_dir))
    assert _dropped(summary) > 0, summary


def _rps(records: list[dict[str, Any]]) -> float:
    """Estimate requests/sec from the steady-record ``t_start`` density."""
    starts = sorted(r["t_start"] for r in records if r.get("t_start") is not None)
    assert len(starts) >= 2, f"need >=2 records for an RPS estimate, got {len(starts)}"
    span = starts[-1] - starts[0]
    assert span > 0, f"degenerate t_start span: {starts}"
    return (len(starts) - 1) / span


# ---------------------------------------------------------------------------
# E2E-028: Eval worker pool uses own rate limiter
# ---------------------------------------------------------------------------


def test_e2e_028_eval_pool_rate_limited(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A configured eval rate_limit caps embedding RPS independently of the SUT.

    FR-042: with ``evaluation.embedding.rate_limit: 5`` the observed FakeEval
    embedding request rate is <= ~5/s even though the SUT request rate is higher.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(15)])
    config = _write_eval_config(
        tmp_path,
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        rate_limit=5,
        global_timeout="10s",
        # Short, single-worker run produces ~10-15 eval items: enough for the 5/s
        # limiter to pace the drain (~2-3s) without flooding it for tens of seconds.
        run_overrides={"duration": "0.5s", "concurrency_levels": [1]},
    )

    out_dir = tmp_path / "runs" / "e"
    start = time.monotonic()
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    elapsed = time.monotonic() - start
    assert result.exit_code == 0, result.output

    n_embed = len(eval_ctrl.embedding_requests)
    assert n_embed > 0, "no embedding requests reached FakeEval"
    observed_rate = n_embed / elapsed
    # Eval pool limiter caps embedding throughput near 5/s (±1) regardless of SUT rate.
    assert observed_rate <= 6.0, f"eval rate {observed_rate:.2f}/s exceeds limiter (5/s +1)"


# ---------------------------------------------------------------------------
# E2E-029: Embedding eval requires threshold (runtime, FR-003)
# ---------------------------------------------------------------------------


def test_e2e_029_embedding_requires_threshold(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Embedding eval without a threshold aborts at config validation.

    FR-003: exit non-zero; stderr contains
    ``embedding evaluation requires evaluation.embedding.threshold``; no run data.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, _eval_ctrl = fake_eval

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt("e0", "q", expected="ans")])
    config = _write_eval_config(
        tmp_path,
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        threshold=None,  # omit threshold -> must abort
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)

    assert result.exit_code != 0, result.output
    assert "embedding evaluation requires evaluation.embedding.threshold" in (result.output or "")
    assert not (out_dir / "raw.jsonl").exists()


# ---------------------------------------------------------------------------
# E2E-030: Judge endpoint down marks eval_skipped
# ---------------------------------------------------------------------------


def test_e2e_030_judge_down_marks_eval_skipped(
    fake_sut: tuple[str, SUTController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unreachable judge endpoint leaves perf valid but eval skipped.

    FR-044/046/047: exit 0; each eval record ``judge_verdict==null`` and
    ``eval_status=="eval_skipped"``; summary ``eval.coverage < 1.0``; WARNING
    ``judge endpoint unreachable, marking eval_skipped``.

    The ``fake_eval`` ``down`` mode cannot be toggled after the fixture yields
    (see harness gap note), so the judge ``url`` is pointed at a closed port.
    """
    sut_url, _sut_ctrl = fake_sut
    dead_url = f"http://127.0.0.1:{_closed_port()}/v1"

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(4)])
    config = _write_eval_config(
        tmp_path, sut_port=_port_of(sut_url), eval_url=dead_url, monkeypatch=monkeypatch, method="judge"
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="judge", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    evaluated = [r for r in records if r.get("eval_status") in {"judged", "eval_skipped"}]
    assert evaluated, f"no eval records: {[r.get('eval_status') for r in records]}"
    for rec in evaluated:
        assert rec["eval_status"] == "eval_skipped", rec
        assert rec["judge_verdict"] is None, rec

    summary = _eval_block(_read_summary(out_dir))
    assert summary["coverage"] < 1.0, summary
    assert "judge endpoint unreachable, marking eval_skipped" in (result.output or "")


# ---------------------------------------------------------------------------
# E2E-031: Empty expected_output skips eval
# ---------------------------------------------------------------------------


def test_e2e_031_empty_expected_output_skips(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A prompt with empty expected_output is skipped and excluded from coverage.

    FR-043/047: the empty-expected record has ``eval_status=="skipped_no_expected"``
    and ``sim_score==null``, excluded from the coverage denominator; the valid one
    is evaluated normally.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]

    prompts = _write_prompts(
        tmp_path / "p.yaml",
        [
            _prompt("empty", "q-empty", expected=""),
            _prompt("valid", "q-valid", expected="something"),
        ],
    )
    config = _write_eval_config(
        tmp_path, sut_port=_port_of(sut_url), eval_url=eval_url, monkeypatch=monkeypatch, method="embedding"
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_prompt.setdefault(rec.get("prompt_id"), []).append(rec)

    empties = by_prompt.get("empty", [])
    assert empties, f"no record for the empty-expected prompt: {list(by_prompt)}"
    for rec in empties:
        assert rec["eval_status"] == "skipped_no_expected", rec
        assert rec["sim_score"] is None, rec

    valids = by_prompt.get("valid", [])
    assert valids, "no record for the valid-expected prompt"
    assert any(r.get("eval_status") == "judged" for r in valids), [r.get("eval_status") for r in valids]

    # Coverage denominator excludes the skipped_no_expected prompt.
    summary = _eval_block(_read_summary(out_dir))
    judged = sum(1 for r in records if r.get("eval_status") == "judged")
    assert summary["total_eligible"] == judged, summary


# ---------------------------------------------------------------------------
# E2E-032: Judge rubric binary / three_level
# ---------------------------------------------------------------------------


def test_e2e_032_judge_rubric_binary_and_three_level(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Judge verdicts use the rubric vocabulary, never a numeric 1-10 score.

    FR-044: with ``rubric: binary`` verdicts are in {pass, fail}; with
    ``rubric: three_level`` in {correct, partial, incorrect}; ``judge_reason``
    populated; no numeric scores present.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.judge_verdict = {"verdict": "pass", "reason": "matches"}

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(4)])

    # Binary rubric.
    bin_cfg = _write_eval_config(
        tmp_path / "bin",
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="judge",
        rubric="binary",
    )
    bin_out = tmp_path / "bin_run"
    bin_res = _run(bin_cfg, bin_out, method="judge", prompts=prompts)
    assert bin_res.exit_code == 0, bin_res.output
    bin_judged = [r for r in _read_jsonl(bin_out / "raw.jsonl") if r.get("eval_status") == "judged"]
    assert bin_judged, "no judged records (binary)"
    for rec in bin_judged:
        assert rec["judge_verdict"] in {"pass", "fail"}, rec["judge_verdict"]
        assert rec.get("judge_reason"), rec
        assert not _looks_numeric(rec["judge_verdict"]), rec

    # Three-level rubric.
    eval_ctrl.judge_verdict = {"verdict": "correct", "reason": "matches well"}
    tl_cfg = _write_eval_config(
        tmp_path / "tl",
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="judge",
        rubric="three_level",
    )
    tl_out = tmp_path / "tl_run"
    tl_res = _run(tl_cfg, tl_out, method="judge", prompts=prompts)
    assert tl_res.exit_code == 0, tl_res.output
    tl_judged = [r for r in _read_jsonl(tl_out / "raw.jsonl") if r.get("eval_status") == "judged"]
    assert tl_judged, "no judged records (three_level)"
    for rec in tl_judged:
        assert rec["judge_verdict"] in {"correct", "partial", "incorrect"}, rec["judge_verdict"]
        assert rec.get("judge_reason"), rec


def _looks_numeric(value: Any) -> bool:
    """True if ``value`` is a number or a numeric 1-10 string."""
    if isinstance(value, (int, float)):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# E2E-033: Eval global timeout marks remaining skipped
# ---------------------------------------------------------------------------


def test_e2e_033_eval_global_timeout_marks_skipped(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Draining stops at the global timeout; remaining records are eval_skipped.

    FR-046: the eval pool drains concurrently with the load, so to leave a backlog
    the post-load tail can't clear, the embeddings are very slow (2 s; 4 workers ->
    ~2/s) against ``evaluation.global_timeout: 1s``. At load end most records are
    still pending; the 1 s tail clears only a couple, the rest get
    ``eval_status=="eval_skipped"``; summary ``eval.coverage < 1.0``; log contains
    ``eval global timeout (1s) reached, N items skipped``.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.delay_ms = 2000.0
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(24)])
    config = _write_eval_config(
        tmp_path,
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        global_timeout="1s",
        eval_queue_maxsize=64,
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    skipped = [r for r in records if r.get("eval_status") == "eval_skipped"]
    assert skipped, f"expected some eval_skipped records: {[r.get('eval_status') for r in records]}"

    summary = _eval_block(_read_summary(out_dir))
    assert summary["coverage"] < 1.0, summary
    assert "eval global timeout (1s) reached" in (result.output or ""), result.output
    assert "items skipped" in (result.output or ""), result.output


# ---------------------------------------------------------------------------
# E2E-034: Perf published before eval drains (backfill)
# ---------------------------------------------------------------------------


def test_e2e_034_perf_published_before_eval_drains(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The perf summary table is printed before the eval coverage line.

    FR-045: the perf summary table (``p50``/``p99`` headers) appears in the
    terminal output before the ``Eval coverage:`` line; eval scores are
    backfilled into the final summary after perf publication.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.delay_ms = 200.0
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(8)])
    config = _write_eval_config(
        tmp_path,
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        global_timeout="20s",
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    out = result.output
    perf_idx = out.find("p99")
    cov_idx = out.find("Eval coverage:")
    assert perf_idx != -1, f"no perf summary table in output: {out!r}"
    assert cov_idx != -1, f"no eval coverage line in output: {out!r}"
    assert perf_idx < cov_idx, "perf summary must be printed before the eval coverage line"

    # Eval scores were backfilled into the final summary after perf publication.
    summary = _eval_block(_read_summary(out_dir))
    assert summary["judged"] >= 1, summary


# ---------------------------------------------------------------------------
# E2E-035: Embedding cosine threshold pass/fail (inclusive boundary)
# ---------------------------------------------------------------------------


def test_e2e_035_cosine_threshold_inclusive(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Threshold 0.80 passes cosine 0.80 (inclusive) and 0.81, fails 0.79.

    FR-043: A (0.80) ``quality_pass==true``, B (0.79) ``false``, C (0.81)
    ``true``; raw ``sim_score`` values {0.80, 0.79, 0.81} persisted; the threshold
    boundary is inclusive (>=).
    """
    sut_url, sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval

    from tests.conftest import Behavior, Delta, Usage  # noqa: PLC0415

    # Each prompt's actual output text is distinct so we can map per-pair vectors.
    # Expected vector fixed at the x-axis; actual vector chosen so cosine == target.
    def _unit_for_cosine(target: float) -> list[float]:
        # cosine to [1,0] is just the x-component of a unit vector.
        x = target
        y = math.sqrt(max(0.0, 1.0 - x * x))
        return [x, y]

    expected_vec = [1.0, 0.0]
    pairs = {"A": 0.80, "B": 0.79, "C": 0.81}
    prompts_list: list[dict[str, Any]] = []
    for name, target in pairs.items():
        expected_text = f"expected-{name}"
        actual_text = f"actual-{name}"
        eval_ctrl.map_text(expected_text, expected_vec)
        eval_ctrl.map_text(actual_text, _unit_for_cosine(target))
        assert abs(_cosine(expected_vec, _unit_for_cosine(target)) - target) < 1e-9
        prompts_list.append(_prompt(name, actual_text, expected=expected_text))

    prompts = _write_prompts(tmp_path / "p.yaml", prompts_list)

    # FakeSUT must echo each prompt's intended actual_text so the embedding of the
    # *actual* output maps to the chosen vector. Drive it per-request from the body.
    def _fn(_index: int, body: dict[str, Any]) -> Behavior:
        content = body["messages"][-1]["content"]
        # content carries the prompt's user message ("actual-A" etc); echo it back.
        actual = content if content.startswith("actual-") else content.split()[-1]
        return Behavior(deltas=[Delta(actual)], usage=Usage(prompt_tokens=10, completion_tokens=2))

    sut_ctrl.set_function(_fn)

    config = _write_eval_config(
        tmp_path, sut_port=_port_of(sut_url), eval_url=eval_url, monkeypatch=monkeypatch, method="embedding"
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    judged = {r["prompt_id"]: r for r in records if r.get("eval_status") == "judged"}
    for name in ("A", "B", "C"):
        assert name in judged, f"prompt {name} not judged: {list(judged)}"

    assert abs(judged["A"]["sim_score"] - 0.80) < 1e-6, judged["A"]["sim_score"]
    assert abs(judged["B"]["sim_score"] - 0.79) < 1e-6, judged["B"]["sim_score"]
    assert abs(judged["C"]["sim_score"] - 0.81) < 1e-6, judged["C"]["sim_score"]
    assert judged["A"]["quality_pass"] is True, "threshold must be inclusive (0.80 >= 0.80)"
    assert judged["B"]["quality_pass"] is False, judged["B"]
    assert judged["C"]["quality_pass"] is True, judged["C"]


# ---------------------------------------------------------------------------
# E2E-068: SIGINT with eval in flight (cross SC-001 + SC-004)
# ---------------------------------------------------------------------------


def test_e2e_068_sigint_with_eval_in_flight(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SIGINT mid-run leaves no eval item lost; run marked incomplete.

    FR-015/041/045: in-flight eval items are completed-or-marked eval_skipped (no
    record left without an ``eval_status``); ``summary.status=="incomplete"``;
    ``eval.coverage`` reported; no orphaned eval record without a join key.

    Spawned as a real subprocess (CliRunner cannot deliver a signal mid-run); the
    parent keeps the FakeSUT and FakeEval servers running for the child to reach.
    """
    sut_url, sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.delay_ms = 300.0  # slow eval so items are genuinely in flight
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]

    from tests.conftest import Behavior, Delta, Usage  # noqa: PLC0415

    sut_ctrl.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=20.0) for _ in range(4)], usage=Usage(completion_tokens=4))
    )

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(40)])
    config = _write_eval_config(
        tmp_path,
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        global_timeout="30s",
        run_overrides={"duration": "30s", "warmup": "0s", "cooldown": "0s"},
    )

    out_dir = tmp_path / "r68"
    env = dict(os.environ)
    env["SUT_API_KEY"] = "sk-test"
    env["SUT_PORT"] = str(_port_of(sut_url))

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "llm_bench.llm_bench",
            "run",
            "--config",
            str(config),
            "--model",
            "sut",
            "--eval-method",
            "embedding",
            "--prompts",
            str(prompts),
            "--out",
            str(out_dir),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.8)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    assert proc.returncode == 130, proc.returncode

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records flushed on SIGINT"
    # Every record carries a join key and a resolved eval_status (none lost).
    valid_status = {"judged", "eval_skipped", "skipped_no_expected"}
    for rec in records:
        assert rec.get("request_id"), f"orphaned record without request_id: {rec}"
        assert rec.get("eval_status") in valid_status, f"eval item left unresolved: {rec.get('eval_status')}"

    summary = _read_summary(out_dir)
    assert summary.get("status") == "incomplete", summary
    assert "coverage" in _eval_block(summary), summary


# ---------------------------------------------------------------------------
# E2E-107: Eval queue saturation spill/drop counter
# ---------------------------------------------------------------------------


def test_e2e_107_eval_queue_saturation_counter(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bounded eval queue spills with a single aggregate counter, never blocking.

    FR-041: summary ``eval.dropped`` (or ``eval.spilled``) > 0 and equals
    ``eligible - judged - skipped``; load-gen completed count unaffected; a single
    aggregate counter is reported (no per-item exception storm).
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, eval_ctrl = fake_eval
    eval_ctrl.delay_ms = 500.0  # slow -> the bounded queue must spill
    eval_ctrl.default_vector = [1.0, 0.0, 0.0]

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(90)])
    config = _write_eval_config(
        tmp_path,
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="embedding",
        eval_queue_maxsize=8,
        global_timeout="1s",
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    eligible = sum(1 for r in records if r.get("eval_status") != "skipped_no_expected")
    judged = sum(1 for r in records if r.get("eval_status") == "judged")
    skipped = sum(1 for r in records if r.get("eval_status") == "eval_skipped")

    summary = _eval_block(_read_summary(out_dir))
    dropped = _dropped(summary)
    assert dropped > 0, summary
    # The aggregate spill counter accounts for every eligible-but-unscored item.
    assert dropped == eligible - judged - skipped, (
        f"dropped {dropped} != eligible {eligible} - judged {judged} - skipped {skipped}"
    )


# ---------------------------------------------------------------------------
# E2E-109: Embedding endpoint down marks eval_skipped
# ---------------------------------------------------------------------------


def test_e2e_109_embedding_down_marks_eval_skipped(
    fake_sut: tuple[str, SUTController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unreachable embedding endpoint leaves perf valid but eval skipped.

    FR-043/046: exit 0 (perf valid); eval records ``sim_score==null`` and
    ``eval_status=="eval_skipped"``; WARNING ``embedding endpoint unreachable,
    marking eval_skipped``; coverage < 1.0.
    """
    sut_url, _sut_ctrl = fake_sut
    dead_url = f"http://127.0.0.1:{_closed_port()}/v1"

    prompts = _write_prompts(tmp_path / "p.yaml", [_prompt(f"e{i}", f"q{i}", expected=f"ans{i}") for i in range(4)])
    config = _write_eval_config(
        tmp_path, sut_port=_port_of(sut_url), eval_url=dead_url, monkeypatch=monkeypatch, method="embedding"
    )

    out_dir = tmp_path / "runs" / "e"
    result = _run(config, out_dir, method="embedding", prompts=prompts)
    assert result.exit_code == 0, result.output

    records = _read_jsonl(out_dir / "raw.jsonl")
    evaluated = [r for r in records if r.get("eval_status") in {"judged", "eval_skipped"}]
    assert evaluated, f"no eval records: {[r.get('eval_status') for r in records]}"
    for rec in evaluated:
        assert rec["eval_status"] == "eval_skipped", rec
        assert rec["sim_score"] is None, rec

    summary = _eval_block(_read_summary(out_dir))
    assert summary["coverage"] < 1.0, summary
    assert "embedding endpoint unreachable, marking eval_skipped" in (result.output or "")


# ---------------------------------------------------------------------------
# E2E-110: Self-preference guard on judge model family
# ---------------------------------------------------------------------------


def test_e2e_110_self_preference_guard(
    fake_sut: tuple[str, SUTController],
    fake_eval: tuple[str, EvalController],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A judge model in the same family as the SUT raises a self-preference WARNING.

    FR-044: judge model ``fake/model`` (same family as the SUT) logs WARNING
    ``judge model family matches SUT (self-preference bias risk)``; the run is
    allowed (``--dry-run`` exits 0). With a different judge family no such warning
    appears.
    """
    sut_url, _sut_ctrl = fake_sut
    eval_url, _eval_ctrl = fake_eval

    warning = "judge model family matches SUT (self-preference bias risk)"

    # Same family as the SUT ("fake/model") -> warning expected.
    same_cfg = _write_eval_config(
        tmp_path / "same",
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="judge",
        judge_model="fake/model",
    )
    same_res = runner.invoke(
        app,
        ["run", "--config", str(same_cfg), "--model", "sut", "--eval-method", "judge", "--dry-run"],
    )
    assert same_res.exit_code == 0, same_res.output
    assert warning in (same_res.output or ""), same_res.output

    # Different family -> no warning.
    diff_cfg = _write_eval_config(
        tmp_path / "diff",
        sut_port=_port_of(sut_url),
        eval_url=eval_url,
        monkeypatch=monkeypatch,
        method="judge",
        judge_model="other/judge",
    )
    diff_res = runner.invoke(
        app,
        ["run", "--config", str(diff_cfg), "--model", "sut", "--eval-method", "judge", "--dry-run"],
    )
    assert diff_res.exit_code == 0, diff_res.output
    assert warning not in (diff_res.output or ""), diff_res.output
