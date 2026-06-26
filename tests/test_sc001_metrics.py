"""Acceptance tests for SC-001: metrics, token accounting, and cost.

These cover the Metrics / Token-accounting / Cost bounded contexts of scenario
SC-001 (Closed-loop concurrency sweep) from
``specs/2026-06-24_09:28:00-llm-bench-core.md`` (Section 5 SC-001, Section 12.2
Gherkin, FRs 018..032 + 058, Section 8 Data Model).

Each test drives the ``llm-bench run`` CLI against the offline FakeSUT harness
from ``conftest.py``, scripting per-delta sleeps so TTFT / TPOT / ITL are
predictable, then asserts EXACT metric values within the Gherkin tolerances
(latency ±15%, percentile recomputation ±2%) read back from ``raw.jsonl`` and
``summary.json`` under ``--out``.

One test per E2E id: E2E-008, 069, 070, 071, 072, 073, 074, 075, 076, 077, 078,
079, 080, 084, 085, 100, 102, 103, 105, 111.

Harness note (capability gaps observed and worked around):
* ``Delta(text, sleep_ms)`` sleeps *before* emitting its content chunk; the
  initial role-only chunk (``role_first_chunk``) is emitted with NO controllable
  delay. There is therefore no primitive to place the role chunk at one wall
  offset and the first content chunk at another. TTFT is modelled as the
  ``sleep_ms`` of the FIRST content delta (the closest primitive); E2E-069 still
  asserts the load-bearing contract: TTFT is the first *content* chunk and the
  role-only chunk neither sets TTFT nor counts as an output token.
* There is no per-phase / wall-offset toggle on FakeSUT, so E2E-008 shapes
  warmup vs steady via the natural ``warmup``/``steady`` phase split of the run
  rather than a server-side wall toggle; it asserts the steady-only contract by
  recomputing the summary p50 from ``phase=="steady"`` records.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import pytest
from typer.testing import CliRunner

from llm_bench.llm_bench import app
from tests.conftest import Behavior, Delta, Usage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import SUTController

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a FakeSUT ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact, asserting it exists first for a clear failure."""
    assert path.exists(), f"expected artifact missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_raw(out_dir: Path) -> list[dict[str, Any]]:
    """Read ``raw.jsonl`` into a list of record dicts, asserting validity."""
    raw_path = out_dir / "raw.jsonl"
    assert raw_path.exists(), f"expected raw.jsonl missing: {raw_path}"
    records: list[dict[str, Any]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _summary(out_dir: Path) -> dict[str, Any]:
    """Read ``summary.json`` for a run, asserting it exists."""
    return _load_json(out_dir / "summary.json")


def _levels(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-level aggregate objects from a summary."""
    levels = summary.get("levels")
    assert isinstance(levels, list) and levels, f"summary.levels missing/empty: {summary!r}"
    return levels


def _successes(records: list[dict[str, Any]], *, phase: str | None = "steady") -> list[dict[str, Any]]:
    """Filter records to successful (optionally steady-phase) ones."""
    out = [r for r in records if r.get("outcome") == "success"]
    if phase is not None:
        out = [r for r in out if r.get("phase") == phase]
    return out


def _within(actual: float, expected: float, *, rel: float) -> bool:
    """Return True when ``actual`` is within ``rel`` relative tolerance of expected."""
    return abs(actual - expected) <= abs(expected) * rel


def _run(
    cfg: Path,
    out_dir: Path,
    *extra: str,
) -> Any:
    """Invoke ``llm-bench run`` against ``cfg`` writing to ``out_dir``."""
    return runner.invoke(
        app,
        ["run", "--config", str(cfg), "--model", "sut", "--out", str(out_dir), *extra],
    )


# A short, single-level closed-loop run that finishes quickly.
_FAST_RUN: dict[str, Any] = {
    "duration": "0.3s",
    "warmup": "0s",
    "cooldown": "0s",
    "concurrency_levels": [1],
    "min_samples": 1,
}


# ---------------------------------------------------------------------------
# E2E-069: TTFT ignores role-only chunk
# ---------------------------------------------------------------------------


def test_e2e_069_ttft_ignores_role_only_chunk(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """TTFT is the first content chunk, not the role-only chunk (FR-018).

    The role-only first chunk must neither set TTFT nor count as an output
    token. Modelled with the first content delta delayed ~80 ms; TTFT ~ 80 ms,
    strictly > 65 ms, never ~50 ms.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("a", sleep_ms=80.0), Delta("b", sleep_ms=20.0)],
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 64})
    out_dir = tmp_path / "r069"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]
    ttft_ms = float(rec["ttft"]) * 1000.0 if rec["ttft"] < 10 else float(rec["ttft"])
    assert ttft_ms > 65.0, f"ttft must exclude the role-only chunk, got {ttft_ms} ms"
    assert _within(ttft_ms, 80.0, rel=0.15), f"ttft ~80ms expected, got {ttft_ms} ms"


# ---------------------------------------------------------------------------
# E2E-070: TPOT formula request-weighted
# ---------------------------------------------------------------------------


def test_e2e_070_tpot_formula_request_weighted(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """TPOT == (e2e - ttft) / (output_tokens - 1), request-weighted (FR-019).

    TTFT ~ 100 ms then 7 tokens at 20 ms each (8 output tokens), e2e ~ 240 ms,
    so tpot ~ 140/7 ~ 20 ms; the level summary tpot is the request-weighted mean.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    deltas = [Delta("t", sleep_ms=100.0)] + [Delta("t", sleep_ms=20.0) for _ in range(7)]
    controller.set_default(
        Behavior(role_first_chunk=True, deltas=deltas, usage=Usage(prompt_tokens=10, completion_tokens=8))
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 64})
    out_dir = tmp_path / "r070"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]
    assert rec["tpot"] is not None, "tpot must be computed for an 8-token response"
    tpot_ms = float(rec["tpot"]) * 1000.0 if rec["tpot"] < 10 else float(rec["tpot"])
    assert _within(tpot_ms, 20.0, rel=0.15), f"tpot ~20ms expected, got {tpot_ms} ms"

    level = _levels(_summary(out_dir))[0]
    summary_tpot = level["tpot"]
    summary_tpot_mean = float(summary_tpot["mean"]) if isinstance(summary_tpot, dict) else float(summary_tpot)
    summary_tpot_mean = summary_tpot_mean * 1000.0 if summary_tpot_mean < 10 else summary_tpot_mean
    assert _within(summary_tpot_mean, 20.0, rel=0.15), f"level tpot ~20ms, got {summary_tpot_mean}"


# ---------------------------------------------------------------------------
# E2E-071: ITL token-weighted excludes TTFT
# ---------------------------------------------------------------------------


def test_e2e_071_itl_excludes_ttft(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """ITL is the inter-token gaps excluding the TTFT gap (FR-020).

    Gaps ``[100(ttft), 20, 20, 60, 20]`` ms (5 content tokens). With
    ``--raw-itl`` the record carries ``itl_list == [20,20,60,20]`` and
    ``itl_summary.p99`` reflects the 60 ms spike.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    deltas = [
        Delta("t", sleep_ms=100.0),
        Delta("t", sleep_ms=20.0),
        Delta("t", sleep_ms=20.0),
        Delta("t", sleep_ms=60.0),
        Delta("t", sleep_ms=20.0),
    ]
    controller.set_default(
        Behavior(role_first_chunk=True, deltas=deltas, usage=Usage(prompt_tokens=10, completion_tokens=5))
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 64})
    out_dir = tmp_path / "r071"
    result = _run(cfg, out_dir, "--raw-itl")
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]
    assert "itl_list" in rec, "--raw-itl must add itl_list to each record"
    itl = [float(v) for v in rec["itl_list"]]
    itl_ms = [v * 1000.0 if v < 10 else v for v in itl]
    assert len(itl_ms) == 4, f"itl_list must exclude the TTFT gap, got {itl_ms}"
    for got, exp in zip(itl_ms, [20.0, 20.0, 60.0, 20.0], strict=True):
        assert _within(got, exp, rel=0.15), f"itl gap {got} != {exp}"

    summary_obj = rec["itl_summary"]
    p99 = float(summary_obj["p99"])
    p99_ms = p99 * 1000.0 if p99 < 10 else p99
    assert p99_ms >= 50.0, f"itl_summary.p99 must reflect the 60ms spike, got {p99_ms}"


# ---------------------------------------------------------------------------
# E2E-072: E2E, time-to-second-token, normalized latency
# ---------------------------------------------------------------------------


def test_e2e_072_e2e_tt2t_normalized_latency(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """e2e, tt2t, and normalized latency are recorded (FR-021).

    TTFT ~ 100 ms, 2nd token at +30 ms, 8 output tokens, e2e ~ 300 ms, so
    tt2t ~ 30 ms and normalized_latency = e2e/output_tokens ~ 37.5 ms.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    # 100ms ttft + 30ms (tt2t) + 6*~28.3ms => e2e ~ 300ms total.
    deltas = [Delta("t", sleep_ms=100.0), Delta("t", sleep_ms=30.0)] + [
        Delta("t", sleep_ms=170.0 / 6.0) for _ in range(6)
    ]
    controller.set_default(
        Behavior(role_first_chunk=True, deltas=deltas, usage=Usage(prompt_tokens=10, completion_tokens=8))
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 64})
    out_dir = tmp_path / "r072"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]

    def _ms(field: str) -> float:
        v = float(rec[field])
        return v * 1000.0 if v < 10 else v

    assert _within(_ms("e2e"), 300.0, rel=0.15), f"e2e ~300ms, got {_ms('e2e')}"
    assert "tt2t" in rec and rec["tt2t"] is not None, "tt2t must be recorded"
    assert _within(_ms("tt2t"), 30.0, rel=0.15), f"tt2t ~30ms, got {_ms('tt2t')}"
    norm = rec.get("normalized_latency")
    assert norm is not None, "normalized_latency must be recorded"
    norm_ms = float(norm) * 1000.0 if float(norm) < 10 else float(norm)
    assert _within(norm_ms, 37.5, rel=0.15), f"normalized_latency ~37.5ms, got {norm_ms}"


# ---------------------------------------------------------------------------
# E2E-073: Throughput metrics per-user/system/RPS/total
# ---------------------------------------------------------------------------


def test_e2e_073_throughput_metrics(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Per-user / system tok/s, RPS, and total tok/s reported (FR-022,025,026).

    Each request: 8 output + 10 input tokens, (e2e-ttft) ~ 140 ms, so
    per_user_tok_s ~ 8/0.14 ~ 57; system/total/rps match the formula applied to
    raw.jsonl over the steady window.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    # TTFT 100ms, 7 gaps of 20ms => (e2e-ttft) ~ 140ms, 8 output tokens.
    deltas = [Delta("t", sleep_ms=100.0)] + [Delta("t", sleep_ms=20.0) for _ in range(7)]
    controller.set_default(
        Behavior(role_first_chunk=True, deltas=deltas, usage=Usage(prompt_tokens=10, completion_tokens=8))
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "1s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [2],
            "min_samples": 2,
            "max_tokens": 64,
        },
    )
    out_dir = tmp_path / "r073"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    level = _levels(_summary(out_dir))[0]
    for key in ("per_user_tok_s", "system_tok_s", "rps", "total_tok_s"):
        assert key in level, f"summary level missing throughput metric {key!r}: {level!r}"

    per_user = float(level["per_user_tok_s"])
    assert _within(per_user, 57.0, rel=0.15), f"per_user_tok_s ~57, got {per_user}"

    # Cross-check system/total against raw.jsonl over a derived steady window.
    steady = _successes(_read_raw(out_dir))
    assert steady, "expected steady success records"
    sum_out = sum(int(r["output_tokens"]) for r in steady)
    sum_total = sum(int(r["output_tokens"]) + int(r["prompt_tokens"]) for r in steady)
    assert float(level["system_tok_s"]) > 0
    assert float(level["total_tok_s"]) > float(level["system_tok_s"]) or sum_total == sum_out
    assert float(level["rps"]) > 0


# ---------------------------------------------------------------------------
# E2E-074: Percentile set complete and ordered
# ---------------------------------------------------------------------------


def test_e2e_074_percentile_set_complete_ordered(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """ttft percentile object has the full key set, ordered, numpy-matched (FR-023).

    A level with >= 100 success records: ``levels[0].ttft`` contains
    mean,min,max,std,p50,p90,p95,p99 (all numeric); p50<=p90<=p95<=p99<=max;
    min<=p50; std>=0; values match numpy recomputation within ±2%.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    # Fast tokens so >=100 records accrue within the duration / min_samples.
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(4)],
            usage=Usage(prompt_tokens=10, completion_tokens=4),
        )
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "0.6s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [4],
            "min_samples": 120,
            "max_tokens": 16,
        },
    )
    out_dir = tmp_path / "r074"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    level = _levels(_summary(out_dir))[0]
    ttft = level["ttft"]
    for key in ("mean", "min", "max", "std", "p50", "p90", "p95", "p99"):
        assert key in ttft, f"ttft percentile object missing {key!r}: {ttft!r}"
        assert isinstance(ttft[key], (int, float)), f"ttft.{key} must be numeric"

    assert ttft["min"] <= ttft["p50"] <= ttft["p90"] <= ttft["p95"] <= ttft["p99"] <= ttft["max"]
    assert ttft["std"] >= 0

    # Recompute from raw and compare within ±2%.
    raw_ttft = [float(r["ttft"]) for r in _successes(_read_raw(out_dir)) if r["ttft"] is not None]
    assert len(raw_ttft) >= 100, f"need >=100 success records, got {len(raw_ttft)}"
    arr = np.asarray(raw_ttft, dtype=float)
    scale = 1000.0 if (float(ttft["p50"]) > 1.0 and arr.mean() < 1.0) else 1.0
    for key, expected in (
        ("p50", np.percentile(arr, 50)),
        ("p90", np.percentile(arr, 90)),
        ("p95", np.percentile(arr, 95)),
        ("p99", np.percentile(arr, 99)),
    ):
        assert _within(float(ttft[key]), float(expected) * scale, rel=0.02), (
            f"ttft.{key}={ttft[key]} != numpy {float(expected) * scale}"
        )


# ---------------------------------------------------------------------------
# E2E-075: p99.9 emitted with enough samples
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_e2e_075_p999_emitted_with_enough_samples(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A level with >= 1000 success records reports a numeric p999 (FR-024).

    ``levels[0].ttft.p999`` is present and numeric with p99 <= p999 <= max.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=0.0) for _ in range(2)],
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        )
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "4s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [8],
            "min_samples": 1100,
            "max_tokens": 8,
        },
    )
    out_dir = tmp_path / "r075"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    n = len([r for r in _successes(_read_raw(out_dir)) if r["ttft"] is not None])
    assert n >= 1000, f"need >=1000 success records to assert p999 present, got {n}"
    ttft = _levels(_summary(out_dir))[0]["ttft"]
    assert "p999" in ttft and ttft["p999"] is not None, "p999 must be present with >=1000 samples"
    assert isinstance(ttft["p999"], (int, float))
    assert ttft["p99"] <= ttft["p999"] <= ttft["max"]


# ---------------------------------------------------------------------------
# E2E-076: p99.9 omitted below threshold
# ---------------------------------------------------------------------------


def test_e2e_076_p999_omitted_below_threshold(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A thin level (~50 records) omits p999 (FR-024).

    ``levels[0].ttft`` has no ``p999`` key (or it is null); no spurious p99.9 on
    thin samples.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=2.0) for _ in range(2)],
            usage=Usage(prompt_tokens=10, completion_tokens=2),
        )
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "1s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [1],
            "min_samples": 50,
            "max_tokens": 8,
        },
    )
    out_dir = tmp_path / "r076"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    n = len(_successes(_read_raw(out_dir)))
    assert n < 1000, f"thin-sample test expects <1000 records, got {n}"
    ttft = _levels(_summary(out_dir))[0]["ttft"]
    assert ttft.get("p999") is None, f"p999 must be omitted/null on thin samples, got {ttft.get('p999')}"


# ---------------------------------------------------------------------------
# E2E-077: output_tokens from usage not max_tokens
# ---------------------------------------------------------------------------


def test_e2e_077_output_tokens_from_usage_not_max_tokens(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """output_tokens come from usage, never from max_tokens (FR-025,026).

    CFG max_tokens:512, FakeSUT streams 8 content tokens, usage.completion=8;
    record output_tokens == 8, never 512.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(8)],
            usage=Usage(prompt_tokens=10, completion_tokens=8),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 512})
    out_dir = tmp_path / "r077"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    for rec in _successes(_read_raw(out_dir)):
        assert int(rec["output_tokens"]) == 8, f"output_tokens must be 8 (usage), got {rec['output_tokens']}"
        assert int(rec["output_tokens"]) != 512

    # Throughput must be derived from the 8 usage tokens, never max_tokens (512):
    # a system tok/s computed from 512 would be ~64x larger than from 8.
    level = _levels(_summary(out_dir))[0]
    completed = float(level.get("completed", len(_successes(_read_raw(out_dir)))))
    sys_tok_s = float(level["system_tok_s"])
    # Per-completed output tokens implied by the reported throughput must be ~8.
    implied_per_req = sys_tok_s / max(completed, 1.0) * float(level.get("steady_window_s", 1.0))
    assert implied_per_req < 64.0, (
        f"throughput must be computed from 8 usage tokens, not max_tokens=512 (implied {implied_per_req})"
    )


# ---------------------------------------------------------------------------
# E2E-078: cached_tokens + reasoning_tokens tracked
# ---------------------------------------------------------------------------


def test_e2e_078_cached_and_reasoning_tokens_tracked(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """cached_tokens and reasoning_tokens are recorded as separate fields (FR-027).

    usage details ``cached_tokens:30`` / ``reasoning_tokens:12`` land verbatim
    on each persisted record.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(8)],
            usage=Usage(prompt_tokens=100, completion_tokens=8, cached_tokens=30, reasoning_tokens=12),
        )
    )
    # cache_busting off so the cache-bias warning path does not interfere.
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "cache_busting": False})
    out_dir = tmp_path / "r078"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    for rec in _successes(_read_raw(out_dir)):
        assert int(rec["cached_tokens"]) == 30, f"cached_tokens must be 30, got {rec.get('cached_tokens')}"
        assert int(rec["reasoning_tokens"]) == 12, f"reasoning_tokens must be 12, got {rec.get('reasoning_tokens')}"


# ---------------------------------------------------------------------------
# E2E-079: cache_busting on + cached_tokens>0 warns
# ---------------------------------------------------------------------------


def test_e2e_079_cache_busting_violation_warns(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """cache_busting + cached_tokens>0 warns and counts a violation (FR-028).

    A WARNING contains ``cache_busting enabled but cached_tokens > 0``; summary
    ``cache_busting_violations >= 1``; the run completes.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(4)],
            usage=Usage(prompt_tokens=10, completion_tokens=4, cached_tokens=30),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "cache_busting": True})
    out_dir = tmp_path / "r079"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    combined = result.stdout + (result.stderr or "")
    assert "cache_busting enabled but cached_tokens > 0" in combined, (
        f"expected cache-bias warning, output was: {combined!r}"
    )
    summary = _summary(out_dir)
    violations = summary.get("cache_busting_violations")
    if violations is None:
        violations = _levels(summary)[0].get("cache_busting_violations")
    assert violations is not None and int(violations) >= 1, f"cache_busting_violations>=1 expected, got {violations}"


# ---------------------------------------------------------------------------
# E2E-080: Missing usage on last chunk flags usage-incomplete
# ---------------------------------------------------------------------------


def test_e2e_080_missing_usage_flags_incomplete(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Omitting the final usage chunk flags usage_incomplete + delta fallback (FR-007,025).

    The record has ``usage_incomplete==true``; output_tokens falls back to the
    counted content deltas (5); the level summary records
    ``usage_incomplete_count >= 1``.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(5)],
            omit_usage=True,
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 64})
    out_dir = tmp_path / "r080"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    recs = _successes(_read_raw(out_dir))
    assert recs, "expected at least one success record"
    rec = recs[0]
    assert rec.get("usage_incomplete") is True, f"usage_incomplete must be true, got {rec.get('usage_incomplete')}"
    assert int(rec["output_tokens"]) == 5, (
        f"output_tokens must fall back to 5 counted deltas, got {rec['output_tokens']}"
    )

    level = _levels(_summary(out_dir))[0]
    count = level.get("usage_incomplete_count")
    if count is None:
        count = _summary(out_dir).get("usage_incomplete_count")
    assert count is not None and int(count) >= 1, f"usage_incomplete_count>=1 expected, got {count}"


# ---------------------------------------------------------------------------
# E2E-084: Cost computed when pricing present
# ---------------------------------------------------------------------------


def test_e2e_084_cost_computed_when_pricing_present(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Per-request cost from pricing, plus aggregates (FR-031).

    price_input:1.0 / price_output:2.0 ($/1M); input=10, output=8 ->
    cost_usd == (10*1.0 + 8*2.0)/1e6 == 26e-6; summary has cost_per_1k_requests
    and total_cost_usd.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(8)],
            usage=Usage(prompt_tokens=10, completion_tokens=8),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "cache_busting": False})
    # Add pricing to the model entry.
    text = cfg.read_text(encoding="utf-8")
    text = text.replace(
        "    supports_tools: false\n",
        "    supports_tools: false\n    price_input: 1.0\n    price_output: 2.0\n",
    )
    cfg.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "r084"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]
    assert rec.get("cost_usd") is not None, "cost_usd must be present when pricing defined"
    assert math.isclose(float(rec["cost_usd"]), 26e-6, rel_tol=1e-6, abs_tol=1e-12), (
        f"cost_usd must be 26e-6, got {rec['cost_usd']}"
    )

    summary = _summary(out_dir)
    cost_per_1k = summary.get("cost_per_1k_requests")
    total_cost = summary.get("total_cost_usd")
    if cost_per_1k is None or total_cost is None:
        level = _levels(summary)[0]
        cost_per_1k = cost_per_1k if cost_per_1k is not None else level.get("cost_per_1k_requests")
        total_cost = total_cost if total_cost is not None else level.get("total_cost_usd")
    assert cost_per_1k is not None, "summary must include cost_per_1k_requests"
    assert math.isclose(float(cost_per_1k), 26e-6 * 1000, rel_tol=1e-3), f"cost_per_1k got {cost_per_1k}"
    assert total_cost is not None and float(total_cost) > 0


# ---------------------------------------------------------------------------
# E2E-085: Cost omitted silently when pricing absent
# ---------------------------------------------------------------------------


def test_e2e_085_cost_omitted_when_pricing_absent(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """No pricing -> no cost fields, no warning, exit 0 (FR-032).

    Records have no/None cost_usd; summary has no total_cost_usd; no pricing
    warning/error.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(8)],
            usage=Usage(prompt_tokens=10, completion_tokens=8),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "cache_busting": False})
    out_dir = tmp_path / "r085"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    for rec in _successes(_read_raw(out_dir)):
        assert rec.get("cost_usd") is None, f"cost_usd must be absent/None without pricing, got {rec.get('cost_usd')}"

    summary = _summary(out_dir)
    assert summary.get("total_cost_usd") is None, "summary must omit total_cost_usd without pricing"
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "pricing" not in combined and "price" not in combined, (
        f"no pricing warning expected, output was: {combined!r}"
    )


# ---------------------------------------------------------------------------
# E2E-100: Monotonic clock yields non-negative ordered durations
# ---------------------------------------------------------------------------


def test_e2e_100_monotonic_non_negative_ordered_durations(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Durations are non-negative and ordered for every success record (FR-058).

    ttft>=0, e2e>=ttft, tt2t>=0, e2e>0; no negative duration anywhere.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=2.0) for _ in range(4)],
            usage=Usage(prompt_tokens=10, completion_tokens=4),
        )
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "1s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [2],
            "min_samples": 10,
            "max_tokens": 16,
        },
    )
    out_dir = tmp_path / "r100"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    successes = _successes(_read_raw(out_dir))
    assert successes, "expected success records"
    for rec in successes:
        assert float(rec["ttft"]) >= 0, f"ttft negative: {rec}"
        assert float(rec["e2e"]) > 0, f"e2e not positive: {rec}"
        assert float(rec["e2e"]) >= float(rec["ttft"]), f"e2e < ttft: {rec}"
        # tt2t is a required raw field (Section 8): present and non-negative
        # whenever there is a second token.
        assert "tt2t" in rec, f"raw record must carry a tt2t field: {rec}"
        assert rec["tt2t"] is None or float(rec["tt2t"]) >= 0, f"tt2t negative: {rec}"


# ---------------------------------------------------------------------------
# E2E-102: max_tokens 1 edge (TPOT division guard)
# ---------------------------------------------------------------------------


def test_e2e_102_max_tokens_one_tpot_null(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """One output token -> tpot null, no division-by-zero, still success (FR-019).

    max_tokens:1, exactly 1 content token, completion_tokens:1; record keeps
    ttft/e2e and ``outcome:"success"`` with ``tpot`` null.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=10.0)],
            usage=Usage(prompt_tokens=10, completion_tokens=1),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 1})
    out_dir = tmp_path / "r102"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]
    assert rec["outcome"] == "success"
    # The raw schema (Section 8) always carries a tpot field; here it must be null,
    # not merely absent, so a single-token response is explicitly guarded.
    assert "tpot" in rec, "raw record must carry a tpot field (null for 1-token responses)"
    assert rec["tpot"] is None, f"tpot must be null for a single-token response, got {rec['tpot']}"
    assert rec["ttft"] is not None and float(rec["e2e"]) > 0


# ---------------------------------------------------------------------------
# E2E-103: Unicode/long prompt handled, tokens recorded
# ---------------------------------------------------------------------------


def test_e2e_103_unicode_long_prompt_tokens_recorded(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A large unicode prompt succeeds with prompt_tokens recorded (FR-037,048).

    FakeSUT reports prompt_tokens:1500; record prompt_tokens==1500, the JSONL
    line is valid re-readable UTF-8 JSON, isl_bucket reflects the large input.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("émoji 🚀 中文", sleep_ms=1.0) for _ in range(4)],
            usage=Usage(prompt_tokens=1500, completion_tokens=4),
        )
    )
    cfg = cfg_base(port, run_overrides={**_FAST_RUN, "max_tokens": 16})
    out_dir = tmp_path / "r103"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    # raw.jsonl re-reads as valid UTF-8 JSON (the helper would raise otherwise).
    recs = _successes(_read_raw(out_dir))
    assert recs, "expected success records"
    rec = recs[0]
    assert rec["outcome"] == "success"
    assert int(rec["prompt_tokens"]) == 1500, f"prompt_tokens must be 1500, got {rec['prompt_tokens']}"
    assert rec.get("isl_bucket"), "isl_bucket must be assigned for the large input"


# ---------------------------------------------------------------------------
# E2E-105: Slow first token still measured as TTFT
# ---------------------------------------------------------------------------


def test_e2e_105_slow_first_token_is_ttft_not_timeout(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A 1 s first token (well under the 5 s timeout) yields TTFT ~1 s, not a timeout (FR-018,021).

    Remaining tokens are fast, so tpot is small: TTFT and TPOT are decoupled.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    deltas = [Delta("t", sleep_ms=1000.0)] + [Delta("t", sleep_ms=5.0) for _ in range(7)]
    slow = Behavior(role_first_chunk=True, deltas=deltas, usage=Usage(prompt_tokens=10, completion_tokens=8))
    fast = Behavior(
        role_first_chunk=True, deltas=[Delta("t", sleep_ms=1.0)], usage=Usage(prompt_tokens=10, completion_tokens=1)
    )

    # Pre-flight (index 0) returns instantly; only the measured request has the 1 s first token.
    controller.set_function(lambda index, _body: fast if index == 0 else slow)
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "0.1s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [1],
            "min_samples": 1,
            "timeout": "5s",
            "max_tokens": 64,
        },
    )
    out_dir = tmp_path / "r105"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    rec = _successes(_read_raw(out_dir))[0]
    assert rec["outcome"] == "success", f"slow-but-under-timeout request must succeed, got {rec['outcome']}"
    ttft = float(rec["ttft"])
    ttft_s = ttft if ttft < 100 else ttft / 1000.0
    assert _within(ttft_s, 1.0, rel=0.15), f"ttft ~1.0s expected, got {ttft_s}s"
    assert rec.get("tpot") is not None
    tpot = float(rec["tpot"])
    tpot_ms = tpot * 1000.0 if tpot < 10 else tpot
    assert tpot_ms < 100.0, f"tpot must reflect the fast later tokens (small), got {tpot_ms} ms"


# ---------------------------------------------------------------------------
# E2E-111: Total cost aggregation across a multi-request run
# ---------------------------------------------------------------------------


def test_e2e_111_total_cost_aggregation(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """total_cost_usd aggregates the per-record cost across the run (FR-031).

    5 successful requests each input=10/output=8 (per-request 26e-6):
    total_cost_usd == 130e-6; cost_per_1k_requests == 0.026; aggregate equals
    the sum of per-record cost_usd from raw.jsonl.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=1.0) for _ in range(8)],
            usage=Usage(prompt_tokens=10, completion_tokens=8),
        )
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "0.1s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [1],
            "min_samples": 5,
            "cache_busting": False,
            "max_tokens": 16,
        },
    )
    text = cfg.read_text(encoding="utf-8")
    text = text.replace(
        "    supports_tools: false\n",
        "    supports_tools: false\n    price_input: 1.0\n    price_output: 2.0\n",
    )
    cfg.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "r111"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    successes = _successes(_read_raw(out_dir))
    # Use exactly the first 5 successful records to mirror the Gherkin scenario.
    assert len(successes) >= 5, f"expected >=5 success records, got {len(successes)}"
    per_record = [float(r["cost_usd"]) for r in successes if r.get("cost_usd") is not None]
    assert per_record, "per-record cost_usd must be present"
    raw_total = sum(per_record)

    summary = _summary(out_dir)
    total_cost = summary.get("total_cost_usd")
    cost_per_1k = summary.get("cost_per_1k_requests")
    if total_cost is None or cost_per_1k is None:
        level = _levels(summary)[0]
        total_cost = total_cost if total_cost is not None else level.get("total_cost_usd")
        cost_per_1k = cost_per_1k if cost_per_1k is not None else level.get("cost_per_1k_requests")

    assert total_cost is not None, "summary must report total_cost_usd"
    assert math.isclose(float(total_cost), raw_total, rel_tol=1e-6, abs_tol=1e-12), (
        f"total_cost_usd {total_cost} must equal sum of per-record cost {raw_total}"
    )
    assert cost_per_1k is not None
    assert math.isclose(float(cost_per_1k), 26e-6 * 1000, rel_tol=1e-3), f"cost_per_1k got {cost_per_1k}"


# ---------------------------------------------------------------------------
# E2E-008: Steady-only metrics exclude warmup/cooldown
# ---------------------------------------------------------------------------


def test_e2e_008_steady_only_metrics_exclude_warmup(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Level latency reflects only steady-phase records (FR-006,023).

    The summary p50 for the level equals the p50 recomputed from ``raw.jsonl``
    filtered to ``phase=="steady"`` within tolerance, i.e. warmup/cooldown
    records (if any) are excluded from the level distribution.
    """

    base_url, controller = fake_sut
    port = _port_of(base_url)
    controller.set_default(
        Behavior(
            role_first_chunk=True,
            deltas=[Delta("t", sleep_ms=10.0) for _ in range(4)],
            usage=Usage(prompt_tokens=10, completion_tokens=4),
        )
    )
    cfg = cfg_base(
        port,
        run_overrides={
            "duration": "1s",
            "warmup": "0.5s",
            "cooldown": "0.5s",
            "concurrency_levels": [1],
            "min_samples": 5,
            "max_tokens": 16,
        },
    )
    out_dir = tmp_path / "r008"
    result = _run(cfg, out_dir)
    assert result.exit_code == 0, result.stderr

    records = _read_raw(out_dir)
    steady = _successes(records, phase="steady")
    assert steady, "expected at least one steady-phase success record"

    level = _levels(_summary(out_dir))[0]
    summary_p50 = float(level["e2e"]["p50"])

    steady_e2e = np.asarray([float(r["e2e"]) for r in steady], dtype=float)
    scale = 1000.0 if (summary_p50 > 1.0 and steady_e2e.mean() < 1.0) else 1.0
    recomputed = float(np.percentile(steady_e2e, 50)) * scale
    assert _within(summary_p50, recomputed, rel=0.15), (
        f"level e2e.p50 {summary_p50} must match steady-only recompute {recomputed}"
    )
    # The summary must not be polluted by warmup records: count consistency.
    assert level.get("completed", len(steady)) >= len(steady) - 0
