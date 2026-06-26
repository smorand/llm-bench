"""Acceptance tests for SC-002: open-loop Poisson arrivals + goodput.

These cover the Open-loop load generation and Goodput bounded contexts of
scenario SC-002 (spec ``specs/2026-06-24_09:28:00-llm-bench-core.md``, Section 5
SC-002, Section 6 FR-016, FR-017, FR-029, FR-030, and the Section 12.2 Gherkin
for E2E-011, 012, 013, 014, 015, 016, 017, 081, 082, 083).

Each test drives the ``llm-bench`` CLI (``run`` subcommand) through Typer's
:class:`CliRunner` against the offline FakeSUT harness from ``conftest.py`` and
asserts exactly the Gherkin observables: arrival counts and distributions in
``raw.jsonl``, ``mode``/``level_or_rate`` fields, the ``max_outstanding`` guard
(server-observed in-flight cap + warning string + counter), seeded
reproducibility, exit codes / stderr, and the goodput fields in ``summary.json``
and ``resolved_config.json``.

Open-loop is requested entirely through ``cfg_base``'s ``run_overrides`` (the
factory forwards ``mode``/``request_rates``/``burstiness``/``max_outstanding``
into the YAML). The spec's Gherkin shows CLI flags such as
``--mode open --request-rate 20``, but those flags are NOT yet exposed by the
``run`` command; passing them would make these tests fail on an *unknown option*
(a CLI-parse error) rather than on the missing open-loop behavior under test, so
the open-loop knobs are driven via config. The one exception is E2E-082, whose
contract is specifically the ``--slo`` CLI override path, so that flag is passed
literally.

These reference open-loop engine behavior and ``summary.json`` /
``resolved_config.json`` fields not yet implemented, so they are expected to be
RED now. They are written to the spec contract, not the current behavior.

Summary / record field keys chosen from the literal Gherkin (so the implementer
matches them):

* Per-rate summary list: ``summary["rates"]`` (a list, mirroring closed-loop
  ``levels``); each entry carries ``level_or_rate`` and the per-rate metrics.
* ``rates[i]["max_outstanding_events"]`` (FR-017 / E2E-016).
* ``rates[i]["goodput_count"]``, ``rates[i]["goodput_attainment"]``,
  ``rates[i]["goodput_rps"]`` (FR-029 / E2E-081, 083).
* Per-record ``mode`` ("open") and ``level_or_rate`` (the arrival rate).
* ``resolved_config.json`` exposes the active SLO thresholds under
  ``slo["ttft_ms"]`` / ``slo["tpot_ms"]`` / ``slo["e2e_ms"]`` (E2E-082).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import statistics
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from typer.testing import CliRunner

from llm_bench.llm_bench import app
from tests.conftest import Behavior, Delta, Usage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import SUTController

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers (mirrored from the SC-001 test files)
# ---------------------------------------------------------------------------


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a FakeSUT ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def _stderr(result: Any) -> str:
    """Return combined stderr+stdout text, robust to CliRunner stream config."""
    parts = []
    with contextlib.suppress(ValueError, AttributeError):
        parts.append(result.stderr or "")
    parts.append(result.stdout or "")
    return "".join(parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a ``raw.jsonl`` file into a list of record dicts (asserts it exists)."""
    assert path.exists(), f"expected raw.jsonl missing: {path}"
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _read_summary(out_dir: Path) -> dict[str, Any]:
    """Load ``summary.json`` from a run directory (asserts it exists)."""
    path = out_dir / "summary.json"
    assert path.exists(), f"expected summary.json missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _rates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-rate summary list (open-loop analogue of ``levels``)."""
    rates = summary.get("rates")
    assert isinstance(rates, list) and rates, f"summary has no rates: {summary!r}"
    return rates


def _invoke(config: Path, out_dir: Path, *extra: str, model: str = "sut") -> Any:
    """Invoke ``llm-bench run`` writing artifacts to ``out_dir``."""
    return runner.invoke(
        app,
        ["run", "--config", str(config), "--model", model, "--out", str(out_dir), *extra],
    )


def _open_overrides(**extra: Any) -> dict[str, Any]:
    """Build an open-loop ``run_overrides`` dict with sensible defaults."""
    overrides: dict[str, Any] = {
        "mode": "open",
        "request_rates": [20],
        "burstiness": 1.0,
        "duration": "3s",
        "warmup": "0s",
        "cooldown": "0s",
        "max_outstanding": 500,
        "min_samples": 1,
    }
    overrides.update(extra)
    return overrides


# ---------------------------------------------------------------------------
# E2E-011: Open-loop Poisson at fixed rate (FR-016)
# ---------------------------------------------------------------------------


def test_e2e_011_open_loop_poisson_fixed_rate(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Rate 100 over 0.6 s yields ~60 arrivals tagged mode:open, level_or_rate:100.

    FR-016: exit 0; record count for rate 100 is within 100*0.6 ±25% (45 to 75);
    every record has ``mode=="open"`` and ``level_or_rate==100``.
    """
    base_url, controller = fake_sut
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides=_open_overrides(request_rates=[100], burstiness=1.0, duration="0.6s", max_outstanding=500),
    )

    out_dir = tmp_path / "r11"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    rate100 = [r for r in records if r["level_or_rate"] == 100]
    assert 45 <= len(rate100) <= 75, f"arrival count out of band: {len(rate100)}"
    assert all(r["mode"] == "open" for r in records), {r["mode"] for r in records}
    assert all(r["level_or_rate"] == 100 for r in records)


# ---------------------------------------------------------------------------
# E2E-012: Burstiness 1.0 inter-arrival is exponential (FR-016)
# ---------------------------------------------------------------------------


def test_e2e_012_burstiness_exponential_inter_arrival(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Burstiness 1.0 gives exponential inter-arrivals (mean ~1/rate, CV ~1).

    FR-016: sorted ``t_start`` deltas have mean ~0.02 s ±20%; the coefficient of
    variation of inter-arrival gaps is within 0.75 to 1.25 (exponential),
    distinguishing it from a constant-rate schedule (CV ~ 0).
    """
    base_url, controller = fake_sut
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides=_open_overrides(request_rates=[50], burstiness=1.0, duration="1s", seed=42, max_outstanding=500),
    )

    out_dir = tmp_path / "r12"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    starts = sorted(r["t_start"] for r in records)
    assert len(starts) >= 30, f"too few arrivals to characterise: {len(starts)}"
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    mean_gap = statistics.fmean(gaps)
    # Mean inter-arrival ~ 1/rate = 0.02 s, ±20%.
    assert abs(mean_gap - 0.02) <= 0.02 * 0.20, f"mean gap {mean_gap} not ~0.02s"
    # Exponential: CV (std/mean) within [0.75, 1.25]; constant-rate would be ~0.
    cv = statistics.pstdev(gaps) / mean_gap
    assert 0.75 <= cv <= 1.25, f"inter-arrival CV {cv} not exponential"


# ---------------------------------------------------------------------------
# E2E-013: max_outstanding guard halts new arrivals (FR-017)
# ---------------------------------------------------------------------------


def test_e2e_013_max_outstanding_guard_halts_arrivals(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A stalling SUT makes the guard cap server-observed in-flight at 10.

    FR-017: with ``max_outstanding:10`` and every response stalling 10 s before
    the first chunk, the server-observed concurrent in-flight never exceeds 10;
    the log contains ``max_outstanding (10) reached, pausing arrivals``; the
    dispatched count is bounded near 10 (not ~200 = rate*duration).
    """
    base_url, controller = fake_sut

    # Pre-flight (index 0) must succeed; every measured response stalls 5 s before
    # the first byte so requests pile up against the guard.
    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 0:
            return Behavior(deltas=[Delta("x", sleep_ms=1.0)], usage=Usage(completion_tokens=1))
        return Behavior(force_timeout_ms=1_200.0, deltas=[Delta("x")])

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "mode": "open",
            "request_rates": [100],
            "burstiness": 1.0,
            "max_outstanding": 10,
            "duration": "0.6s",
            "warmup": "0s",
            "cooldown": "0s",
            "timeout": "1s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r13"
    result = _invoke(config, out_dir)
    # The run itself completes (exit 0); the guard pauses arrivals, it does not abort.
    assert result.exit_code == 0, _stderr(result)

    # Server never saw more than max_outstanding concurrent in-flight.
    assert controller.max_in_flight <= 10, controller.max_in_flight
    text = _stderr(result)
    assert "max_outstanding (10) reached, pausing arrivals" in text, text
    # Dispatched count bounded near the cap, not the offered ~200 (rate*duration).
    # The preflight is one request; measured dispatches stay near the cap.
    assert controller.request_count <= 40, f"dispatched far above cap: {controller.request_count}"


# ---------------------------------------------------------------------------
# E2E-014: Open-loop arrivals independent of completions (FR-016)
# ---------------------------------------------------------------------------


def test_e2e_014_arrivals_independent_of_completions(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Arrival t_start follows the Poisson schedule despite bimodal latency.

    FR-016: with bimodal completion latency (half ~5 ms, half ~500 ms), the
    correlation between a request's ``t_start`` and the *previous* request's
    ``e2e`` is < 0.2, proving arrivals are scheduled independently of
    completions (open-loop, not closed-loop pacing).
    """
    base_url, controller = fake_sut

    # Bimodal latency: even-indexed requests fast (~5 ms), odd-indexed slow (~200 ms).
    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        slow = index % 2 == 1
        per_delta = 100.0 if slow else 2.5
        return Behavior(deltas=[Delta("x", sleep_ms=per_delta) for _ in range(2)], usage=Usage(completion_tokens=2))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides=_open_overrides(request_rates=[60], duration="1s", max_outstanding=500),
    )

    out_dir = tmp_path / "r14"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    # Open-loop: arrivals are scheduled on the Poisson clock, not gated by the
    # (bimodal, partly ~500 ms) completion latency, so the arrival count tracks
    # rate*duration. A closed-loop single VU would complete far fewer (it blocks
    # on each slow completion before issuing the next request).
    assert all(r["mode"] == "open" for r in records), {r["mode"] for r in records}
    assert len(records) >= 40, f"arrivals gated by completions (closed-loop-like): {len(records)}"

    ordered = sorted(records, key=lambda r: r["t_start"])
    # Pair each request's t_start with the previous request's measured e2e.
    pairs = [(cur["t_start"], prev["e2e"]) for prev, cur in itertools.pairwise(ordered) if prev.get("e2e") is not None]
    assert len(pairs) >= 20, f"too few pairs to correlate: {len(pairs)}"
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    corr = statistics.correlation(xs, ys) if len(set(ys)) > 1 else 0.0
    assert abs(corr) < 0.2, f"arrivals coupled to completions: corr={corr}"


# ---------------------------------------------------------------------------
# E2E-015: Open-loop request_rate 0 rejected (FR-016)
# ---------------------------------------------------------------------------


def test_e2e_015_request_rate_zero_rejected(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A request_rate of 0 aborts with a non-zero exit and no run data.

    FR-016: exit non-zero; stderr contains ``request_rate must be > 0``; no run
    data is written.
    """
    base_url, _controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "mode": "open",
            "request_rates": [0],
            "burstiness": 1.0,
            "duration": "0.6s",
            "warmup": "0s",
            "cooldown": "0s",
            "max_outstanding": 500,
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r15"
    result = _invoke(config, out_dir)

    assert result.exit_code != 0, _stderr(result)
    assert "request_rate must be > 0" in _stderr(result)
    assert not out_dir.exists() or not any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# E2E-016: max_outstanding hit emits warning + counter (FR-017)
# ---------------------------------------------------------------------------


def test_e2e_016_max_outstanding_events_counter(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Hitting the guard increments a summary counter and prints the count.

    FR-017: with the E2E-013 stalling setup, ``rates[0].max_outstanding_events``
    is > 0 and the terminal prints ``max_outstanding reached N times`` where N
    matches the recorded counter.
    """
    base_url, controller = fake_sut

    # Pre-flight (index 0) must succeed; every measured response stalls 5 s so the
    # guard engages and increments its counter.
    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 0:
            return Behavior(deltas=[Delta("x", sleep_ms=1.0)], usage=Usage(completion_tokens=1))
        return Behavior(force_timeout_ms=1_200.0, deltas=[Delta("x")])

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "mode": "open",
            "request_rates": [100],
            "burstiness": 1.0,
            "max_outstanding": 10,
            "duration": "0.6s",
            "warmup": "0s",
            "cooldown": "0s",
            "timeout": "1s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r16"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    summary = _read_summary(out_dir)
    rate0 = _rates(summary)[0]
    events = rate0["max_outstanding_events"]
    assert events > 0, f"guard never recorded: {rate0}"
    assert f"max_outstanding reached {events} times" in _stderr(result), _stderr(result)


# ---------------------------------------------------------------------------
# E2E-017: Open-loop seeded arrivals reproducible (FR-016,033)
# ---------------------------------------------------------------------------


def test_e2e_017_seeded_arrivals_reproducible(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Two seeded open-loop runs share identical inter-arrival gap sequences.

    FR-016/FR-033: with ``seed:42`` and identical config, the ordered sequence
    of inter-arrival gaps (rounded to 1 ms) in ``a/raw.jsonl`` equals that of
    ``b/raw.jsonl``.
    """
    base_url, controller = fake_sut
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))
    )

    def _gaps_ms(out_dir: Path) -> list[int]:
        records = _read_jsonl(out_dir / "raw.jsonl")
        starts = sorted(r["t_start"] for r in records)
        return [round((b - a) * 1000.0) for a, b in itertools.pairwise(starts)]

    seqs: list[list[int]] = []
    for name in ("a", "b"):
        config = cfg_base(
            _port_of(base_url),
            run_overrides=_open_overrides(request_rates=[100], duration="0.6s", seed=42, max_outstanding=500),
        )
        out_dir = tmp_path / name
        result = _invoke(config, out_dir)
        assert result.exit_code == 0, _stderr(result)
        seqs.append(_gaps_ms(out_dir))

    common = min(len(seqs[0]), len(seqs[1]))
    assert common > 1, "need several arrivals to show a non-trivial gap sequence"
    assert seqs[0][:common] == seqs[1][:common], (
        f"seeded inter-arrival gaps not reproducible: {seqs[0][:common]} != {seqs[1][:common]}"
    )


# ---------------------------------------------------------------------------
# E2E-081: Goodput counts only SLO-meeting requests (FR-029,030)
# ---------------------------------------------------------------------------


def test_e2e_081_goodput_counts_only_slo_meeting(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Goodput counts only requests passing every interactive SLO threshold.

    FR-029: interactive profile ``{ttft<500, tpot<50, e2e<5000}`` (ms); three
    requests A (pass), B (fail ttft), C (fail tpot); ``goodput_count==1``,
    ``goodput_attainment ~ 1/3``, and ``goodput_rps == 1/steady_window``.

    The three outcomes are shaped via per-request FakeSUT timing:
    * A: tiny ttft + tiny per-token gaps -> ttft~100ms, tpot~20ms, e2e~300ms.
    * B: a large pre-first-chunk stall -> ttft~600ms (fails ttft<500).
    * C: tiny ttft but large per-token gaps -> tpot~80ms (fails tpot<50).
    """
    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        measured = index - 1  # index 0 is the pre-flight
        if measured == 0:  # A: passes all thresholds
            return Behavior(deltas=[Delta("x", sleep_ms=20.0) for _ in range(5)], usage=Usage(completion_tokens=5))
        if measured == 1:  # B: fails ttft (~600ms before first content chunk)
            return Behavior(
                deltas=[Delta("x", sleep_ms=600.0), *(Delta("x", sleep_ms=5.0) for _ in range(4))],
                usage=Usage(completion_tokens=5),
            )
        # C (measured >= 2): fails tpot (~80ms per output token).
        return Behavior(deltas=[Delta("x", sleep_ms=80.0) for _ in range(5)], usage=Usage(completion_tokens=5))

    controller.set_function(_fn)
    # Closed-loop, level 1, exactly 3 steady requests so A/B/C map 1:1.
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "mode": "closed",
            "concurrency_levels": [1],
            "duration": "1s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
            "slo_profile": "interactive",
        },
    )

    out_dir = tmp_path / "r81"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    summary = _read_summary(out_dir)
    # Goodput is reported per level/rate; closed-loop uses ``levels``.
    levels = summary.get("levels")
    assert isinstance(levels, list) and levels, summary
    level0 = levels[0]
    assert level0["goodput_count"] == 1, level0
    assert abs(level0["goodput_attainment"] - (1.0 / 3.0)) <= 0.05, level0["goodput_attainment"]
    # goodput_rps == goodput_count / steady_window; steady_window is positive.
    steady_window = level0.get("steady_window")
    assert steady_window and steady_window > 0, level0
    assert abs(level0["goodput_rps"] - (1.0 / steady_window)) <= 1e-6, level0


# ---------------------------------------------------------------------------
# E2E-082: SLO profiles selectable + overridable (FR-029,030)
# ---------------------------------------------------------------------------


def _write_slo_config(
    tmp_path: Path,
    port: int,
    monkeypatch: Any,
    *,
    slo_profile: str,
) -> Path:
    """Write a CFG_BASE-style config that also defines ``slo_profiles``.

    The shared ``cfg_base`` factory only renders the ``models``/``run`` blocks,
    so the two named SLO profiles (interactive/relaxed) are written by hand here,
    mirroring the factory's literal-token style for ``${SUT_PORT}``/``$ENV:``.
    """
    monkeypatch.setenv("SUT_API_KEY", "sk-test")
    monkeypatch.setenv("SUT_PORT", str(port))
    lines = [
        "models:",
        "  - name: sut",
        "    base_url: http://127.0.0.1:${SUT_PORT}/v1",
        "    model: fake/model",
        "    api_key: $ENV:SUT_API_KEY",
        "    supports_vision: false",
        "    supports_tools: false",
        "slo_profiles:",
        "  interactive:",
        "    ttft_ms: 500",
        "    tpot_ms: 50",
        "    e2e_ms: 5000",
        "  relaxed:",
        "    ttft_ms: 2000",
        "    tpot_ms: 200",
        "    e2e_ms: 20000",
        "run:",
        "  mode: closed",
        "  duration: 0.45s",
        "  warmup: 0s",
        "  cooldown: 0s",
        "  min_samples: 1",
        "  concurrency_levels: [1]",
        "  max_tokens: 8",
        "  ignore_eos: true",
        "  temperature: 0.0",
        "  cache_busting: true",
        "  retries: 0",
        "  timeout: 5s",
        "  seed: 42",
        f"  slo_profile: {slo_profile}",
    ]
    path = tmp_path / f"config_{slo_profile}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_e2e_082_slo_profiles_selectable_and_overridable(
    fake_sut: tuple[str, SUTController],
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """interactive/relaxed profiles and a ``--slo`` CLI override drive thresholds.

    FR-030: ``resolved_config.json`` records the active thresholds: interactive
    ``ttft_ms:500``, relaxed ``ttft_ms:2000``, override ``ttft_ms:300``; goodput
    is computed against the active thresholds (a request with ttft ~400 passes
    under interactive but fails under the 300 override).
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)

    def _resolved_slo(out_dir: Path) -> dict[str, Any]:
        path = out_dir / "resolved_config.json"
        assert path.exists(), f"missing resolved_config.json: {path}"
        resolved = json.loads(path.read_text(encoding="utf-8"))
        slo = resolved.get("slo")
        assert isinstance(slo, dict), f"resolved_config has no active slo block: {resolved!r}"
        return slo

    # Single request with ttft ~400 ms (one ~400 ms first-chunk stall): it passes
    # goodput under interactive (ttft<500) but fails under the 300 override.
    controller.set_default(
        Behavior(
            deltas=[Delta("x", sleep_ms=400.0), *(Delta("x", sleep_ms=2.0) for _ in range(3))],
            usage=Usage(completion_tokens=4),
        )
    )

    # interactive -> ttft_ms 500 active.
    cfg_i = _write_slo_config(tmp_path, port, monkeypatch, slo_profile="interactive")
    out_i = tmp_path / "r82i"
    res_i = _invoke(cfg_i, out_i)
    assert res_i.exit_code == 0, _stderr(res_i)
    assert _resolved_slo(out_i)["ttft_ms"] == 500

    # relaxed -> ttft_ms 2000 active.
    cfg_r = _write_slo_config(tmp_path, port, monkeypatch, slo_profile="relaxed")
    out_r = tmp_path / "r82r"
    res_r = _invoke(cfg_r, out_r)
    assert res_r.exit_code == 0, _stderr(res_r)
    assert _resolved_slo(out_r)["ttft_ms"] == 2000

    # CLI override --slo ttft_ms=300 -> ttft_ms 300 active, and goodput reflects it.
    cfg_o = _write_slo_config(tmp_path, port, monkeypatch, slo_profile="interactive")
    out_o = tmp_path / "r82o"
    res_o = _invoke(cfg_o, out_o, "--slo", "ttft_ms=300")
    assert res_o.exit_code == 0, _stderr(res_o)
    assert _resolved_slo(out_o)["ttft_ms"] == 300

    # Under interactive (500) the ttft~400 request passes goodput; under the 300
    # override it fails. Goodput is computed against the active thresholds.
    summ_i = _read_summary(out_i)["levels"][0]
    summ_o = _read_summary(out_o)["levels"][0]
    assert summ_i["goodput_count"] >= 1, summ_i
    assert summ_o["goodput_count"] == 0, summ_o


# ---------------------------------------------------------------------------
# E2E-083: Goodput zero when none meet SLO (FR-029,030)
# ---------------------------------------------------------------------------


def test_e2e_083_goodput_zero_when_none_meet_slo(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """When every request fails ttft, goodput is zero with no division error.

    FR-029: interactive profile but every request has ttft ~600 ms (> 500);
    ``goodput_count==0``, ``goodput_attainment==0.0``, ``goodput_rps==0.0`` and
    the run completes without a division-by-zero error.
    """
    base_url, controller = fake_sut

    # Pre-flight (index 0) returns instantly; every measured response stalls ~600 ms
    # before the first content chunk (fails ttft<500).
    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 0:
            return Behavior(deltas=[Delta("x", sleep_ms=1.0)], usage=Usage(completion_tokens=1))
        return Behavior(
            deltas=[Delta("x", sleep_ms=600.0), *(Delta("x", sleep_ms=2.0) for _ in range(3))],
            usage=Usage(completion_tokens=4),
        )

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "mode": "closed",
            "concurrency_levels": [1],
            "duration": "0.7s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
            "slo_profile": "interactive",
        },
    )

    out_dir = tmp_path / "r83"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    level0 = _read_summary(out_dir)["levels"][0]
    assert level0["goodput_count"] == 0, level0
    assert level0["goodput_attainment"] == 0.0, level0
    assert level0["goodput_rps"] == 0.0, level0
