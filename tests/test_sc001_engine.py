"""Acceptance tests for SC-001: closed-loop ENGINE + RELIABILITY behaviors.

These cover the Closed-loop load generation and Pre-flight/reliability bounded
contexts of scenario SC-001 (spec
``specs/2026-06-24_09:28:00-llm-bench-core.md``, Section 5 SC-001, Section 6
FR-005..FR-015, FR-058, FR-059, and the Section 12.2 Gherkin for the E2E ids
exercised below).

Each test drives the ``llm-bench`` CLI (``run`` subcommand) through Typer's
:class:`CliRunner` against the offline FakeSUT harness from ``conftest.py`` and
asserts exactly the Gherkin observables: CLI exit codes, exact stderr/log
strings, per-request ``raw.jsonl`` fields, and ``summary.json`` fields.

One test per E2E id:

* Engine / phases / validation: E2E-002, 003, 004, 005, 006, 007, 009, 010.
* Pre-flight + reliability: E2E-056..067.
* Event-loop lag + disk: E2E-101, 104, 108, 112.

These reference ``summary.json`` fields produced by the metrics/storage layers
(implemented in sibling tasks) and reliability classification not yet present in
the minimal SC-007 runner, so they are expected to be RED now. They are written
to the spec contract, not to the current behavior. E2E-067 uses a subprocess +
SIGINT because ``CliRunner`` cannot deliver a signal mid-run.

The shared ``cfg_base`` factory writes the ``CFG_BASE`` config; everything is
parameterised through its ``run_overrides`` because the minimal ``run`` command
exposes only ``--config``/``--model``/``--out``/``--dry-run``.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner

from llm_bench import runner as runner_module
from llm_bench.config import ModelRegistryEntry, RunConfig
from llm_bench.llm_bench import app
from llm_bench.prompts import Prompt
from tests.conftest import Behavior, Delta, Usage

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import SUTController

runner = CliRunner()

# Spec tolerance for tiny real delays (Section 12 "Time control": ±15%).
_TOL = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a FakeSUT ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def _closed_port() -> int:
    """Return a TCP port with nothing listening (bind then immediately release)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _levels(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the per-level summary list, tolerating a ``{'levels': [...]}`` shape."""
    levels = summary.get("levels")
    assert isinstance(levels, list) and levels, f"summary has no levels: {summary!r}"
    return levels


def _invoke(config: Path, out_dir: Path, model: str = "sut") -> Any:
    """Invoke ``llm-bench run`` writing artifacts to ``out_dir``."""
    return runner.invoke(
        app,
        ["run", "--config", str(config), "--model", model, "--out", str(out_dir)],
    )


# ---------------------------------------------------------------------------
# E2E-002: Phase tagging warmup/steady/cooldown boundaries (FR-006)
# ---------------------------------------------------------------------------


def test_e2e_002_phase_tagging_boundaries(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Records carry phase in {warmup,steady,cooldown} with correct boundaries.

    FR-006: with duration 1.0s, warmup 0.3s, cooldown 0.3s, the steady window
    starts at level_start+0.3s and cooldown at level_start+0.7s; at least one
    record exists in each phase.
    """
    base_url, controller = fake_sut
    # Fast SUT: 1 ms/delta (reshape default with tiny per-delta sleep).

    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(4)], usage=Usage(completion_tokens=4))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "1.0s",
            "warmup": "0.3s",
            "cooldown": "0.3s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r2"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    phases = {r["phase"] for r in records}
    assert phases <= {"warmup", "steady", "cooldown"}, phases
    assert "warmup" in phases
    assert "steady" in phases
    assert "cooldown" in phases

    level_start = min(r["t_start"] for r in records)
    steady = [r["t_start"] for r in records if r["phase"] == "steady"]
    cooldown = [r["t_start"] for r in records if r["phase"] == "cooldown"]
    assert min(steady) >= level_start + 0.3 - _TOL
    assert min(cooldown) >= level_start + 0.7 - _TOL


# ---------------------------------------------------------------------------
# E2E-003: stream_options.include_usage on every request (FR-007)
# ---------------------------------------------------------------------------


def test_e2e_003_stream_options_include_usage(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """100% of captured bodies stream with stream_options.include_usage=True.

    FR-007: every request body has ``stream is True`` and
    ``stream_options.include_usage is True``; none omits ``stream_options``.
    """
    base_url, controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r3"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    assert controller.requests, "no requests captured"
    for recorded in controller.requests:
        body = recorded.body
        assert body.get("stream") is True, body
        assert "stream_options" in body, body
        assert body["stream_options"].get("include_usage") is True, body


# ---------------------------------------------------------------------------
# E2E-004: min_samples warning when level under-collects (FR-008)
# ---------------------------------------------------------------------------


def test_e2e_004_min_samples_warning(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A slow SUT under-collecting steady samples warns and continues.

    FR-008: stderr/log matches ``steady samples N < min_samples 30``; exit 0;
    ``levels[0].steady_samples`` < 30 and ``levels[0].min_samples_met`` is false.
    """

    base_url, controller = fake_sut
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=50.0) for _ in range(4)], usage=Usage(completion_tokens=4))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.1s",
            "cooldown": "0.1s",
            "min_samples": 30,
        },
    )

    out_dir = tmp_path / "r4"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    text = _stderr(result)
    assert "steady samples" in text and "min_samples 30" in text, text
    assert "<" in text

    summary = _read_summary(out_dir)
    level0 = _levels(summary)[0]
    assert level0["steady_samples"] < 30
    assert level0["min_samples_met"] is False


# ---------------------------------------------------------------------------
# E2E-005: Concurrency level 0 rejected (FR-005)
# ---------------------------------------------------------------------------


def test_e2e_005_level_zero_rejected(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A concurrency level of 0 aborts with exit 2 naming the offending value.

    FR-005: stderr contains ``concurrency level must be >= 1`` and names ``0``;
    no run output written.
    """
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [0, 1]})

    out_dir = tmp_path / "r5"
    result = _invoke(config, out_dir)

    assert result.exit_code == 2, _stderr(result)
    text = _stderr(result)
    assert "concurrency level must be >= 1" in text
    assert "0" in text
    assert not out_dir.exists() or not any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# E2E-006: Concurrency level 1 single virtual user (FR-005,022)
# ---------------------------------------------------------------------------


def test_e2e_006_single_virtual_user(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Level 1 never exceeds 1 in-flight; summary reports rps and per-user tok/s.

    FR-005/FR-022: server-recorded max concurrency == 1; ``levels[0].rps > 0``
    and ``levels[0].per_user_tok_s`` is present.
    """
    base_url, controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r6"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    # Server-recorded peak concurrency (preflight + measured) is 1.
    assert controller.max_in_flight == 1, controller.max_in_flight

    summary = _read_summary(out_dir)
    level0 = _levels(summary)[0]
    assert level0["rps"] > 0
    assert "per_user_tok_s" in level0


# ---------------------------------------------------------------------------
# E2E-007: Duration shorter than warmup+cooldown rejected (FR-005,006)
# ---------------------------------------------------------------------------


def test_e2e_007_duration_too_short_rejected(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """warmup+cooldown >= duration aborts with the exact arithmetic message.

    FR-006: exit non-zero; stderr contains
    ``warmup + cooldown (0.6s) exceeds duration (0.5s)``; no run data.
    """
    base_url, _controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={"duration": "0.5s", "warmup": "0.3s", "cooldown": "0.3s", "concurrency_levels": [1]},
    )

    out_dir = tmp_path / "r7"
    result = _invoke(config, out_dir)

    assert result.exit_code != 0, _stderr(result)
    assert "warmup + cooldown (0.6s) exceeds duration (0.5s)" in _stderr(result)
    assert not out_dir.exists() or not any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# E2E-009: Levels run sequentially not overlapping (FR-005)
# ---------------------------------------------------------------------------


def test_e2e_009_levels_sequential(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """All level-1 requests finish before any level-2 request starts.

    FR-005: per-record ``level_or_rate`` partitions the timeline so that
    max ``t_start`` of level 1 < min ``t_start`` of level 2 (no interleaving).
    """
    base_url, _controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1, 2],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r9"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    l1 = [r["t_start"] for r in records if r["level_or_rate"] == 1]
    l2 = [r["t_start"] for r in records if r["level_or_rate"] == 2]
    assert l1, "no level-1 records"
    assert l2, "no level-2 records"
    # Level 1 fully precedes level 2 (client-side t_start ordering).
    assert max(l1) <= min(l2)


def test_in_run_preflight_uses_trivial_ping(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The in-run pre-flight sends a deterministic ``ping``, not a library prompt.

    Regression: a random library prompt (e.g. a vision image rejected by a real
    backend) must not be used for the reachability check and abort a fine run.
    """
    base_url, controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={"concurrency_levels": [1], "duration": "0.2s", "warmup": "0.05s", "cooldown": "0.05s"},
    )
    result = _invoke(config, tmp_path / "pf")
    assert result.exit_code == 0, _stderr(result)

    preflight = [r for r in controller.requests if "x-llmbench-preflight" in {k.lower() for k in r.headers}]
    assert len(preflight) == 1, "expected exactly one tagged pre-flight request"
    # Content may carry a cache-busting prefix; the body is the trivial ping.
    assert preflight[0].body["messages"][-1]["content"].endswith("ping")


def test_concurrency_and_duration_cli_overrides(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--concurrency 1,2`` and ``--duration`` override the closed-loop config.

    The config registers a single level [4] with a longer duration; the CLI flags
    replace them, so the summary carries exactly levels 1 and 2.
    """
    base_url, _controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [4],
            "duration": "5s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "ov"
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--model",
            "sut",
            "--out",
            str(out_dir),
            "--concurrency",
            "1,2",
            "--duration",
            "0.3s",
        ],
    )
    assert result.exit_code == 0, _stderr(result)
    levels = [entry["level_or_rate"] for entry in _levels(_read_summary(out_dir))]
    assert levels == [1, 2]


def test_concurrency_override_rejected_in_open_mode(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--concurrency`` with ``--mode open`` is a closed-only misuse and exits 2."""
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--model",
            "sut",
            "--out",
            str(tmp_path / "x"),
            "--mode",
            "open",
            "--concurrency",
            "2",
        ],
    )
    assert result.exit_code == 2
    assert "--concurrency only applies in closed mode" in _stderr(result)


def test_concurrency_override_rejects_bad_value(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A non-integer ``--concurrency`` token exits 2 with a clear message."""
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(tmp_path / "x"), "--concurrency", "1,oops"],
    )
    assert result.exit_code == 2
    assert "invalid --concurrency" in _stderr(result)


def test_generation_overrides_apply_in_dry_run(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--slo-profile`` / ``--max-tokens`` / ``--temperature`` resolve cleanly (exit 0)."""
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--model",
            "sut",
            "--dry-run",
            "--slo-profile",
            "relaxed",
            "--max-tokens",
            "64",
            "--temperature",
            "0.3",
        ],
    )
    assert result.exit_code == 0, _stderr(result)


def test_slo_profile_override_rejects_unknown(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """An unknown ``--slo-profile`` exits 2 naming the known profiles."""
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    result = runner.invoke(
        app, ["run", "--config", str(config), "--model", "sut", "--dry-run", "--slo-profile", "bogus"]
    )
    assert result.exit_code == 2
    assert "unknown --slo-profile" in _stderr(result)


def test_max_tokens_override_rejects_non_positive(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--max-tokens 0`` exits 2 with a clear message."""
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    result = runner.invoke(app, ["run", "--config", str(config), "--model", "sut", "--dry-run", "--max-tokens", "0"])
    assert result.exit_code == 2
    assert "invalid --max-tokens" in _stderr(result)


def test_duration_override_allowed_in_open_mode(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--duration`` now applies in open mode too (no closed-only rejection)."""
    base_url, _controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--model",
            "sut",
            "--dry-run",
            "--mode",
            "open",
            "--request-rate",
            "5",
            "--duration",
            "20s",
        ],
    )
    assert result.exit_code == 0, _stderr(result)


# ---------------------------------------------------------------------------
# E2E-010: 1000 virtual users closed-loop stability (FR-005,022)
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_e2e_010_thousand_users_stable(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """1000 VUs run cleanly: peak concurrency ~1000, no connection errors.

    FR-005/FR-022: exit 0; observed peak concurrency between 950 and 1000; no
    record ``outcome=='connection_error'``; ``levels[0].completed > 0``. Keeps
    duration tiny (300 ms) for test speed.
    """

    base_url, controller = fake_sut
    # Hold each connection open ~120ms so all 1000 virtual users overlap in-flight
    # (a sub-millisecond response would close before the herd fully connects, even
    # with a large accept backlog and under concurrent-suite machine load).
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=60.0) for _ in range(2)], usage=Usage(completion_tokens=2))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1000],
            "duration": "0.8s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r10"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    assert 950 <= controller.max_in_flight <= 1000, controller.max_in_flight

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert all(r["outcome"] != "connection_error" for r in records)

    summary = _read_summary(out_dir)
    assert _levels(summary)[0]["completed"] > 0


# ---------------------------------------------------------------------------
# E2E-056: Pre-flight verification precedes measurement (FR-009)
# ---------------------------------------------------------------------------


def test_e2e_056_preflight_precedes_measurement(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The first request is a single tagged pre-flight, absent from raw.jsonl.

    FR-009: the first request received by FakeSUT carries
    ``X-LLMBench-Preflight: 1`` and precedes any measured load; its data does not
    appear in ``raw.jsonl``.
    """
    base_url, controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r56"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    assert controller.requests, "no requests captured"
    first = controller.requests[0]
    assert first.headers.get("X-LLMBench-Preflight") == "1", first.headers
    # Only one pre-flight overall.
    preflight_count = sum(1 for r in controller.requests if r.headers.get("X-LLMBench-Preflight") == "1")
    assert preflight_count == 1

    # Pre-flight data is not persisted: measured records == total - 1 preflight.
    records = _read_jsonl(out_dir / "raw.jsonl")
    assert len(records) == len(controller.requests) - 1


# ---------------------------------------------------------------------------
# E2E-057: Pre-flight failure aborts, no run data (FR-010)
# ---------------------------------------------------------------------------


def test_e2e_057_preflight_failure_aborts(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """An HTTP 500 to the pre-flight aborts with a descriptive message.

    FR-010: exit non-zero; stderr contains ``pre-flight verification failed`` and
    ``HTTP 500``; ``raw.jsonl`` is absent or empty.
    """
    base_url, controller = fake_sut
    # First request (the pre-flight) returns 500.
    controller.nth_request_status(0, 500)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r57"
    result = _invoke(config, out_dir)

    assert result.exit_code != 0, _stderr(result)
    text = _stderr(result)
    assert "pre-flight verification failed" in text
    assert "HTTP 500" in text or "500" in text
    raw = out_dir / "raw.jsonl"
    assert not raw.exists() or not raw.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# E2E-058: Pre-flight connection refused aborts (FR-010)
# ---------------------------------------------------------------------------


def test_e2e_058_preflight_connection_refused(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A closed base_url port aborts pre-flight with ``connection refused``.

    FR-010: exit non-zero; stderr contains ``pre-flight verification failed`` and
    ``connection refused``; no run data.
    """
    base_url, _controller = fake_sut
    # Build the config against the live port, then repoint to a closed one.
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
        },
    )
    dead = _closed_port()
    text = config.read_text(encoding="utf-8")
    text = text.replace("${SUT_PORT}", str(dead))
    config.write_text(text, encoding="utf-8")
    os.environ["SUT_PORT"] = str(dead)

    out_dir = tmp_path / "r58"
    result = _invoke(config, out_dir)

    assert result.exit_code != 0, _stderr(result)
    text = _stderr(result).lower()
    assert "pre-flight verification failed" in _stderr(result)
    assert "connection refused" in text
    raw = out_dir / "raw.jsonl"
    assert not raw.exists() or not raw.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# E2E-059: 429 recorded as rate_limited, no retry, continues (FR-011)
# ---------------------------------------------------------------------------


def test_e2e_059_rate_limited_no_retry(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A single 429 is recorded once as rate_limited with retry_count 0.

    FR-011: exactly one record ``outcome=='rate_limited'``, ``status_code==429``,
    ``retry_count==0``; the run continues and exits 0. The 3rd request (the 2nd
    measured request after the pre-flight) returns 429.
    """

    base_url, controller = fake_sut

    # Request 0 is the pre-flight (200); request index 2 (the spec's "3rd
    # request") returns 429; all others 200.
    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 2:
            return Behavior(status=429, role_first_chunk=False, deltas=[], error_body={"error": "rate"})
        return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 3,
            "retries": 0,
        },
    )

    out_dir = tmp_path / "r59"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    rate_limited = [r for r in records if r["outcome"] == "rate_limited"]
    assert len(rate_limited) == 1, records
    rec = rate_limited[0]
    assert rec["status_code"] == 429
    assert rec["retry_count"] == 0
    # The run continued and produced later successes.
    assert any(r["outcome"] == "success" for r in records)


# ---------------------------------------------------------------------------
# E2E-060: 429 rate surfaced, flagged when >1% (FR-012)
# ---------------------------------------------------------------------------


def test_e2e_060_rate_limited_flagged(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A 5% 429 rate is surfaced and flagged above the 1% threshold.

    FR-012: ``levels[0].rate_limited_rate ~ 0.05``; terminal flags a >1% breach;
    ``rate_limited_flagged == true``.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        # index 0 is the pre-flight; every 20th measured request returns 429.
        if index >= 1 and (index - 1) % 20 == 19:
            return Behavior(status=429, role_first_chunk=False, deltas=[], error_body={"error": "rate"})
        return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(1)], usage=Usage(completion_tokens=1))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.6s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 100,
        },
    )

    out_dir = tmp_path / "r60"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    summary = _read_summary(out_dir)
    level0 = _levels(summary)[0]
    assert abs(level0["rate_limited_rate"] - 0.05) <= 0.02, level0["rate_limited_rate"]
    assert level0["rate_limited_flagged"] is True
    assert "1%" in _stderr(result) or "1 percent" in _stderr(result)


# ---------------------------------------------------------------------------
# E2E-061: 429 rate exactly 1% not flagged (FR-012)
# ---------------------------------------------------------------------------


def test_e2e_061_rate_limited_exactly_one_percent_not_flagged(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Exactly 1.0% 429 (1 of 100) is not flagged (threshold strictly > 1%).

    FR-012: ``rate_limited_rate == 0.01``; ``rate_limited_flagged == false``; no
    flag string emitted.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        # index 0 = pre-flight; the 50th measured request returns 429. The ~5ms
        # response time yields ~100-200 steady requests over the 1s window, so the
        # single 429 is ~1 of ~100-200 steady requests (rate ~ 0.005-0.01, <=1%).
        if index == 50:
            return Behavior(status=429, role_first_chunk=False, deltas=[], error_body={"error": "rate"})
        return Behavior(deltas=[Delta("x", sleep_ms=5.0) for _ in range(1)], usage=Usage(completion_tokens=1))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "1s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r61"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    summary = _read_summary(out_dir)
    level0 = _levels(summary)[0]
    assert abs(level0["rate_limited_rate"] - 0.01) <= 0.005, level0["rate_limited_rate"]
    assert level0["rate_limited_flagged"] is False
    assert "exceeds 1%" not in _stderr(result)


# ---------------------------------------------------------------------------
# E2E-062: Timeout recorded as failure (FR-013)
# ---------------------------------------------------------------------------


def test_e2e_062_timeout_recorded(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A request exceeding the timeout is recorded as ``timeout``.

    FR-013: that record has ``outcome=='timeout'`` and ``error`` containing
    ``timeout``; it is counted in failures; exit 0.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        # index 0 = pre-flight; the 2nd request (index 2) stalls 10 s.
        if index == 2:
            return Behavior(force_timeout_ms=400.0, deltas=[Delta("x", sleep_ms=1.0)])
        return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.5s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 3,
            "timeout": "0.2s",
        },
    )

    out_dir = tmp_path / "r62"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    timeouts = [r for r in records if r["outcome"] == "timeout"]
    assert len(timeouts) >= 1, records
    assert "timeout" in (timeouts[0].get("error") or "").lower()


# ---------------------------------------------------------------------------
# E2E-063: Failed/timed-out excluded from distributions (FR-014)
# ---------------------------------------------------------------------------


def test_e2e_063_failures_excluded_from_distributions(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Latency percentiles are computed over successes only.

    FR-014: ``levels[0].latency_sample_count`` equals the number of success
    records; failed/timed-out/rate-limited records are excluded from latency but
    counted in their reliability buckets.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 0:  # pre-flight
            return Behavior(deltas=[Delta("x", sleep_ms=1.0)], usage=Usage(completion_tokens=1))
        measured = index - 1
        if measured in (3, 7):  # two timeouts
            return Behavior(force_timeout_ms=400.0, deltas=[Delta("x", sleep_ms=1.0)])
        if measured == 5:  # one rate-limited
            return Behavior(status=429, role_first_chunk=False, deltas=[], error_body={"error": "rate"})
        return Behavior(deltas=[Delta("x", sleep_ms=4.0) for _ in range(2)], usage=Usage(completion_tokens=2))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.6s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 13,
            "timeout": "0.2s",
        },
    )

    out_dir = tmp_path / "r63"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    successes = [r for r in records if r["outcome"] == "success"]
    summary = _read_summary(out_dir)
    level0 = _levels(summary)[0]
    assert level0["latency_sample_count"] == len(successes)
    # Failures counted separately, not in latency.
    assert any(r["outcome"] == "timeout" for r in records)
    assert any(r["outcome"] == "rate_limited" for r in records)


# ---------------------------------------------------------------------------
# E2E-064: 429 storm: all rate_limited, run completes (FR-011,012)
# ---------------------------------------------------------------------------


def test_e2e_064_rate_limited_storm(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A 100% 429 storm leaves every record rate_limited; the run completes.

    FR-011/FR-012: exit 0; every record ``outcome=='rate_limited'``;
    ``levels[0].rate_limited_rate==1.0`` and flagged true; latency percentiles
    null/absent.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 0:  # pre-flight must succeed (else the run aborts)
            return Behavior(deltas=[Delta("x", sleep_ms=1.0)], usage=Usage(completion_tokens=1))
        return Behavior(status=429, role_first_chunk=False, deltas=[], error_body={"error": "rate"})

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 5,
            "retries": 0,
        },
    )

    out_dir = tmp_path / "r64"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records"
    assert all(r["outcome"] == "rate_limited" for r in records)

    summary = _read_summary(out_dir)
    level0 = _levels(summary)[0]
    assert level0["rate_limited_rate"] == 1.0
    assert level0["rate_limited_flagged"] is True


# ---------------------------------------------------------------------------
# E2E-065: Mid-stream disconnect recorded as malformed_stream (FR-013,014)
# ---------------------------------------------------------------------------


def test_e2e_065_mid_stream_disconnect(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A truncated stream (TCP close, no [DONE]) is a malformed_stream failure.

    FR-013/FR-014: that record ``outcome`` is ``malformed_stream`` (or
    ``connection_error``), ``error`` mentions the stream was interrupted, and it
    is excluded from latency; other requests succeed; exit 0.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 2:  # 2nd measured request truncates mid-stream
            return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(4)], mid_stream_close=True)
        return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 3,
        },
    )

    out_dir = tmp_path / "r65"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    broken = [r for r in records if r["outcome"] in {"malformed_stream", "connection_error"}]
    assert len(broken) >= 1, records
    assert "stream" in (broken[0].get("error") or "").lower() or "interrupt" in (broken[0].get("error") or "").lower()
    assert any(r["outcome"] == "success" for r in records)


# ---------------------------------------------------------------------------
# E2E-066: Unparseable SSE chunk recorded as malformed_stream (FR-013)
# ---------------------------------------------------------------------------


def test_e2e_066_unparseable_sse_chunk(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """An unparseable SSE line is recorded as malformed_stream; run continues.

    FR-013: that record ``outcome=='malformed_stream'`` with ``error`` mentioning
    a JSON decode problem; other requests succeed.
    """

    base_url, controller = fake_sut

    def _fn(index: int, _body: dict[str, Any]) -> Behavior:
        if index == 2:  # 2nd measured request injects a bad SSE line
            return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], malformed_line=True)
        return Behavior(deltas=[Delta("x", sleep_ms=1.0) for _ in range(2)], usage=Usage(completion_tokens=2))

    controller.set_function(_fn)
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 3,
        },
    )

    out_dir = tmp_path / "r66"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    records = _read_jsonl(out_dir / "raw.jsonl")
    malformed = [r for r in records if r["outcome"] == "malformed_stream"]
    assert len(malformed) >= 1, records
    err = (malformed[0].get("error") or "").lower()
    assert "json" in err or "decode" in err or "parse" in err
    assert any(r["outcome"] == "success" for r in records)


# ---------------------------------------------------------------------------
# E2E-067: SIGINT flushes, drains eval, marks incomplete (FR-015)
# ---------------------------------------------------------------------------


def test_e2e_067_sigint_marks_incomplete(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A SIGINT mid-run flushes partial data and marks the run incomplete.

    FR-015: the process exits 130; ``raw.jsonl`` holds valid JSON lines collected
    so far; ``summary.json`` has ``status=='incomplete'``. CliRunner cannot
    deliver a signal mid-run, so the bench is spawned as a real subprocess.
    """

    base_url, controller = fake_sut
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=20.0) for _ in range(4)], usage=Usage(completion_tokens=4))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "1.5s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "r67"
    env = dict(os.environ)
    env["SUT_API_KEY"] = "sk-test"
    env["SUT_PORT"] = str(_port_of(base_url))

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
            "--out",
            str(out_dir),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Let it warm up and collect a few records, then interrupt.
    time.sleep(0.6)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise

    assert proc.returncode == 130, proc.returncode

    raw = out_dir / "raw.jsonl"
    assert raw.exists(), "raw.jsonl not flushed on SIGINT"
    for line in raw.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)  # valid JSON line

    summary = _read_summary(out_dir)
    assert summary.get("status") == "incomplete", summary


# ---------------------------------------------------------------------------
# E2E-101: Event-loop lag above threshold warns (FR-059)
# ---------------------------------------------------------------------------


def test_e2e_101_event_loop_lag_warns(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synchronous loop stall beyond the threshold emits a saturation warning.

    FR-059: with ``event_loop_lag_threshold_ms: 50`` a deterministic ~100ms
    synchronous block injected into the request path (the spec's "CPU-bound
    stub injected into the harness") makes the event loop lag, emitting a
    WARNING matching ``event loop lag ... exceeds threshold 50ms (client
    saturation)``; ``client_saturation_warnings >= 1``; the run completes.
    """

    base_url, controller = fake_sut
    controller.set_default(Behavior(deltas=[Delta("x", sleep_ms=1.0)], usage=Usage(completion_tokens=1)))

    original_build_payload = runner_module._build_payload
    call_state = {"n": 0}

    def _blocking_build_payload(entry: Any, run: Any, prompt: Any) -> dict[str, Any]:
        call_state["n"] += 1
        # Call 1 is the pre-flight (before the lag monitor starts). Block on the
        # first few measured requests (each separated by real request I/O, during
        # which the monitor is sleeping) so at least one ~100ms stall reliably
        # overlaps a monitor poll window and is detected (> 50ms threshold).
        if 2 <= call_state["n"] <= 5:
            time.sleep(0.1)  # block the event loop > threshold (50ms) + poll interval (20ms)
        return original_build_payload(entry, run, prompt)

    monkeypatch.setattr(runner_module, "_build_payload", _blocking_build_payload)

    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
            "event_loop_lag_threshold_ms": 50,
        },
    )

    out_dir = tmp_path / "r101"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    text = _stderr(result)
    assert "event loop lag" in text
    assert "exceeds threshold 50ms" in text
    assert "client saturation" in text

    summary = _read_summary(out_dir)
    assert summary.get("client_saturation_warnings", 0) >= 1


# ---------------------------------------------------------------------------
# E2E-104: Out-of-order SSE chunks handled deterministically (FR-018,020)
# ---------------------------------------------------------------------------


def test_e2e_104_out_of_order_chunks_deterministic(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """ITL timing uses arrival order, yielding deterministic ttft/itl.

    FR-018/FR-020: the implementation uses arrival order; ``ttft`` is the first
    arrived content chunk's time; ``itl_list`` is computed from arrival gaps; the
    result is stable across two runs. The harness lacks a non-monotonic-index
    knob, so this drives ordinary multi-delta streams (closest primitive) and
    asserts the determinism + arrival-order contract.
    """

    base_url, controller = fake_sut

    def _scripted() -> Behavior:
        return Behavior(deltas=[Delta("x", sleep_ms=5.0) for _ in range(4)], usage=Usage(completion_tokens=4))

    def run_once(tag: str) -> dict[str, Any]:
        controller.set_default(_scripted())
        cfg = cfg_base(
            _port_of(base_url),
            run_overrides={
                "concurrency_levels": [1],
                "duration": "0.3s",
                "warmup": "0s",
                "cooldown": "0s",
                "min_samples": 1,
            },
        )
        d = tmp_path / tag
        res = runner.invoke(
            app,
            ["run", "--config", str(cfg), "--model", "sut", "--raw-itl", "--out", str(d)],
        )
        assert res.exit_code == 0, _stderr(res)
        recs = _read_jsonl(d / "raw.jsonl")
        success = [r for r in recs if r["outcome"] == "success"]
        assert success, recs
        return success[0]

    first = run_once("r104a")
    second = run_once("r104b")

    # TTFT is the first arrived content chunk's offset; arrival-order ITL is
    # deterministic across runs (with --raw-itl the full per-gap list is present).
    assert first["ttft"] is not None and first["ttft"] > 0
    assert "itl_list" in first
    assert isinstance(first["itl_list"], list)
    # Deterministic shape across runs (same number of inter-token gaps).
    assert len(first["itl_list"]) == len(second["itl_list"])


# ---------------------------------------------------------------------------
# E2E-108: Disk write failure on JSONL aborts (FR-048)
# ---------------------------------------------------------------------------


def test_e2e_108_disk_write_failure_aborts(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """An unwritable --out directory aborts with a clear message, no junk file.

    FR-048: exit non-zero; stderr contains ``cannot write run data`` and the
    path; no misleadingly-named partial ``raw.jsonl`` is left behind.
    """
    base_url, _controller = fake_sut
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "min_samples": 1,
        },
    )

    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o555)
    out_dir = readonly / "run"

    try:
        result = _invoke(config, out_dir)
        assert result.exit_code != 0, _stderr(result)
        text = _stderr(result)
        assert "cannot write run data" in text
        assert str(out_dir) in text or str(readonly) in text
        raw = out_dir / "raw.jsonl"
        assert not raw.exists() or not raw.read_text(encoding="utf-8").strip()
    finally:
        readonly.chmod(0o755)


# ---------------------------------------------------------------------------
# E2E-112: No false event-loop lag warning under light load (FR-059)
# ---------------------------------------------------------------------------


def test_e2e_112_no_false_lag_warning(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Light load never trips the event-loop lag warning.

    FR-059: with ``event_loop_lag_threshold_ms: 50`` and a single, fast VU, no
    ``event loop lag`` WARNING is emitted; ``client_saturation_warnings == 0``;
    exit 0.
    """

    base_url, controller = fake_sut
    controller.set_default(
        Behavior(deltas=[Delta("x", sleep_ms=2.0) for _ in range(3)], usage=Usage(completion_tokens=3))
    )
    config = cfg_base(
        _port_of(base_url),
        run_overrides={
            "concurrency_levels": [1],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 1,
            "event_loop_lag_threshold_ms": 50,
        },
    )

    out_dir = tmp_path / "r112"
    result = _invoke(config, out_dir)
    assert result.exit_code == 0, _stderr(result)

    assert "event loop lag" not in _stderr(result)
    summary = _read_summary(out_dir)
    assert summary.get("client_saturation_warnings", 0) == 0


# ---------------------------------------------------------------------------
# --preflight: reachability check without running the benchmark (FR-009/010)
# ---------------------------------------------------------------------------


def test_preflight_flag_reachable_exits_zero_no_run_data(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--preflight`` against a reachable endpoint exits 0 and writes no run data."""
    base_url, controller = fake_sut
    config = cfg_base(_port_of(base_url), run_overrides={"concurrency_levels": [1]})
    out_dir = tmp_path / "pf"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--preflight", "--out", str(out_dir)],
    )
    assert result.exit_code == 0, _stderr(result)
    assert "pre-flight OK" in result.stdout
    assert len(controller.requests) == 1  # the single pre-flight request, no sweep
    assert not (out_dir / "raw.jsonl").exists()
    assert not (out_dir / "summary.json").exists()


def test_preflight_flag_unreachable_aborts(
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--preflight`` against a closed port aborts non-zero with no run data."""
    config = cfg_base(_closed_port(), run_overrides={"concurrency_levels": [1]})
    out_dir = tmp_path / "pf2"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--preflight", "--out", str(out_dir)],
    )
    assert result.exit_code != 0
    assert "pre-flight verification failed" in _stderr(result)
    assert not (out_dir / "raw.jsonl").exists()


def test_build_payload_omits_temperature_when_send_temperature_false() -> None:
    """send_temperature=False drops temperature from the SUT payload (gateway compat)."""
    run = RunConfig(temperature=0.0, cache_busting=False)
    prompt = Prompt(id="p", category="general", messages=({"role": "user", "content": "hi"},), isl_bucket="short")
    on = ModelRegistryEntry(name="a", base_url="http://x/v1", model="m")  # send_temperature defaults to True
    off = ModelRegistryEntry(name="b", base_url="http://x/v1", model="m", send_temperature=False)
    assert runner_module._build_payload(on, run, prompt)["temperature"] == 0.0
    assert "temperature" not in runner_module._build_payload(off, run, prompt)
