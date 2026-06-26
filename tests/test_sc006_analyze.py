"""Acceptance tests for SC-006: ad-hoc DuckDB analysis via ``analyze``.

These cover the Reporting/analysis path of scenario SC-006 (Section 5 of the
spec ``specs/2026-06-24_09:28:00-llm-bench-core.md``): the operator runs
``llm-bench analyze <path> --sql "<query>"`` and the system executes the query
through DuckDB directly on the file (table name ``data``) and prints the result
(FR-052, with FR-050 for the Parquet rollup and FR-023 for the recomputed
percentile).

One test per E2E id from Section 12.2: E2E-043, 044, 045, 046, 047, 048. Each
asserts exactly the observables named in the matching Gherkin, driven through
the ``llm-bench`` CLI (``analyze`` subcommand) via Typer's :class:`CliRunner`.

The ``analyze`` command is currently a scaffolding stub that always prints a
"not implemented" line on stderr and exits 1, so every test here fails for the
right reason (wrong exit code / missing query output) until the DuckDB-backed
implementation lands.

Output-format assumption (so the implementer can match it): result-row tests
assert *tolerantly* that the bare numeric value appears in stdout (e.g.
``"50" in out`` and ``"400" in out``), and also accept the friendlier
``key=value`` form (``n=50``). The implementer is free to print a DuckDB table,
a ``key=value`` line, or any rendering that surfaces the values; only the value
substrings are load-bearing. The runs needing real artifacts (E2E-044, 047) are
produced by invoking ``llm-bench run`` against the offline FakeSUT first.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from typer.testing import CliRunner

from llm_bench.llm_bench import app

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact into a list of dicts, asserting it exists first."""
    assert path.exists(), f"expected artifact missing: {path}"
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write ``records`` as a JSONL file (one compact JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in records)
    path.write_text(text, encoding="utf-8")
    return path


def _analyze(data: Path, sql: str) -> Any:
    """Invoke ``llm-bench analyze <data> --sql <sql>`` via the CliRunner."""
    return runner.invoke(app, ["analyze", str(data), "--sql", sql])


def _has_value(out: str, name: str, value: object) -> bool:
    """True if ``out`` surfaces a result column ``name`` with ``value``.

    Tolerant by design: matches the bare value substring (a DuckDB table cell)
    or the friendlier ``name=value`` form so the implementer is free to pick a
    rendering. The bare-value check is anchored on a digit boundary so ``"50"``
    does not spuriously match inside ``"500"``.
    """
    sval = str(value)
    if f"{name}={sval}" in out:
        return True
    return re.search(rf"(?<!\d){re.escape(sval)}(?!\d)", out) is not None


def _short_run(
    cfg_base: Callable[..., Path],
    port: int,
    *,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a CFG_BASE config tuned for a quick single-level closed-loop run."""
    run_overrides: dict[str, Any] = {
        "duration": "0.3s",
        "warmup": "0.05s",
        "cooldown": "0.05s",
        "concurrency_levels": [1],
        "min_samples": 2,
    }
    if overrides:
        run_overrides.update(overrides)
    return cfg_base(port, run_overrides=run_overrides)


def _make_success_record(index: int, *, output_tokens: int, ttft: float = 0.05) -> dict[str, Any]:
    """Build one synthetic ``raw.jsonl`` success record for hand-built fixtures."""
    return {
        "run_id": "r1",
        "model": "fake/model",
        "mode": "closed",
        "level_or_rate": 1,
        "phase": "steady",
        "seed": 42,
        "request_id": f"req-{index:04d}",
        "prompt_id": "p-0",
        "category": "general",
        "isl_bucket": "short",
        "osl_bucket": "short",
        "prompt_tokens": 10,
        "output_tokens": output_tokens,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "usage_incomplete": False,
        "t_start": float(index),
        "ttft": ttft,
        "tt2t": ttft + 0.01,
        "e2e": ttft + 0.2,
        "tpot": 0.02,
        "itl_summary": {"mean": 0.02, "p50": 0.02, "p95": 0.02, "p99": 0.02, "max": 0.02},
        "outcome": "success",
        "status_code": 200,
        "retry_count": 0,
        "error": None,
        "cost_usd": None,
        "sim_score": None,
        "quality_pass": None,
        "judge_verdict": None,
        "judge_reason": None,
        "eval_status": "skipped_no_expected",
    }


# ---------------------------------------------------------------------------
# E2E-043: analyze DuckDB query on JSONL
# ---------------------------------------------------------------------------


def test_e2e_043_analyze_jsonl_count_and_sum(tmp_path: Path) -> None:
    """count + sum over a hand-built raw.jsonl (50 records, tokens summing to 400).

    FR-052: exit 0; stdout surfaces ``n==50`` and ``tok==400`` (DuckDB reads the
    JSONL file directly via the table name ``data``, no ETL step).
    """
    # 50 success records; output_tokens = 8 each => sum 400.
    records = [_make_success_record(i, output_tokens=8) for i in range(50)]
    raw = _write_jsonl(tmp_path / "runs" / "r1" / "raw.jsonl", records)
    assert sum(r["output_tokens"] for r in records) == 400

    result = _analyze(raw, "SELECT count(*) AS n, sum(output_tokens) AS tok FROM data")

    assert result.exit_code == 0, result.stderr or result.output
    out = result.stdout
    assert _has_value(out, "n", 50), f"missing n=50 in: {out!r}"
    assert _has_value(out, "tok", 400), f"missing tok=400 in: {out!r}"


# ---------------------------------------------------------------------------
# E2E-044: analyze on Parquet rollup
# ---------------------------------------------------------------------------


def test_e2e_044_analyze_parquet_rollup_count(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """count(*) over rollup.parquet equals the JSONL record count for the run.

    FR-050/052: produce a real run (so ``rollup.parquet`` exists), then assert
    that ``SELECT count(*) ... FROM data`` over the Parquet file surfaces the
    same record count as ``raw.jsonl``.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "runs" / "r1"
    run_result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )
    assert run_result.exit_code == 0, run_result.stderr

    jsonl_count = len(_read_jsonl(out_dir / "raw.jsonl"))
    assert jsonl_count >= 1, "real run produced no records"

    parquet = out_dir / "rollup.parquet"
    assert parquet.exists(), f"missing rollup.parquet: {parquet}"

    result = _analyze(parquet, "SELECT count(*) AS n FROM data")

    assert result.exit_code == 0, result.stderr or result.output
    assert _has_value(result.stdout, "n", jsonl_count), (
        f"expected n={jsonl_count} (jsonl line count) in: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# E2E-045: analyze invalid SQL returns error
# ---------------------------------------------------------------------------


def test_e2e_045_analyze_invalid_sql_parser_error(tmp_path: Path) -> None:
    """Invalid SQL aborts non-zero with a clean DuckDB parser error.

    FR-052/EXC-006b: exit non-zero; stderr mentions the DuckDB parser error
    (``Parser Error`` / ``syntax error``); not a bare Python traceback.
    """
    records = [_make_success_record(i, output_tokens=8) for i in range(3)]
    raw = _write_jsonl(tmp_path / "runs" / "r1" / "raw.jsonl", records)

    result = _analyze(raw, "SELEKT * FROM data")

    assert result.exit_code != 0, result.output
    err = result.stderr or result.output
    lowered = err.lower()
    assert "parser error" in lowered or "syntax error" in lowered, f"no DuckDB parser error in stderr: {err!r}"
    # A clean, framed error - not a raw Python traceback dump.
    assert "Traceback (most recent call last)" not in err, f"raw traceback leaked: {err!r}"


# ---------------------------------------------------------------------------
# E2E-046: analyze missing data file aborts
# ---------------------------------------------------------------------------


def test_e2e_046_analyze_missing_file_aborts(tmp_path: Path) -> None:
    """A missing data file aborts with a descriptive ``data file not found`` error.

    FR-052/EXC-006a: exit non-zero; stderr contains
    ``data file not found: <path>`` naming the exact path given.
    """
    missing = tmp_path / "runs" / "none" / "raw.jsonl"
    assert not missing.exists()

    result = _analyze(missing, "SELECT 1")

    assert result.exit_code != 0, result.output
    err = result.stderr or result.output
    assert f"data file not found: {missing}" in err, f"missing not-found message for {missing}: {err!r}"


# ---------------------------------------------------------------------------
# E2E-047: analyze recomputes p99 matching summary
# ---------------------------------------------------------------------------


def test_e2e_047_analyze_recomputes_p99_matches_summary(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """quantile_cont(ttft, 0.99) over raw.jsonl matches summary ttft p99 (±2%).

    FR-023/052: produce a real run with enough steady-success records at a single
    level, then assert the analyze-computed p99 matches ``summary.json``'s
    ``levels[0].ttft.p99`` within ±2% (tolerant to a seconds/ms unit difference).
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(
        port,
        run_overrides={
            "duration": "0.3s",
            "warmup": "0s",
            "cooldown": "0s",
            "concurrency_levels": [1],
            "min_samples": 2,
            "max_tokens": 8,
        },
    )

    out_dir = tmp_path / "runs" / "r1"
    run_result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )
    assert run_result.exit_code == 0, run_result.stderr

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    levels = summary["levels"]
    assert isinstance(levels, list) and levels, f"summary.levels missing/empty: {summary!r}"
    ttft_obj = levels[0]["ttft"]
    summary_p99 = float(ttft_obj["p99"] if isinstance(ttft_obj, dict) else ttft_obj)

    result = _analyze(
        out_dir / "raw.jsonl",
        "SELECT quantile_cont(ttft, 0.99) AS p99 FROM data "
        "WHERE phase='steady' AND level_or_rate=1 AND outcome='success'",
    )
    assert result.exit_code == 0, result.stderr or result.output

    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", result.stdout)
    floats = [float(tok) for tok in nums if re.search(r"\d", tok)]
    assert floats, f"no numeric p99 in analyze output: {result.stdout!r}"

    # The raw ttft is in seconds (<1); the summary may be in ms. Accept either by
    # comparing against both the raw value and its x1000 scaling within ±2%.
    def _close(actual: float, expected: float) -> bool:
        tol = max(abs(expected) * 0.02, 1e-9)
        return abs(actual - expected) <= tol

    assert any(_close(val, summary_p99) or _close(val * 1000.0, summary_p99) for val in floats), (
        f"analyze p99 {floats} does not match summary ttft.p99 {summary_p99} within 2%"
    )


# ---------------------------------------------------------------------------
# E2E-048: analyze empty dataset returns zero rows
# ---------------------------------------------------------------------------


def test_e2e_048_analyze_empty_dataset_zero(tmp_path: Path) -> None:
    """count(*) over a 0-record raw.jsonl returns n=0 with no exception.

    FR-052: exit 0; stdout surfaces ``n==0``; the empty input is handled cleanly
    (DuckDB still resolves the ``data`` table schema for an empty file).
    """
    raw = _write_jsonl(tmp_path / "runs" / "zero" / "raw.jsonl", [])
    assert raw.exists() and raw.read_text(encoding="utf-8") == ""

    result = _analyze(raw, "SELECT count(*) AS n FROM data")

    assert result.exit_code == 0, result.stderr or result.output
    assert _has_value(result.stdout, "n", 0), f"missing n=0 in: {result.stdout!r}"
