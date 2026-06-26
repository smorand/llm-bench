"""Acceptance tests for SC-001: prompts, anti-cache, storage, terminal, observability.

These cover the closed-loop core journey of scenario SC-001 (Section 5 of the
spec ``specs/2026-06-24_09:28:00-llm-bench-core.md``) plus its prompt-selection,
cache-busting, request-tagging, storage, rollup, and observability side effects.

One test per E2E id from Section 12.2: E2E-001, 086, 087, 088, 089, 090, 091,
092, 093, 094, 095, 096, 098. Each asserts exactly the observables named in the
matching Gherkin, driven through the ``llm-bench`` CLI (``run`` subcommand) with
its ``--prompts``/``--seed``/``--raw-itl``/``--out`` flags, against the offline
FakeSUT harness from ``conftest.py``.

The relevant FRs are FR-033..037 (prompts and anti-cache), FR-048..053 (storage
+ terminal summary), and FR-056/FR-057 (observability + no leakage), with the
``raw.jsonl`` field set and supporting artifacts (``rollup.parquet``,
``summary.json``, ``traces.jsonl``) from Section 8 (Data Model).

The built-in prompt library does not exist yet, so the default-library tests
(E2E-089) assert via observable behavior (records carry a ``prompt_id`` and a
known category from the packaged set) rather than by importing library internals.
The override/empty cases (E2E-090/091) write a ``--prompts`` YAML to ``tmp_path``
using the FR-036 prompt shape: a list of mappings with ``id``, ``category``,
``messages[]``, optional ``expected_output`` and ``isl_bucket``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from llm_bench.config import DEFAULT_CONFIG_FILE, DEFAULT_PROMPTS_FILE
from llm_bench.llm_bench import app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pytest

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


def _user_content(body: dict[str, Any]) -> str:
    """Return the last user message content from a recorded request body."""
    messages = body.get("messages") or []
    assert messages, f"request body has no messages: {body!r}"
    content = messages[-1].get("content")
    # Multimodal content may be a list of parts; SC-001 prompts are plain text.
    assert isinstance(content, str), f"expected text content, got {content!r}"
    return content


def _short_run(
    cfg_base: Callable[..., Path],
    port: int,
    *,
    overrides: dict[str, Any] | None = None,
    api_key: str = "sk-test",
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
    return cfg_base(port, run_overrides=run_overrides, api_key=api_key)


def _write_prompts(path: Path, prompts: list[dict[str, Any]]) -> Path:
    """Write a ``--prompts`` YAML file using the FR-036 prompt shape.

    Each prompt is a mapping with ``id``, ``category``, ``messages`` (a list of
    ``{role, content}`` mappings), and optional ``expected_output`` /
    ``isl_bucket``. The list is rendered by hand so an empty list serializes to
    the literal ``[]`` required by E2E-091.
    """
    path.write_text(yaml.safe_dump(prompts, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# E2E-001: Closed-loop sweep two levels writes JSONL + terminal summary
# ---------------------------------------------------------------------------


def test_e2e_001_core_journey_jsonl_and_terminal(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Two-level sweep writes raw.jsonl records and a rich per-level table.

    FR-005/006/007/048/053: exit 0; ``raw.jsonl`` has records for each
    concurrency level with the required fields and ``outcome=="success"``; stdout
    shows a per-level rich table with ``p50``/``p99`` headers; every FakeSUT
    request body had ``stream:true`` and ``stream_options.include_usage:true``.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(
        port,
        run_overrides={
            "concurrency_levels": [1, 2],
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 2,
        },
    )

    out_dir = tmp_path / "runs" / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.stderr
    records = _read_jsonl(out_dir / "raw.jsonl")
    levels = {r["level_or_rate"] for r in records}
    assert 1 in levels, f"no record for level 1: {levels}"
    assert 2 in levels, f"no record for level 2: {levels}"

    required = {"run_id", "model", "mode", "level_or_rate", "phase", "ttft", "e2e", "output_tokens", "outcome"}
    for record in records:
        assert required <= record.keys(), f"missing fields: {required - record.keys()}"
        assert record["model"] == "fake/model"
        assert record["mode"] == "closed"
    successes = [r for r in records if r["phase"] == "steady"]
    assert successes, "no steady records"
    assert all(r["outcome"] == "success" for r in successes)

    # Rich terminal table: a row per level plus the percentile headers.
    out = result.stdout
    assert "p50" in out
    assert "p99" in out

    # Every measured request used streaming with usage.
    assert controller.requests, "no requests captured"
    for recorded in controller.requests:
        assert recorded.body.get("stream") is True
        assert recorded.body.get("stream_options", {}).get("include_usage") is True


# ---------------------------------------------------------------------------
# E2E-086: Seeded random prompt selection reproducible
# ---------------------------------------------------------------------------


def test_e2e_086_seeded_prompt_selection_reproducible(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Two identical seeded runs produce the same ordered prompt_id sequence.

    FR-033: with ``seed:42`` and ``cache_busting:false``, the ordered sequence of
    ``prompt_id`` values in ``a/raw.jsonl`` equals that in ``b/raw.jsonl``.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    overrides = {"seed": 42, "cache_busting": False}

    seqs: list[list[str]] = []
    for name in ("a", "b"):
        config = _short_run(cfg_base, port, overrides=overrides)
        out_dir = tmp_path / name
        result = runner.invoke(
            app,
            ["run", "--config", str(config), "--model", "sut", "--seed", "42", "--out", str(out_dir)],
        )
        assert result.exit_code == 0, result.stderr
        records = _read_jsonl(out_dir / "raw.jsonl")
        seqs.append([r["prompt_id"] for r in records])

    # The exact request count varies between two duration-based runs (timing), but
    # the seeded selection sequence must be identical up to the common length:
    # request k always selects the same prompt given the same seed (FR-033).
    common = min(len(seqs[0]), len(seqs[1]))
    assert common > 1, "need several requests to show a non-trivial sequence"
    assert seqs[0][:common] == seqs[1][:common], (
        f"seeded prompt_id sequence not reproducible: {seqs[0][:common]} != {seqs[1][:common]}"
    )


# ---------------------------------------------------------------------------
# E2E-087: cache_busting injects unique prefix per request
# ---------------------------------------------------------------------------


def test_e2e_087_cache_busting_unique_prefix(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A unique prefix is prepended to each request's user message.

    FR-034: with ``cache_busting:true`` and a single prompt reused across many
    requests, the outgoing user-message contents all differ (distinct UUID-like
    prefix per request) while the body *after* the prefix is identical.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)

    # One prompt forced for every request so any content difference is the prefix.
    prompts = _write_prompts(
        tmp_path / "one.yaml",
        [
            {
                "id": "solo-1",
                "category": "general",
                "isl_bucket": "short",
                "messages": [{"role": "user", "content": "FIXED-BODY-CONTENT"}],
            }
        ],
    )
    config = _short_run(cfg_base, port, overrides={"cache_busting": True})

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    # Exclude the deterministic pre-flight "ping" (tagged via its header) so only
    # the measured, cache-busted requests are compared.
    measured = [r for r in controller.requests if "x-llmbench-preflight" not in {k.lower() for k in r.headers}]
    contents = [_user_content(r.body) for r in measured]
    assert len(contents) >= 2, "need at least two requests to compare prefixes"
    # Every outgoing user message is distinct (unique prefix per request).
    assert len(set(contents)) == len(contents), f"duplicate user contents: {contents}"
    # The shared suffix (body after the prefix) is identical across requests.
    assert all(c.endswith("FIXED-BODY-CONTENT") for c in contents), contents
    prefixes = [c[: -len("FIXED-BODY-CONTENT")] for c in contents]
    assert all(p.strip() for p in prefixes), f"empty prefixes: {prefixes!r}"
    assert len(set(prefixes)) == len(prefixes), f"prefixes not unique: {prefixes!r}"


# ---------------------------------------------------------------------------
# E2E-088: ignore_eos + max_tokens deterministic OSL
# ---------------------------------------------------------------------------


def test_e2e_088_ignore_eos_max_tokens_deterministic_osl(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """ignore_eos + max_tokens yield a deterministic output length on every record.

    FR-035: every request body carries ``max_tokens:8`` and ``ignore_eos:true``
    (or an equivalent extra-body), and every record has ``output_tokens==8``.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port, overrides={"ignore_eos": True, "max_tokens": 8})

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    def _has_flag(body: dict[str, Any], key: str, value: Any) -> bool:
        if body.get(key) == value:
            return True
        extra = body.get("extra_body")
        return isinstance(extra, dict) and extra.get(key) == value

    assert controller.requests, "no requests captured"
    for recorded in controller.requests:
        assert _has_flag(recorded.body, "max_tokens", 8), recorded.body
        assert _has_flag(recorded.body, "ignore_eos", True), recorded.body

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written"
    assert all(r["output_tokens"] == 8 for r in records), [r["output_tokens"] for r in records]


# ---------------------------------------------------------------------------
# E2E-089: Built-in prompt library used by default
# ---------------------------------------------------------------------------


def test_e2e_089_builtin_prompt_library_default(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """With no ``--prompts``, records carry prompt_ids from the packaged library.

    FR-036: exit 0; every record has a non-empty ``prompt_id`` and a ``category``
    drawn from the packaged set; the run does not fail for lack of prompts.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written from default library"
    known_categories = {"coding", "synthesis", "tool-use", "vision", "general"}
    for record in records:
        assert record.get("prompt_id"), f"empty prompt_id: {record}"
        assert record.get("category") in known_categories, record.get("category")
    # At least one packaged prompt id is actually used.
    assert {r["prompt_id"] for r in records}, "no prompt_id present"


# ---------------------------------------------------------------------------
# E2E-090: --prompts override loads external prompts
# ---------------------------------------------------------------------------


def test_e2e_090_prompts_override(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``--prompts`` replaces the built-in library entirely.

    FR-036: every record has ``prompt_id=="custom-99"`` and no built-in ids
    appear.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    prompts = _write_prompts(
        tmp_path / "myprompts.yaml",
        [
            {
                "id": "custom-99",
                "category": "coding",
                "isl_bucket": "medium",
                "messages": [{"role": "user", "content": "write a quicksort"}],
            }
        ],
    )
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written"
    assert {r["prompt_id"] for r in records} == {"custom-99"}, {r["prompt_id"] for r in records}


# ---------------------------------------------------------------------------
# E2E-091: Empty prompts file aborts
# ---------------------------------------------------------------------------


def test_e2e_091_empty_prompts_aborts(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """An empty ``--prompts`` file aborts with ``no prompts loaded``.

    FR-036: exit non-zero; stderr contains ``no prompts loaded from empty.yaml``;
    no run data is written.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    empty = tmp_path / "empty.yaml"
    empty.write_text("[]\n", encoding="utf-8")
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(empty), "--out", str(out_dir)],
    )

    assert result.exit_code != 0, result.output
    assert "no prompts loaded from empty.yaml" in result.stderr
    assert not (out_dir / "raw.jsonl").exists()


# ---------------------------------------------------------------------------
# E2E-092: Request tagged category + ISL/OSL buckets
# ---------------------------------------------------------------------------


def test_e2e_092_request_tagged_category_and_buckets(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Each record carries category, isl_bucket, and a derived osl_bucket.

    FR-037: with a ``coding``/``isl_bucket:medium`` prompt and ``max_tokens:8``,
    every record has ``category=="coding"``, ``isl_bucket=="medium"`` and an
    ``osl_bucket`` derived from output_tokens (8 -> "short").
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    prompts = _write_prompts(
        tmp_path / "tagged.yaml",
        [
            {
                "id": "coding-medium",
                "category": "coding",
                "isl_bucket": "medium",
                "messages": [{"role": "user", "content": "refactor this function"}],
            }
        ],
    )
    config = _short_run(cfg_base, port, overrides={"max_tokens": 8})

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written"
    for record in records:
        assert record.get("category") == "coding", record.get("category")
        assert record.get("isl_bucket") == "medium", record.get("isl_bucket")
        assert record.get("osl_bucket") == "short", record.get("osl_bucket")


# ---------------------------------------------------------------------------
# E2E-093: One JSONL record per request
# ---------------------------------------------------------------------------


def test_e2e_093_one_record_per_request_unique(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``raw.jsonl`` has exactly one unique record per measured request.

    FR-048: line count equals the server-side measured request count (excluding
    the pre-flight request); each line is valid JSON with a unique ``request_id``;
    no duplicates.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    records = _read_jsonl(out_dir / "raw.jsonl")
    # Server counts every request including the single pre-flight; raw excludes it.
    measured = controller.request_count - 1
    assert measured >= 1, f"expected at least one measured request, got {measured}"
    assert len(records) == measured, f"{len(records)} records != {measured} measured requests"

    request_ids = [r["request_id"] for r in records]
    assert len(set(request_ids)) == len(request_ids), "duplicate request_id found"


# ---------------------------------------------------------------------------
# E2E-094: ITL summary default, full list only with --raw-itl
# ---------------------------------------------------------------------------


def test_e2e_094_itl_summary_default_list_with_flag(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """itl_summary by default; itl_list only with ``--raw-itl``; p95 agree.

    FR-020/049: without the flag every record has
    ``itl_summary{mean,p50,p95,p99,max}`` and no ``itl_list``; with ``--raw-itl``
    records also carry ``itl_list``; the summary p95 matches the list p95 (±2%).
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port)

    # Run 1: default (summary only).
    out_default = tmp_path / "r_default"
    result_d = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_default)],
    )
    assert result_d.exit_code == 0, result_d.stderr
    default_records = _read_jsonl(out_default / "raw.jsonl")
    assert default_records, "no records in default run"
    for record in default_records:
        summary = record.get("itl_summary")
        assert isinstance(summary, dict), f"missing itl_summary: {record}"
        assert {"mean", "p50", "p95", "p99", "max"} <= summary.keys(), summary
        assert "itl_list" not in record or record["itl_list"] is None

    # Run 2: with --raw-itl (full list present).
    out_raw = tmp_path / "r_raw"
    result_r = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--raw-itl", "--out", str(out_raw)],
    )
    assert result_r.exit_code == 0, result_r.stderr
    raw_records = _read_jsonl(out_raw / "raw.jsonl")
    assert raw_records, "no records in raw-itl run"
    checked = False
    for record in raw_records:
        itl_list = record.get("itl_list")
        assert isinstance(itl_list, list), f"missing itl_list with --raw-itl: {record}"
        if len(itl_list) >= 2:
            # Compute the expected p95 with the SAME method the implementation uses
            # (numpy linear-interpolation percentile) so the comparison is exact and
            # not sensitive to a nearest-rank-vs-interpolation mismatch under jitter.
            list_p95 = float(np.percentile(np.asarray(itl_list, dtype=float), 95))
            summary_p95 = record["itl_summary"]["p95"]
            tol = max(abs(list_p95) * 0.02, 1e-9)
            assert abs(summary_p95 - list_p95) <= tol, f"p95 mismatch: {summary_p95} vs {list_p95}"
            checked = True
    assert checked, "no record had enough ITL samples to compare p95"


# ---------------------------------------------------------------------------
# E2E-095: Parquet rollup + summary JSON at end
# ---------------------------------------------------------------------------


def test_e2e_095_rollup_parquet_and_summary(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A completed run writes a readable rollup.parquet and summary.json at end.

    FR-050: ``rollup.parquet`` is readable by pyarrow/DuckDB and its row count
    equals the ``raw.jsonl`` line count; ``summary.json`` carries ``levels[]`` and
    ``run_id``. (The "absent mid-run" clause cannot be observed via a CliRunner
    end-state; only the final artifacts are asserted here.)
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    raw_records = _read_jsonl(out_dir / "raw.jsonl")
    parquet_path = out_dir / "rollup.parquet"
    assert parquet_path.exists(), f"missing rollup.parquet: {parquet_path}"
    table = pq.read_table(parquet_path)
    assert table.num_rows == len(raw_records), f"{table.num_rows} parquet rows != {len(raw_records)} jsonl lines"

    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), f"missing summary.json: {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "run_id" in summary, summary.keys()
    assert isinstance(summary.get("levels"), list) and summary["levels"], summary.get("levels")


# ---------------------------------------------------------------------------
# E2E-096: Secrets/prompts not in OTel traces
# ---------------------------------------------------------------------------


def test_e2e_096_no_secret_or_prompt_in_traces(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """traces.jsonl records spans but leaks neither the api_key nor prompt text.

    FR-057: with api_key ``sk-secret-xyz`` and a known prompt string, the trace
    file exists with spans, but grep for the secret and for the prompt text both
    return zero matches.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    secret = "sk-secret-xyz"
    prompt_text = "UNIQUE-PROMPT-TEXT-MARKER-12345"
    prompts = _write_prompts(
        tmp_path / "secret_prompts.yaml",
        [
            {
                "id": "leaky-1",
                "category": "general",
                "isl_bucket": "short",
                "messages": [{"role": "user", "content": prompt_text}],
            }
        ],
    )
    config = _short_run(cfg_base, port, api_key=secret)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    traces_path = out_dir / "traces.jsonl"
    assert traces_path.exists(), f"missing traces.jsonl: {traces_path}"
    spans = _read_jsonl(traces_path)
    assert spans, "traces.jsonl has no spans"

    raw = traces_path.read_text(encoding="utf-8")
    assert raw.count(secret) == 0, "api key leaked into traces"
    assert raw.count(prompt_text) == 0, "prompt text leaked into traces"


# ---------------------------------------------------------------------------
# E2E-098: Structured logs emitted
# ---------------------------------------------------------------------------


def test_e2e_098_structured_logs_run_events(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Run emits JSON log lines including run_started and run_completed events.

    FR-056: stderr log lines are JSON objects with at least
    ``{timestamp, level, event}``; at least one ``run_started`` and one
    ``run_completed`` event are present; the lines parse as JSON.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = _short_run(cfg_base, port)

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--log-format", "json", "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    events: list[str] = []
    parsed_any = False
    for raw_line in result.stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        parsed_any = True
        assert {"timestamp", "level", "event"} <= obj.keys(), f"log line missing keys: {obj}"
        events.append(obj["event"])

    assert parsed_any, f"no JSON log lines on stderr: {result.stderr!r}"
    assert "run_started" in events, f"no run_started event in {events}"
    assert "run_completed" in events, f"no run_completed event in {events}"


# ---------------------------------------------------------------------------
# Default config locations (~/.config/llm-bench/)
# ---------------------------------------------------------------------------


def test_default_config_locations_under_config_dir(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Defaults live under ~/.config/llm-bench/; a present prompts.yaml is used."""
    assert str(DEFAULT_CONFIG_FILE).endswith("/.config/llm-bench/config.yaml")
    assert str(DEFAULT_PROMPTS_FILE).endswith("/.config/llm-bench/prompts.yaml")

    base_url, _controller = fake_sut
    # The autouse guard redirects DEFAULT_CONFIG_DIR to a tmp dir; dropping a
    # prompts.yaml there exercises the "no --prompts -> default file is used" path.
    config_dir = tmp_path / "default-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_prompts(
        config_dir / "prompts.yaml",
        [{"id": "cfgdir-prompt", "category": "general", "messages": [{"role": "user", "content": "hi"}]}],
    )

    config = _short_run(cfg_base, _port_of(base_url))
    out_dir = tmp_path / "r"
    result = runner.invoke(app, ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)])
    assert result.exit_code == 0, result.stderr
    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records
    assert all(r["prompt_id"] == "cfgdir-prompt" for r in records)


def test_default_out_dir_used_when_omitted(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no --out, a timestamped run dir is created under the data home."""
    base_url, _controller = fake_sut
    runs_root = tmp_path / "datahome-runs"
    monkeypatch.setattr("llm_bench.config.DEFAULT_RUNS_DIR", runs_root)

    config = _short_run(cfg_base, _port_of(base_url))
    result = runner.invoke(app, ["run", "--config", str(config), "--model", "sut"])  # no --out
    assert result.exit_code == 0, result.stderr
    assert "writing run artifacts to" in result.stdout

    subdirs = [p for p in runs_root.iterdir() if p.is_dir()] if runs_root.exists() else []
    assert len(subdirs) == 1, subdirs
    assert (subdirs[0] / "raw.jsonl").exists()
    assert (subdirs[0] / "summary.json").exists()
