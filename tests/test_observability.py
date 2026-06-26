"""Tests for the logging and tracing configuration modules.

These exercise the shipped observability helpers end-to-end against temporary
directories so the JSONL trace export and the dual log handlers are verified.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
from opentelemetry import trace

from llm_bench.logging_config import setup_logging
from llm_bench.tracing import configure_tracing, trace_span

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_tracer_provider() -> None:
    """Reset the global OpenTelemetry tracer provider before each test.

    OpenTelemetry refuses to override an already-set global provider, so the
    sentinel guarding it is cleared to let each test bind a fresh exporter that
    writes into its own temporary directory.
    """
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    """setup_logging creates <app_name>.log and routes records to it."""
    setup_logging("llm-bench", log_dir=tmp_path)
    logging.getLogger("test").info("hello world")
    for handler in logging.getLogger().handlers:
        handler.flush()

    log_file = tmp_path / "llm-bench.log"
    assert log_file.exists()
    assert "hello world" in log_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected"),
    [(False, False, logging.INFO), (True, False, logging.DEBUG), (False, True, logging.WARNING)],
)
def test_setup_logging_levels(tmp_path: Path, *, verbose: bool, quiet: bool, expected: int) -> None:
    """Verbose and quiet flags select DEBUG and WARNING respectively."""
    setup_logging("llm-bench", log_dir=tmp_path, verbose=verbose, quiet=quiet)
    assert logging.getLogger().level == expected


def test_configure_tracing_exports_jsonl(tmp_path: Path) -> None:
    """configure_tracing writes spans as JSONL with attributes and events."""
    configure_tracing("llm-bench", log_dir=tmp_path)

    with trace_span("llm.call", {"model": "fake/model", "tokens": 8}) as span:
        span.add_event("first_token")

    otel_file = tmp_path / "llm-bench-otel.log"
    assert otel_file.exists()
    records = [json.loads(line) for line in otel_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    record = records[-1]
    assert record["name"] == "llm.call"
    assert record["attributes"]["model"] == "fake/model"
    assert any(event["name"] == "first_token" for event in record.get("events", []))


def test_trace_span_records_exception(tmp_path: Path) -> None:
    """trace_span marks the span as errored and re-raises on exceptions."""
    configure_tracing("llm-bench", log_dir=tmp_path)

    with pytest.raises(ValueError, match="boom"), trace_span("bench.level"):
        raise ValueError("boom")

    otel_file = tmp_path / "llm-bench-otel.log"
    records = [json.loads(line) for line in otel_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r["name"] == "bench.level" and r["status"] == "ERROR" for r in records)
