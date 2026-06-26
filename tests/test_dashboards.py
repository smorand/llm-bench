"""Tests for the dashboard pivot engine and form<->YAML helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from llm_bench.dashboards import (
    STARTER_DASHBOARD,
    DashboardError,
    build_dashboard_yaml,
    compute_panel,
    dashboard_to_form,
    parse_dashboard,
    read_steady_records,
)

if TYPE_CHECKING:
    from pathlib import Path


def _rec(level: int, e2e: float, out: int, bucket: str, *, phase: str = "steady", outcome: str = "success") -> dict:
    return {
        "level_or_rate": level,
        "e2e": e2e,
        "output_tokens": out,
        "prompt_tokens": 100,
        "osl_bucket": bucket,
        "phase": phase,
        "outcome": outcome,
    }


def test_parse_dashboard_valid_and_errors() -> None:
    """A valid dashboard parses; bad x/metric/empty panels raise clear errors."""
    panels = parse_dashboard(STARTER_DASHBOARD)
    assert [p.title for p in panels] == [
        "Latency vs load",
        "Throughput vs load",
        "E2E latency by output length",
        "Tokens per request vs load",
    ]
    assert panels[2].group == "osl_bucket"

    with pytest.raises(DashboardError, match="x must be one of"):
        parse_dashboard("- {title: bad, x: nope, values: [{metric: e2e, agg: p50}]}")
    with pytest.raises(DashboardError, match="unknown metric"):
        parse_dashboard("- {title: bad, x: level_or_rate, values: [{metric: nope, agg: p50}]}")
    with pytest.raises(DashboardError, match="at least one value"):
        parse_dashboard("- {title: bad, x: level_or_rate, values: []}")
    with pytest.raises(DashboardError, match="list of panels"):
        parse_dashboard("just a string")


def test_read_steady_records_filters(tmp_path: Path) -> None:
    """Only steady-phase successful requests are returned."""
    run = tmp_path / "r1"
    run.mkdir()
    lines = [
        _rec(1, 1.0, 10, "short"),
        _rec(1, 2.0, 20, "short", phase="warmup"),
        _rec(1, 3.0, 30, "short", outcome="timeout"),
    ]
    (run / "raw.jsonl").write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    records = read_steady_records(run)
    assert len(records) == 1
    assert records[0]["e2e"] == 1.0


def test_compute_panel_x_series_and_units() -> None:
    """A simple panel pivots by x with ms-scaled latency and a numeric->line chart."""
    records = [_rec(1, 1.0, 10, "short"), _rec(1, 3.0, 10, "short"), _rec(2, 2.0, 10, "short")]
    panel = parse_dashboard("- {title: L, x: level_or_rate, values: [{metric: e2e, agg: p50}]}")[0]
    result = compute_panel(records, panel)
    assert result.chart == "line"  # x is numeric
    assert result.x_values == [1.0, 2.0]
    # e2e p50 at level 1 = median(1,3)=2.0s -> 2000ms; level 2 = 2000ms
    assert result.series[0]["values"] == [2000.0, 2000.0]
    assert "ms" in result.series[0]["name"]


def test_compute_panel_grouping_one_series_per_group() -> None:
    """A group dimension yields one series per distinct value."""
    records = [_rec(1, 1.0, 10, "short"), _rec(1, 4.0, 200, "long"), _rec(2, 2.0, 10, "short")]
    panel = parse_dashboard("- {title: G, x: level_or_rate, group: osl_bucket, values: [{metric: e2e, agg: max}]}")[0]
    result = compute_panel(records, panel)
    names = [s["name"] for s in result.series]
    assert any("[short]" in n for n in names)
    assert any("[long]" in n for n in names)


def test_compute_panel_categorical_x_is_bar() -> None:
    """A categorical x auto-selects a bar chart and orders buckets by size."""
    records = [_rec(1, 1.0, 10, "long"), _rec(2, 2.0, 10, "short"), _rec(2, 2.0, 10, "medium")]
    panel = parse_dashboard("- {title: B, x: osl_bucket, values: [{metric: output_tokens, agg: mean}]}")[0]
    result = compute_panel(records, panel)
    assert result.chart == "bar"
    assert result.x_values == ["short", "medium", "long"]  # bucket order, not alphabetic


def test_compute_panel_derived_throughput() -> None:
    """rps / system_tok_s are computed as rates over each x group's steady window."""
    records = [
        {**_rec(1, 1.0, 100, "short"), "t_start": 0.0},
        {**_rec(1, 1.0, 100, "short"), "t_start": 1.0},  # window = (1.0+1.0) - 0.0 = 2s, 2 reqs
    ]
    panel = parse_dashboard(
        "- {title: T, x: level_or_rate, values: [{metric: rps, agg: mean}, {metric: system_tok_s, agg: mean}]}"
    )[0]
    result = compute_panel(records, panel)
    assert result.series[0]["values"] == [1.0]  # 2 requests / 2s window
    assert result.series[1]["values"] == [100.0]  # 200 output tokens / 2s window
    assert "req/s" in result.series[0]["name"]


def test_dashboard_form_round_trip() -> None:
    """dashboard_to_form -> build_dashboard_yaml reproduces the panels."""
    form = dashboard_to_form(STARTER_DASHBOARD)
    payload = json.loads(json.dumps(form))  # simulate the JSON wire
    rebuilt = build_dashboard_yaml(payload)
    panels = parse_dashboard(rebuilt)
    assert [p.title for p in panels] == [p["title"] for p in form]
    assert panels[2].group == "osl_bucket"


def test_build_dashboard_yaml_validation() -> None:
    """Empty panels and missing x/values raise before anything is written."""
    with pytest.raises(DashboardError, match="at least one panel"):
        build_dashboard_yaml([])
    with pytest.raises(DashboardError, match="valid x dimension"):
        build_dashboard_yaml([{"title": "x", "values": [{"metric": "e2e", "agg": "p50"}]}])
    with pytest.raises(DashboardError, match="add at least one value"):
        build_dashboard_yaml([{"title": "x", "x": "level_or_rate", "values": []}])
