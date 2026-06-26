"""Custom dashboards: pivot a run's per-request records into chart series.

A dashboard is a list of panels. Each panel pivots the steady-window, successful
requests of one run:

* ``x``     - a dimension column on the x-axis (e.g. ``level_or_rate``),
* ``group`` - an optional second dimension; one series per distinct value,
* ``values``- one or more ``{metric, agg}`` pairs (each becomes a series),
* ``chart`` - ``line`` / ``bar`` / ``auto`` (auto = line for a numeric x, else bar).

Dashboards live as YAML files under ``~/.config/llm-bench/dashboards/`` and are
edited in the report's Dashboards tab. The pivot is computed in pure Python (numpy
for percentiles) over ``raw.jsonl`` - no database and no ad-hoc SQL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from pathlib import Path


class DashboardError(Exception):
    """A user-facing dashboard definition or rendering error."""


# Dimensions usable for x / group (categorical or ordinal record columns).
DIMENSIONS: tuple[str, ...] = (
    "level_or_rate",
    "model",
    "category",
    "isl_bucket",
    "osl_bucket",
    "mode",
    "outcome",
    "prompt_id",
)

# Metric name -> (record column, scale, unit) for the value series.
_METRICS: dict[str, tuple[str, float, str]] = {
    "ttft": ("ttft", 1000.0, "ms"),
    "tpot": ("tpot", 1000.0, "ms"),
    "e2e": ("e2e", 1000.0, "ms"),
    "tt2t": ("tt2t", 1000.0, "ms"),
    "normalized_latency": ("normalized_latency", 1000.0, "ms"),
    "input_tokens": ("prompt_tokens", 1.0, "tok"),
    "output_tokens": ("output_tokens", 1.0, "tok"),
    "cached_tokens": ("cached_tokens", 1.0, "tok"),
    "reasoning_tokens": ("reasoning_tokens", 1.0, "tok"),
    "cost_usd": ("cost_usd", 1.0, "usd"),
    "sim_score": ("sim_score", 1.0, "score"),
}
# Derived throughput rates computed over each group's steady window (agg ignored).
_DERIVED: dict[str, str] = {"rps": "req/s", "system_tok_s": "tok/s"}
METRICS: tuple[str, ...] = (*_METRICS, *_DERIVED)

# Aggregations -> percentile q (None = a non-percentile reducer handled below).
_PERCENTILES: dict[str, float] = {"p50": 50, "p90": 90, "p95": 95, "p99": 99}
AGGS: tuple[str, ...] = ("p50", "p90", "p95", "p99", "mean", "max", "min", "count")

_CHART_TYPES: frozenset[str] = frozenset({"line", "bar", "auto"})
# Buckets sort by size, not alphabetically.
_BUCKET_ORDER: dict[str, int] = {"short": 0, "medium": 1, "long": 2}

# ---------------------------------------------------------------------------
# Help text (rendered in the Dashboards tab; kept beside the registries above)
# ---------------------------------------------------------------------------

# What each dimension means as an x-axis or a 'group' (series-splitter).
DIMENSION_HELP: dict[str, str] = {
    "level_or_rate": "The load step of the sweep: number of concurrent clients (closed mode) or arrival rate in req/s (open mode). The usual x-axis.",
    "model": "The benchmarked model id. Constant within a single run (a run targets one model).",
    "category": "The prompt's category: coding, synthesis, tool-use, vision, or general.",
    "isl_bucket": "Input-length class of the prompt (by prompt tokens): short / medium / long. Use it to see how input size affects TTFT.",
    "osl_bucket": "Output-length class of the response (by output tokens): short / medium / long. Use it to see how generation length affects E2E latency.",
    "mode": "Load mode (closed or open). Constant within a single run.",
    "outcome": "Per-request result. Charts use steady successful requests only, so this is effectively constant ('success').",
    "prompt_id": "The exact prompt sent. Useful to compare individual prompts.",
}

# What each value metric measures (and its unit). Derived rates ignore the agg.
METRIC_HELP: dict[str, str] = {
    "ttft": "Time To First Token (ms): send → first content token. The prefill latency; grows with input length.",
    "tpot": "Time Per Output Token (ms): the steady generation pace, (E2E - TTFT) / (output_tokens - 1).",
    "e2e": "End-to-end latency (ms): send → last token.",
    "tt2t": "Time to second token (ms): the first inter-token gap after TTFT.",
    "normalized_latency": "E2E latency normalised by output length (ms): isolates speed from how much was generated.",
    "input_tokens": "Prompt (input) tokens per request, from the server usage.",
    "output_tokens": "Completion (output) tokens per request, from the server usage.",
    "cached_tokens": "Prompt tokens served from the provider's cache (when reported).",
    "reasoning_tokens": "Reasoning / thinking tokens (when the model reports them).",
    "cost_usd": "Estimated cost per request (only non-zero when the model has pricing in the config).",
    "sim_score": "Embedding cosine similarity of the answer to the prompt's expected_output (quality eval).",
    "rps": "Requests per second over the x-group's steady window. A rate, so the chosen aggregation is ignored.",
    "system_tok_s": "Aggregate output tokens/second across concurrent requests (server capacity). A rate; aggregation ignored.",
}

# How each aggregation reduces a metric's per-request values.
AGG_HELP: dict[str, str] = {
    "p50": "Median (50th percentile): the typical value.",
    "p90": "90th percentile.",
    "p95": "95th percentile.",
    "p99": "99th percentile: the slow tail.",
    "mean": "Arithmetic average (sensitive to outliers).",
    "max": "Largest value.",
    "min": "Smallest value.",
    "count": "Number of requests (the metric value is ignored).",
}


@dataclass(frozen=True)
class ValueSpec:
    """One ``{metric, agg}`` series within a panel."""

    metric: str
    agg: str


@dataclass(frozen=True)
class Panel:
    """A single dashboard panel (a pivot over one run's steady requests)."""

    title: str
    x: str
    values: tuple[ValueSpec, ...]
    group: str | None = None
    chart: str = "auto"


@dataclass
class PanelResult:
    """Computed series for a panel, ready to render as a chart + table."""

    title: str
    x_label: str
    x_values: list[Any]
    series: list[dict[str, Any]] = field(default_factory=list)
    chart: str = "line"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DashboardError(message)


def parse_dashboard(text: str, source: str = "dashboard") -> list[Panel]:
    """Parse and validate a dashboard YAML document into panels."""
    try:
        raw = yaml.safe_load(text) if text.strip() else []
    except yaml.YAMLError as exc:
        raise DashboardError(f"invalid dashboard YAML: {exc}") from exc
    _require(isinstance(raw, list), f"{source} must be a top-level list of panels")
    panels = [_parse_panel(item, index) for index, item in enumerate(raw)]
    _require(bool(panels), "a dashboard needs at least one panel")
    return panels


def _parse_panel(item: Any, index: int) -> Panel:
    _require(isinstance(item, dict), f"panel #{index + 1} is malformed")
    title = str(item.get("title") or f"Panel {index + 1}")
    x = str(item.get("x") or "")
    _require(x in DIMENSIONS, f"panel {title!r}: x must be one of {list(DIMENSIONS)}")
    group_raw = item.get("group")
    group = str(group_raw) if group_raw else None
    _require(group is None or group in DIMENSIONS, f"panel {title!r}: group must be a dimension")
    chart = str(item.get("chart") or "auto")
    _require(chart in _CHART_TYPES, f"panel {title!r}: chart must be line/bar/auto")
    values = tuple(_parse_value(v, title) for v in item.get("values", []))
    _require(bool(values), f"panel {title!r}: at least one value is required")
    return Panel(title=title, x=x, values=values, group=group, chart=chart)


def _parse_value(item: Any, title: str) -> ValueSpec:
    _require(isinstance(item, dict), f"panel {title!r}: each value must be a mapping")
    metric = str(item.get("metric") or "")
    agg = str(item.get("agg") or "p50")
    _require(metric in _METRICS or metric in _DERIVED, f"panel {title!r}: unknown metric {metric!r}")
    _require(agg in AGGS, f"panel {title!r}: unknown agg {agg!r}")
    return ValueSpec(metric=metric, agg=agg)


def read_steady_records(run_dir: Path) -> list[dict[str, Any]]:
    """Read a run's steady-window successful requests from ``raw.jsonl``."""
    path = run_dir / "raw.jsonl"
    records: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DashboardError(f"cannot read {path}: {exc}") from exc
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("phase") == "steady" and row.get("outcome") == "success":
            records.append(row)
    return records


def _aggregate(values: list[float], agg: str) -> float | None:
    if agg == "count":
        return float(len(values))
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if agg in _PERCENTILES:
        return float(np.percentile(array, _PERCENTILES[agg]))
    if agg == "mean":
        return float(np.mean(array))
    if agg == "max":
        return float(np.max(array))
    return float(np.min(array))


def _sort_key(values: set[Any]) -> list[Any]:
    """Order x/group values: numeric ascending, else bucket order, else alphabetic."""
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=lambda v: (_BUCKET_ORDER.get(str(v), 99), str(v)))


def compute_panel(records: list[dict[str, Any]], panel: Panel) -> PanelResult:
    """Pivot ``records`` into the panel's chart series (one per value x group)."""
    x_values = _sort_key({row.get(panel.x) for row in records if row.get(panel.x) is not None})
    groups = (
        _sort_key({row.get(panel.group) for row in records if row.get(panel.group) is not None})
        if panel.group
        else [None]
    )

    series: list[dict[str, Any]] = []
    for spec in panel.values:
        for group_value in groups:
            cells = [
                [
                    row
                    for row in records
                    if row.get(panel.x) == x_value and (panel.group is None or row.get(panel.group) == group_value)
                ]
                for x_value in x_values
            ]
            points, name = _series_for(spec, cells)
            if group_value is not None:
                name += f" [{group_value}]"
            series.append({"name": name, "values": points})

    chart = panel.chart
    if chart == "auto":
        chart = "line" if _is_numeric(x_values) else "bar"
    return PanelResult(title=panel.title, x_label=panel.x, x_values=x_values, series=series, chart=chart)


def _series_for(spec: ValueSpec, cells: list[list[dict[str, Any]]]) -> tuple[list[float | None], str]:
    """Reduce per-x record groups into a series of values and its legend name."""
    if spec.metric in _DERIVED:
        points = [_rate(rows, spec.metric) for rows in cells]
        return points, f"{spec.metric} ({_DERIVED[spec.metric]})"
    column, scale, unit = _METRICS[spec.metric]
    points = [
        _aggregate([float(row[column]) * scale for row in rows if isinstance(row.get(column), (int, float))], spec.agg)
        for rows in cells
    ]
    unit_suffix = f" ({unit})" if unit in {"ms", "usd"} else ""
    return points, f"{spec.metric} {spec.agg}{unit_suffix}"


def _rate(rows: list[dict[str, Any]], metric: str) -> float | None:
    """Compute a throughput rate (``rps`` / ``system_tok_s``) over a group's window."""
    starts = [row["t_start"] for row in rows if isinstance(row.get("t_start"), (int, float))]
    ends = [
        row["t_start"] + row["e2e"]
        for row in rows
        if isinstance(row.get("t_start"), (int, float)) and isinstance(row.get("e2e"), (int, float))
    ]
    if not starts or not ends:
        return None
    window = float(max(ends) - min(starts))
    if window <= 0:
        return None
    if metric == "rps":
        return len(rows) / window
    return float(sum(row.get("output_tokens", 0) for row in rows)) / window


def _is_numeric(values: list[Any]) -> bool:
    try:
        [float(v) for v in values]
    except (TypeError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Form <-> YAML for the editor
# ---------------------------------------------------------------------------


def dashboard_to_form(text: str) -> list[dict[str, Any]]:
    """Parse a dashboard YAML into the editor's structured panel shape."""
    panels = parse_dashboard(text)
    return [
        {
            "title": panel.title,
            "x": panel.x,
            "group": panel.group or "",
            "chart": panel.chart,
            "values": [{"metric": v.metric, "agg": v.agg} for v in panel.values],
        }
        for panel in panels
    ]


def build_dashboard_yaml(items: list[Any]) -> str:
    """Reconstruct dashboard YAML from the editor payload (validates structure)."""
    _require(isinstance(items, list) and bool(items), "add at least one panel")
    panels: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        _require(isinstance(item, dict), f"panel #{index + 1} is malformed")
        title = str(item.get("title") or f"Panel {index + 1}").strip()
        x = str(item.get("x") or "")
        _require(x in DIMENSIONS, f"panel {title!r}: pick a valid x dimension")
        values = [
            {"metric": v.get("metric"), "agg": v.get("agg") or "p50"}
            for v in item.get("values", [])
            if isinstance(v, dict) and v.get("metric")
        ]
        _require(bool(values), f"panel {title!r}: add at least one value")
        panel: dict[str, Any] = {"title": title, "x": x}
        group = str(item.get("group") or "")
        if group:
            panel["group"] = group
        chart = str(item.get("chart") or "auto")
        if chart != "auto":
            panel["chart"] = chart
        panel["values"] = values
        panels.append(panel)
    # Validate the whole thing round-trips before returning.
    text = "# Dashboard edited in the llm-bench report 'Dashboards' tab.\n" + yaml.safe_dump(
        panels, sort_keys=False, allow_unicode=True, width=100
    )
    parse_dashboard(text)
    return text


STARTER_DASHBOARD: str = """\
# Default dashboard scaffolded by 'llm-bench init'. Edit in the Dashboards tab.
# Each panel pivots a run's steady requests: x (dimension), optional group, and
# one or more {metric, agg} values. chart: line | bar | auto.
- title: Latency vs load
  x: level_or_rate
  values:
    - {metric: ttft, agg: p50}
    - {metric: ttft, agg: p99}
    - {metric: e2e, agg: p50}
    - {metric: e2e, agg: p99}
- title: Throughput vs load
  x: level_or_rate
  values:
    - {metric: system_tok_s, agg: mean}
    - {metric: rps, agg: mean}
- title: E2E latency by output length
  x: level_or_rate
  group: osl_bucket
  values:
    - {metric: e2e, agg: p50}
- title: Tokens per request vs load
  x: level_or_rate
  values:
    - {metric: input_tokens, agg: p50}
    - {metric: output_tokens, agg: p50}
"""
