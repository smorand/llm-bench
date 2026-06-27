"""Local report browser: a tiny web server that presents run results simply.

``llm-bench serve`` starts a local HTTP server (IBM Carbon styling),
opens the browser, and shows, in three tabs:

* Reports - per-run recap (latency / throughput / goodput tables and SVG charts)
  with a drop-down to switch between every run under the runs directory.
* Run - pick a model from the config and launch a benchmark, with a live
  progress bar; the run executes as a subprocess and lands in the runs directory.
* Prompts - choose, edit, or create a prompt-library file under the prompts
  directory; saving validates the file before writing it.

Everything is server-rendered, self-contained HTML with inline CSS/SVG and no
external assets (bar an optional IBM Plex web font, with a system fallback), so a
page never depends on a CDN and never renders blank.
"""

from __future__ import annotations

import json
import re
import subprocess  # nosec B404  (used to launch our own CLI with a fixed argv, never a shell)
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import yaml

from llm_bench.config import default_run_dir
from llm_bench.dashboards import (
    AGG_HELP,
    AGGS,
    DIMENSION_HELP,
    DIMENSIONS,
    METRIC_HELP,
    METRICS,
    DashboardError,
    build_dashboard_yaml,
    compute_panel,
    dashboard_to_form,
    parse_dashboard,
    read_steady_records,
)
from llm_bench.prompts import PromptError, parse_prompts
from llm_bench.runner import parse_duration

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# IBM Carbon categorical palette for chart series.
_SERIES_COLORS = ("#0f62fe", "#da1e28", "#198038", "#8a3ffc", "#007d79", "#ff832b")

# A polyline needs at least two points; a single point renders as a lone dot.
_MIN_LINE_POINTS = 2

# Run-form presets (the user can also type extra comma-separated values).
_CONCURRENCY_PRESETS = (1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100, 128, 150, 200, 256, 300, 400, 500)
_CONCURRENCY_DEFAULT = frozenset({1, 2, 4, 8, 16, 32})
_RATE_PRESETS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
_RATE_DEFAULT = frozenset({1, 2, 5, 10, 20})
# (cli value, display label) for the single-choice duration picker.
_DURATION_CHOICES = (
    ("30s", "30 s"),
    ("1m", "1 min"),
    ("2m", "2 min"),
    ("3m", "3 min"),
    ("5m", "5 min"),
    ("10m", "10 min"),
    ("15m", "15 min"),
    ("20m", "20 min"),
    ("30m", "30 min"),
    ("60m", "60 min"),
    ("120m", "120 min"),
)
_DURATION_DEFAULT = "30s"


# One-line explanations surfaced by the (i) popovers.
_FIELD_INFO: dict[str, str] = {
    "Model": "Registry entry from your config.yaml to benchmark (its API key is resolved from env vars at run time).",
    "Mode": "closed = N parallel clients (concurrency sweep); open = fixed Poisson arrival rate in req/s. Open gives honest tail latency under saturation; closed is optimistic.",
    "Concurrency": "Closed-loop levels: each is a number of parallel clients, run one after another. Tick presets and/or add extra comma-separated values.",
    "Arrival rate": "Open-loop arrival rates in req/s, run one after another. Tick presets and/or add extra comma-separated values.",
    "Duration": "How long load is held at each level/rate (the per-level warmup and cooldown windows come from the config).",
    "Max tokens": "Cap on output tokens generated per request. Strongly affects end-to-end latency and throughput.",
    "Temperature": "Sampling temperature (0 = deterministic, higher = more random).",
    "SLO profile": "Threshold set (TTFT / TPOT / E2E) used to compute goodput. 'interactive' is strict, 'relaxed' is lenient.",
    "Seed": "Seed for reproducible prompt selection: the same seed replays the same prompt sequence.",
    "Prompts": "Which prompt library to send. Pick a file from ~/.config/llm-bench/prompts/, or the built-in default. Manage files in the Prompts tab.",
    "Quality eval": "Optional output-quality scoring (async, never perturbs timing). 'embedding' = cosine vs the prompt's expected_output; 'judge' = an LLM grades it. Both fill the quality_score (0..1) metric. Only prompts that declare an expected_output are scored.",
    "Judge model": "Which registry model grades the answers. '— from config —' keeps evaluation.judge.model; otherwise the chosen model's endpoint/key are used as the judge.",
    "Judge rubric": "How the judge scores: 'score' = the model returns a 0..1 number; 'three_level' = correct/partial/incorrect; 'binary' = pass/fail (categorical verdicts are mapped to 0..1 too).",
    "Embedding model": "How to embed for cosine scoring. 'local · CPU/GPU' = built-in fastembed (no server to run; downloads the model once). Or a registry model that serves /v1/embeddings (most chat gateways do not). '— from config —' keeps evaluation.embedding.",
}


@dataclass(frozen=True)
class RunRequest:
    """A validated launch request assembled from the Run-tab form."""

    model: str
    mode: str = "closed"
    load: str = ""  # comma list: concurrency levels (closed) or arrival rates (open)
    duration: str = ""
    max_tokens: str = ""
    temperature: str = ""
    slo_profile: str = ""
    seed: str = ""
    prompts: str = ""  # absolute path to a prompts file, or "" for the default library
    eval_method: str = ""  # "", "embedding", or "judge"
    judge_model: str = ""  # registry entry name to use as the judge
    judge_rubric: str = ""  # "", "score", "three_level", or "binary"
    embedding_model: str = ""  # registry entry name to use as the embeddings endpoint


def _info(label: str, infos: dict[str, str]) -> str:
    """Return an ``(i)`` popover button for ``label`` if a tip exists, else ''."""
    tip = infos.get(label)
    if not tip:
        return ""
    return f"<button class='i' type='button' onclick='tip(this)' data-tip=\"{escape(tip)}\">i</button>"


class ReportServeError(Exception):
    """Raised when a run cannot be resolved for serving."""


# ---------------------------------------------------------------------------
# Run discovery and resolution
# ---------------------------------------------------------------------------


def iter_runs(runs_dir: Path) -> list[Path]:
    """Return run directories holding a ``summary.json``, newest name first."""
    if not runs_dir.is_dir():
        return []
    runs = [child for child in runs_dir.iterdir() if child.is_dir() and (child / "summary.json").is_file()]
    return sorted(runs, key=lambda directory: directory.name, reverse=True)


def resolve_run(arg: str, runs_dir: Path) -> Path:
    """Resolve a run argument to a directory.

    Accepts a full path or a bare run name looked up under ``runs_dir``. Raises
    :class:`ReportServeError` when neither exists.
    """
    candidate = Path(arg).expanduser()
    if candidate.is_dir():
        return candidate
    under_runs = runs_dir / arg
    if under_runs.is_dir():
        return under_runs
    raise ReportServeError(f"run not found: {arg!r} (looked in {runs_dir})")


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty mapping when absent or invalid."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping without resolving ``$ENV:`` references, empty on error."""
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def list_config_models(config_path: Path | None) -> list[dict[str, str]]:
    """List ``{name, model}`` registry entries from the raw config (no ``$ENV:``).

    Reading names without resolving secrets means the Run tab works even when the
    model's API-key env vars are not set in the server's environment.
    """
    if config_path is None:
        return []
    raw = _load_yaml(config_path)
    models = raw.get("models", [])
    entries: list[dict[str, str]] = []
    for entry in models if isinstance(models, list) else []:
        if isinstance(entry, dict) and entry.get("name"):
            entries.append({"name": str(entry["name"]), "model": str(entry.get("model", ""))})
    return entries


# ---------------------------------------------------------------------------
# Prompt-file management
# ---------------------------------------------------------------------------

_PROMPT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def list_prompt_files(prompts_dir: Path | None) -> list[str]:
    """List ``*.yaml`` file names in the prompts directory, sorted."""
    if prompts_dir is None or not prompts_dir.is_dir():
        return []
    return sorted(p.name for p in prompts_dir.glob("*.yaml") if p.is_file())


def safe_prompt_path(prompts_dir: Path, name: str) -> Path:
    """Resolve ``name`` to a ``.yaml`` file inside ``prompts_dir`` (no traversal).

    Raises :class:`ReportServeError` on an unsafe name or one that escapes the dir.
    """
    stem = name.strip()
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    if not stem or not _PROMPT_NAME_RE.match(stem):
        raise ReportServeError(f"invalid prompt file name: {name!r}")
    target = (prompts_dir / f"{stem}.yaml").resolve()
    if target.parent != prompts_dir.resolve():
        raise ReportServeError(f"prompt file must live in {prompts_dir}")
    return target


def read_prompt_file(prompts_dir: Path, name: str) -> str:
    """Return the text of a prompts file in ``prompts_dir`` (raises if absent)."""
    target = safe_prompt_path(prompts_dir, name)
    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportServeError(f"cannot read {name}: {exc}") from exc


def save_prompt_file(prompts_dir: Path, name: str, content: str) -> str:
    """Validate ``content`` as a prompt library and write it; return the file name.

    Raises :class:`ReportServeError` when the name is unsafe or the content is not
    a valid prompt set, so invalid YAML is never persisted.
    """
    target = safe_prompt_path(prompts_dir, name)
    try:
        parse_prompts(content, target.name)
    except PromptError as exc:
        raise ReportServeError(f"not a valid prompt set: {exc}") from exc
    prompts_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target.name


def save_dashboard_file(dashboards_dir: Path, name: str, content: str) -> str:
    """Validate ``content`` as a dashboard and write it; return the file name."""
    target = safe_prompt_path(dashboards_dir, name)
    try:
        parse_dashboard(content, target.name)
    except DashboardError as exc:
        raise ReportServeError(f"not a valid dashboard: {exc}") from exc
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target.name


class _BlockDumper(yaml.SafeDumper):
    """YAML dumper that renders multi-line strings as readable ``|`` blocks."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> Any:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_BlockDumper.add_representer(str, _represent_str)


def _message_to_form(message: dict[str, Any]) -> dict[str, Any]:
    """Normalise one chat message to the editor shape (string or JSON content)."""
    content = message.get("content")
    if isinstance(content, str):
        return {"role": str(message.get("role", "user")), "content": content, "json": False}
    # Multimodal / structured content is preserved verbatim as JSON text.
    return {"role": str(message.get("role", "user")), "content": json.dumps(content, ensure_ascii=False), "json": True}


def prompts_to_form(text: str) -> list[dict[str, Any]]:
    """Parse prompt-library YAML into the structured shape the editor renders.

    Returns one entry per prompt with string fields plus ``messages`` (role +
    string/JSON content) and raw-YAML ``tools``/``tool_results`` escape hatches.
    """
    raw = yaml.safe_load(text) if text.strip() else []
    if not isinstance(raw, list):
        raise ReportServeError("prompt file must hold a top-level list of prompts")
    form: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        messages = [_message_to_form(m) for m in item.get("messages", []) if isinstance(m, dict)]
        form.append(
            {
                "id": str(item.get("id", "")),
                "category": str(item.get("category", "general")),
                "isl_bucket": str(item.get("isl_bucket", "short")),
                "messages": messages,
                "expected_output": str(item.get("expected_output") or ""),
                "tools": yaml.safe_dump(item["tools"], sort_keys=False).strip() if item.get("tools") else "",
                "tool_results": (
                    yaml.safe_dump(item["tool_results"], sort_keys=False).strip() if item.get("tool_results") else ""
                ),
            }
        )
    return form


def _form_message_to_mapping(message: dict[str, Any], prompt_id: str) -> dict[str, Any] | None:
    """Rebuild one chat message from the editor shape, or ``None`` when empty."""
    role = str(message.get("role", "user"))
    content = message.get("content", "")
    if message.get("json"):
        try:
            return {"role": role, "content": json.loads(content)}
        except (json.JSONDecodeError, TypeError) as exc:
            raise ReportServeError(f"prompt {prompt_id!r}: message content is not valid JSON") from exc
    if not isinstance(content, str) or not content.strip():
        return None
    return {"role": role, "content": content}


def _form_yaml_field(raw: str, prompt_id: str, field: str, expect: type) -> Any:
    """Parse a raw-YAML ``tools``/``tool_results`` field, validating its type."""
    if not raw or not raw.strip():
        return None
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ReportServeError(f"prompt {prompt_id!r}: {field} is not valid YAML") from exc
    if not isinstance(value, expect):
        raise ReportServeError(f"prompt {prompt_id!r}: {field} must be a {expect.__name__}")
    return value


def build_prompts_yaml(items: list[Any]) -> str:
    """Reconstruct prompt-library YAML from the editor's structured payload.

    Raises :class:`ReportServeError` with a per-prompt message on invalid input.
    """
    if not isinstance(items, list) or not items:
        raise ReportServeError("add at least one prompt")
    prompts: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ReportServeError(f"prompt #{index + 1} is malformed")
        prompt_id = str(item.get("id", "")).strip()
        if not prompt_id:
            raise ReportServeError(f"prompt #{index + 1}: an id is required")
        messages = [
            built
            for message in item.get("messages", [])
            if isinstance(message, dict) and (built := _form_message_to_mapping(message, prompt_id)) is not None
        ]
        if not messages:
            raise ReportServeError(f"prompt {prompt_id!r}: at least one non-empty message is required")
        mapping: dict[str, Any] = {
            "id": prompt_id,
            "category": str(item.get("category") or "general"),
            "isl_bucket": str(item.get("isl_bucket") or "short"),
            "messages": messages,
        }
        expected = str(item.get("expected_output") or "")
        if expected.strip():
            mapping["expected_output"] = expected
        tools = _form_yaml_field(str(item.get("tools") or ""), prompt_id, "tools", list)
        if tools is not None:
            mapping["tools"] = tools
        tool_results = _form_yaml_field(str(item.get("tool_results") or ""), prompt_id, "tool_results", dict)
        if tool_results is not None:
            mapping["tool_results"] = tool_results
        prompts.append(mapping)
    header = "# Prompt library edited in the llm-bench report 'Prompts' tab.\n"
    return header + yaml.dump(prompts, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=100)


def _prompts_content_from_form(form: dict[str, list[str]]) -> str:
    """Turn a save form into YAML text: ``payload`` (structured editor) or raw ``content``."""
    payload = form.get("payload", [""])[0]
    if payload:
        try:
            items = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ReportServeError(f"malformed editor payload: {exc}") from exc
        return build_prompts_yaml(items)
    return form.get("content", [""])[0]


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _num(value: Any, decimals: int = 1) -> str:
    """Render a number with fixed decimals, or ``-`` when missing."""
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------


def _point_tip(x_label: str, x_value: Any, name: str, value: float) -> str:
    """Build the popover text shown when a data point is clicked."""
    return f"{x_label} {x_value} · {name}: {_num(value, 1)}"


def _point(px: float, py: float, color: str, x_label: str, x_value: Any, name: str, value: float) -> str:
    """Render a clickable data point: a hover <title> plus a click popover."""
    tip = _point_tip(x_label, x_value, name, value)
    return (
        f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{color}' class='pt' "
        f"onclick='tip(this)' data-tip=\"{escape(tip)}\"><title>{escape(tip)}</title></circle>"
    )


def _svg_line_chart(title: str, x_label: str, x_values: list[float], series: list[dict[str, Any]]) -> str:
    """Render a small self-contained SVG line chart.

    ``series`` is a list of ``{name, color, values}`` where ``values`` aligns with
    ``x_values`` (``None`` entries are skipped). Returns an ``<svg>`` fragment.
    """
    width, height = 560, 300
    left, right, top, bottom = 56, 16, 40, 70
    plot_w, plot_h = width - left - right, height - top - bottom

    flat = [v for entry in series for v in entry["values"] if isinstance(v, (int, float))]
    y_max = max(flat) * 1.15 if flat else 1.0
    y_max = y_max or 1.0
    n = len(x_values)

    def x_pos(index: int) -> float:
        return left + (plot_w / 2 if n == 1 else plot_w * index / (n - 1))

    def y_pos(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}' class='chart'>",
        f"<text x='{left}' y='22' class='chart-title'>{escape(title)}</text>",
    ]
    for tick in range(5):
        gy = top + plot_h - plot_h * tick / 4
        parts.append(f"<line x1='{left}' y1='{gy:.1f}' x2='{width - right}' y2='{gy:.1f}' class='grid'/>")
        parts.append(f"<text x='{left - 8}' y='{gy + 4:.1f}' class='ax-y'>{y_max * tick / 4:.0f}</text>")
    for index, xval in enumerate(x_values):
        parts.append(f"<text x='{x_pos(index):.1f}' y='{height - bottom + 16}' class='ax-x'>{escape(str(xval))}</text>")
    parts.append(
        f"<text x='{left + plot_w / 2:.1f}' y='{height - bottom + 32}' class='ax-label'>{escape(x_label)}</text>"
    )

    for entry in series:
        color = entry["color"]
        pts = [(i, x_pos(i), y_pos(v), v) for i, v in enumerate(entry["values"]) if isinstance(v, (int, float))]
        if len(pts) >= _MIN_LINE_POINTS:
            poly = " ".join(f"{px:.1f},{py:.1f}" for _, px, py, _ in pts)
            parts.append(f"<polyline points='{poly}' fill='none' stroke='{color}' stroke-width='2'/>")
        for index, px, py, value in pts:
            parts.append(_point(px, py, color, x_label, x_values[index], entry["name"], value))

    legend_x = left
    for entry in series:
        parts.append(
            f"<rect x='{legend_x}' y='{height - bottom + 50}' width='10' height='10' fill='{entry['color']}'/>"
        )
        parts.append(
            f"<text x='{legend_x + 14}' y='{height - bottom + 59}' class='legend'>{escape(entry['name'])}</text>"
        )
        legend_x += 14 + 8 * len(entry["name"]) + 16
    parts.append("</svg>")
    return "".join(parts)


def _svg_bar_chart(title: str, x_label: str, x_values: list[Any], series: list[dict[str, Any]]) -> str:
    """Render a small grouped-bar SVG chart (one bar per series within each x slot)."""
    width, height = 560, 300
    left, right, top, bottom = 56, 16, 40, 70
    plot_w, plot_h = width - left - right, height - top - bottom

    flat = [v for entry in series for v in entry["values"] if isinstance(v, (int, float))]
    y_max = (max(flat) * 1.15 if flat else 1.0) or 1.0
    n_x = max(1, len(x_values))
    n_s = max(1, len(series))
    slot = plot_w / n_x
    bar_w = slot * 0.8 / n_s

    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}' class='chart'>",
        f"<text x='{left}' y='22' class='chart-title'>{escape(title)}</text>",
    ]
    for tick in range(5):
        gy = top + plot_h - plot_h * tick / 4
        parts.append(f"<line x1='{left}' y1='{gy:.1f}' x2='{width - right}' y2='{gy:.1f}' class='grid'/>")
        parts.append(f"<text x='{left - 8}' y='{gy + 4:.1f}' class='ax-y'>{y_max * tick / 4:.0f}</text>")
    for xi, xval in enumerate(x_values):
        slot_left = left + slot * xi + slot * 0.1
        cx = left + slot * xi + slot / 2
        parts.append(f"<text x='{cx:.1f}' y='{height - bottom + 16}' class='ax-x'>{escape(str(xval))}</text>")
        for si, entry in enumerate(series):
            value = entry["values"][xi] if xi < len(entry["values"]) else None
            if not isinstance(value, (int, float)):
                continue
            bar_h = (value / y_max) * plot_h
            bx = slot_left + si * bar_w
            by = top + plot_h - bar_h
            tip = _point_tip(x_label, xval, entry["name"], value)
            parts.append(
                f"<rect x='{bx:.1f}' y='{by:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' fill='{entry['color']}' "
                f"class='pt' onclick='tip(this)' data-tip=\"{escape(tip)}\"><title>{escape(tip)}</title></rect>"
            )
    parts.append(
        f"<text x='{left + plot_w / 2:.1f}' y='{height - bottom + 32}' class='ax-label'>{escape(x_label)}</text>"
    )
    legend_x = left
    for entry in series:
        parts.append(
            f"<rect x='{legend_x}' y='{height - bottom + 50}' width='10' height='10' fill='{entry['color']}'/>"
        )
        parts.append(
            f"<text x='{legend_x + 14}' y='{height - bottom + 59}' class='legend'>{escape(entry['name'])}</text>"
        )
        legend_x += 14 + 8 * len(entry["name"]) + 16
    parts.append("</svg>")
    return "".join(parts)


def _render_panel(records: list[dict[str, Any]], panel: Any) -> str:
    """Compute one dashboard panel and render its interactive chart (click a point)."""
    result = compute_panel(records, panel)
    colored = [{**s, "color": _SERIES_COLORS[i % len(_SERIES_COLORS)]} for i, s in enumerate(result.series)]
    if not result.x_values:
        chart = f"<p class='empty'>No data for x = {escape(result.x_label)} in this run.</p>"
    elif result.chart == "bar":
        chart = _svg_bar_chart(result.title, result.x_label, result.x_values, colored)
    else:
        chart = _svg_line_chart(result.title, result.x_label, result.x_values, colored)
    return f"<div class='panel'><h2>{escape(result.title)}</h2>{chart}</div>"


def render_dashboard(records: list[dict[str, Any]], dashboard_text: str) -> str:
    """Render every panel of a dashboard against a run's steady records."""
    panels = parse_dashboard(dashboard_text)
    return "".join(_render_panel(records, panel) for panel in panels)


# ---------------------------------------------------------------------------
# Run launcher (subprocess + progress)
# ---------------------------------------------------------------------------


def build_run_command(config_path: Path | None, req: RunRequest, out_dir: Path) -> list[str]:
    """Build the argv to launch a benchmark run as a subprocess (no shell)."""
    cmd = [sys.executable, "-m", "llm_bench", "run", "-m", req.model, "-o", str(out_dir), "--mode", req.mode]
    if config_path is not None:
        cmd += ["-c", str(config_path)]
    load = [item.strip() for item in req.load.split(",") if item.strip()]
    if req.mode == "open":
        for rate in load:
            cmd += ["--request-rate", rate]
    elif load:
        cmd += ["--concurrency", ",".join(load)]
    for flag, value in (
        ("--duration", req.duration),
        ("--max-tokens", req.max_tokens),
        ("--temperature", req.temperature),
        ("--slo-profile", req.slo_profile),
        ("--seed", req.seed),
        ("--prompts", req.prompts),
        ("--eval-method", req.eval_method),
        ("--judge-model", req.judge_model),
        ("--judge-rubric", req.judge_rubric),
        ("--embedding-model", req.embedding_model),
    ):
        if value:
            cmd += [flag, value]
    return cmd


def estimate_run_seconds(config_path: Path | None, req: RunRequest) -> float:
    """Estimate total wall-clock seconds for a run from the config + overrides."""
    run = _load_yaml(config_path).get("run", {}) if config_path is not None else {}
    run = run if isinstance(run, dict) else {}

    def _dur(value: Any, fallback: float) -> float:
        try:
            return parse_duration(value)
        except (ValueError, TypeError):
            return fallback

    per_level = (
        _dur(req.duration or run.get("duration"), 2.0) + _dur(run.get("warmup"), 0.5) + _dur(run.get("cooldown"), 0.5)
    )
    load = [item for item in req.load.split(",") if item.strip()]
    if load:
        n_levels = len(load)
    else:
        raw_levels = run.get("concurrency_levels")
        n_levels = len(raw_levels) if isinstance(raw_levels, list) and raw_levels else 1
    return max(1.0, max(1, n_levels) * per_level + 5.0)


class JobRegistry:
    """In-memory registry of launched runs, keyed by an opaque job id."""

    def __init__(self, config_path: Path | None) -> None:
        self._config_path = config_path
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, req: RunRequest) -> str:
        """Launch a run subprocess and return its job id."""
        out_dir = default_run_dir(req.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = (out_dir / "launch.log").open("w", encoding="utf-8")
        cmd = build_run_command(self._config_path, req, out_dir)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)  # nosec B603  (fixed argv, no shell)
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "proc": proc,
                "log": log,
                "out_dir": out_dir,
                "model": req.model,
                "started": time.monotonic(),
                "estimate": estimate_run_seconds(self._config_path, req),
            }
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        """Return the current state of a job for the polling UI."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"state": "unknown"}
        proc = job["proc"]
        elapsed = time.monotonic() - job["started"]
        estimate = job["estimate"]
        code = proc.poll()
        if code is None:
            # Two phases: the load sweep, then the asynchronous quality eval drain.
            in_eval = "eval_started" in _tail(job["out_dir"] / "launch.log", limit=4000)
            load_pct = 100.0 if in_eval else (min(99.0, 100.0 * elapsed / estimate) if estimate else 5.0)
            return {
                "state": "running",
                "phase": "eval" if in_eval else "load",
                "pct": round(load_pct, 1),
                "elapsed": round(elapsed),
                "estimate": round(estimate),
            }
        job["log"].close()
        if code == 0 and (job["out_dir"] / "summary.json").is_file():
            return {"state": "done", "pct": 100.0, "run": job["out_dir"].name}
        return {"state": "failed", "message": _tail(job["out_dir"] / "launch.log"), "code": code}

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return every launched job with its current status (newest first)."""
        with self._lock:
            order = [(jid, job["started"], job["model"]) for jid, job in self._jobs.items()]
        order.sort(key=lambda item: item[1], reverse=True)
        return [{"job": jid, "model": model, **self.status(jid)} for jid, _started, model in order]


def _tail(path: Path, limit: int = 400) -> str:
    """Return the last ``limit`` characters of a text file, for error display."""
    try:
        return path.read_text(encoding="utf-8")[-limit:].strip()
    except OSError:
        return "run failed (no log available)"


# ---------------------------------------------------------------------------
# Page rendering (Carbon shell)
# ---------------------------------------------------------------------------

_STYLE = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap');
* { box-sizing: border-box; }
body { font-family: 'IBM Plex Sans', -apple-system, system-ui, sans-serif; margin: 0; color: #161616; background: #f4f4f4; }
.hdr { background: #161616; color: #fff; height: 48px; display: flex; align-items: center; padding: 0 16px; gap: 12px; }
.hdr b { font-weight: 600; }
.hdr .muted { color: #c6c6c6; font-size: .85rem; }
.tabs { display: flex; background: #fff; border-bottom: 1px solid #e0e0e0; padding: 0 16px; }
.tabs a { padding: .8rem 1rem; text-decoration: none; color: #525252; border-bottom: 2px solid transparent; font-size: .9rem; }
.tabs a.active { color: #161616; border-bottom-color: #0f62fe; font-weight: 600; }
.main { padding: 1.5rem 16px; max-width: 1180px; }
h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 1rem; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 1.25rem 0 .5rem; }
select, input { font: inherit; padding: .45rem .6rem; background: #fff; border: none; border-bottom: 1px solid #8d8d8d; color: #161616; }
select { min-width: 22rem; }
label { display: block; font-size: .78rem; color: #525252; margin: .8rem 0 .25rem; }
.btn { font: inherit; font-weight: 600; background: #0f62fe; color: #fff; border: none; padding: .6rem 2.5rem .6rem 1rem; cursor: pointer; }
.btn:hover { background: #0353e9; }
.btn:disabled { background: #c6c6c6; cursor: default; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; background: #fff; font-variant-numeric: tabular-nums; }
th, td { padding: .5rem .7rem; text-align: right; border-bottom: 1px solid #e0e0e0; font-size: .85rem; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #fff; color: #161616; font-weight: 600; border-bottom: 1px solid #8d8d8d; }
tbody tr:hover { background: #e8e8e8; }
.meta { margin: .3rem 0 .8rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.chip { background: #e0e0e0; padding: .2rem .6rem; font-size: .8rem; }
.chip b { font-weight: 600; margin-right: .3rem; }
.charts { display: flex; flex-wrap: wrap; gap: 1rem; margin: .5rem 0 1rem; }
.chart { width: 560px; max-width: 100%; background: #fff; border: 1px solid #e0e0e0; }
.chart-title { font-size: 13px; font-weight: 600; fill: #161616; }
.grid { stroke: #e0e0e0; stroke-width: 1; }
.ax-y { font-size: 10px; fill: #6f6f6f; text-anchor: end; }
.ax-x { font-size: 10px; fill: #6f6f6f; text-anchor: middle; }
.ax-label, .legend { font-size: 11px; fill: #525252; text-anchor: middle; }
.legend { text-anchor: start; }
.empty { color: #6f6f6f; }
.pt { cursor: pointer; }
.pt:hover { stroke: #161616; stroke-width: 1.5; }
.helprow { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: flex-start; margin-top: .5rem; }
.helprow > details { flex: 1 1 22rem; min-width: 18rem; }
.help-box h3 { font-size: .85rem; margin: .8rem 0 .2rem; }
table.help { font-size: .8rem; }
table.help td { vertical-align: top; border-bottom: 1px solid #f0f0f0; }
table.help td:first-child { white-space: nowrap; color: #0f62fe; width: 1%; }
.note { color: #6f6f6f; font-size: .8rem; }
.bar { height: 8px; background: #e0e0e0; margin: .3rem 0 .5rem; max-width: 30rem; overflow: hidden; }
.bar > span { display: block; height: 100%; width: 0; background: #0f62fe; transition: width .4s; }
.bar > span.indet { width: 35%; animation: indet 1.1s ease-in-out infinite; }
@keyframes indet { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
.run-badge { margin-left: auto; color: #fff; background: #0f62fe; padding: .15rem .6rem; font-size: .8rem; text-decoration: none; }
.jobrow { border: 1px solid #e0e0e0; background: #fff; padding: .6rem .8rem; margin: .5rem 0; max-width: 34rem; }
.jobhead { font-size: .9rem; margin-bottom: .2rem; }
.barlabel { font-size: .72rem; color: #6f6f6f; }
.row { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: flex-end; }
.fld { margin: .9rem 0; }
.grid { display: flex; flex-wrap: wrap; gap: .35rem; margin: .3rem 0; max-width: 44rem; }
.cb { display: inline-flex; align-items: center; gap: .25rem; background: #fff; border: 1px solid #e0e0e0;
      padding: .25rem .5rem; font-size: .82rem; cursor: pointer; }
.radios { display: flex; gap: .8rem; }
textarea { width: 100%; max-width: 60rem; font-family: 'IBM Plex Mono', monospace; font-size: .82rem;
           padding: .5rem; border: 1px solid #8d8d8d; background: #fff; color: #161616; margin: .2rem 0; }
.card { border: 1px solid #e0e0e0; background: #fff; padding: .7rem .8rem; margin: .7rem 0; max-width: 62rem; }
.crow { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .4rem; }
.crow label, .adv { font-size: .75rem; color: #525252; }
.adv { display: block; margin-top: .4rem; }
.crow input.p-id { min-width: 14rem; }
.msgs { margin: .2rem 0; }
.msg { display: flex; gap: .4rem; align-items: flex-start; margin: .3rem 0; }
.msg .m-role { flex: 0 0 7rem; }
.msg .m-content { flex: 1; }
.mini { font: inherit; font-size: .8rem; background: #e0e0e0; border: none; padding: .3rem .6rem; cursor: pointer; }
.mini:hover { background: #c6c6c6; }
.rm-card { margin-left: auto; background: none; border: none; color: #da1e28; cursor: pointer;
           padding: .2rem .3rem; display: inline-flex; align-items: center; }
.rm-card:hover { color: #a2191f; }
details { margin: .4rem 0; }
summary { font-size: .8rem; color: #0f62fe; cursor: pointer; }
.panel { margin: 1.2rem 0; }
.vals .msg { gap: .4rem; }
.i { display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px;
     margin-left: 5px; border: none; border-radius: 50%; background: #0f62fe; color: #fff; font-size: 10px;
     font-style: italic; font-weight: 700; line-height: 1; cursor: pointer; vertical-align: middle; padding: 0; }
.i:hover { background: #0353e9; }
#tip { position: absolute; display: none; max-width: 290px; background: #393939; color: #fff;
       padding: .5rem .65rem; font-size: .8rem; line-height: 1.4; z-index: 50; box-shadow: 0 2px 8px rgba(0,0,0,.35); }
"""

_TIP_JS = """
function tip(btn){
  let t = document.getElementById('tip');
  if(!t){ t = document.createElement('div'); t.id = 'tip'; document.body.appendChild(t); }
  if(t.style.display === 'block' && t.dataset.tip === btn.dataset.tip){ t.style.display = 'none'; return; }
  t.textContent = btn.dataset.tip; t.dataset.tip = btn.dataset.tip;
  const r = btn.getBoundingClientRect();
  t.style.left = (window.scrollX + r.left) + 'px';
  t.style.top = (window.scrollY + r.bottom + 6) + 'px';
  t.style.display = 'block';
}
document.addEventListener('click', e => {
  const t = document.getElementById('tip');
  if(t && !e.target.closest('.i') && !e.target.closest('.pt') && e.target.id !== 'tip'){ t.style.display = 'none'; }
});
async function pollRunningBadge(){
  const b = document.getElementById('running-badge');
  if(!b) return;
  try {
    const r = await (await fetch('/run/jobs')).json();
    const running = (r.jobs||[]).filter(x => x.state === 'running');
    if(running.length){
      const ev = running.some(x => x.phase === 'eval');
      b.textContent = '▶ ' + running.length + ' run' + (running.length>1?'s':'') + (ev ? ' (scoring quality)' : ' in progress');
      b.style.display = 'inline';
      setTimeout(pollRunningBadge, 2000);
    } else { b.style.display = 'none'; setTimeout(pollRunningBadge, 5000); }
  } catch(e){ setTimeout(pollRunningBadge, 5000); }
}
pollRunningBadge();
"""


def _shell(active: str, body: str, runs_count: int) -> str:
    """Wrap ``body`` in the Carbon header + tab bar (with the (i) popover script)."""

    def tab(name: str, label: str, href: str) -> str:
        cls = " class='active'" if name == active else ""
        return f"<a href='{href}'{cls}>{label}</a>"

    tabs = tab("dashboards", "Dashboards", "/") + tab("run", "Run", "/run") + tab("prompts", "Prompts", "/prompts")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>llm-bench report</title><style>{_STYLE}</style></head><body>"
        f"<div class='hdr'><b>llm-bench</b><span class='muted'>report · {runs_count} run(s)</span>"
        "<a id='running-badge' class='run-badge' href='/run' style='display:none'></a></div>"
        f"<nav class='tabs'>{tabs}</nav><main class='main'>{body}</main>"
        f"<script>{_TIP_JS}</script></body></html>"
    )


def _label(text: str) -> str:
    """Render a form label with its ``(i)`` popover."""
    return f"<label>{escape(text)}{_info(text, _FIELD_INFO)}</label>"


def _checkbox_grid(name: str, presets: tuple[int, ...], default: frozenset[int]) -> str:
    """Render a grid of preset checkboxes (those in ``default`` start checked)."""
    boxes = "".join(
        f"<label class='cb'><input type='checkbox' name='{name}' value='{value}'"
        f"{' checked' if value in default else ''}>{value}</label>"
        for value in presets
    )
    return f"<div class='grid'>{boxes}</div>"


def _duration_select() -> str:
    """Render the single-choice duration picker with an 'Other' free-text option."""
    options = "".join(
        f"<option value='{value}'{' selected' if value == _DURATION_DEFAULT else ''}>{label}</option>"
        for value, label in _DURATION_CHOICES
    )
    return (
        "<select name='duration' onchange=\"document.getElementById('dur-other').style.display="
        "this.value==='other'?'inline-block':'none'\">"
        f"{options}<option value='other'>Other…</option></select>"
        "<input id='dur-other' name='duration_other' placeholder='e.g. 45s, 4m' size='8' style='display:none'>"
    )


def slo_profile_options(config_path: Path | None) -> tuple[list[str], str]:
    """Return the selectable SLO profile names (config + built-ins) and the default."""
    raw = _load_yaml(config_path) if config_path is not None else {}
    configured = raw.get("slo_profiles", {})
    names = list(configured) if isinstance(configured, dict) else []
    for builtin in ("interactive", "relaxed"):
        if builtin not in names:
            names.append(builtin)
    run = raw.get("run", {})
    default = str(run.get("slo_profile", "")) if isinstance(run, dict) else ""
    return names, default


def build_run_page(config_path: Path | None, prompts_dir: Path | None, runs_count: int) -> str:
    """Render the Run tab: model picker, mode + load + tuning fields, progress area."""
    models = list_config_models(config_path)
    if not models:
        inner = (
            "<p class='empty'>No models found. Create "
            "<code>~/.config/llm-bench/config.yaml</code> (run <code>llm-bench init</code>) "
            "and register models under <code>models:</code>.</p>"
        )
        return _shell("run", f"<h1>Run a benchmark</h1>{inner}", runs_count)

    model_options = "".join(
        f"<option value='{escape(m['name'])}'>{escape(m['name'])} · {escape(m['model'])}</option>" for m in models
    )
    eval_model_options = "<option value=''>— from config —</option>" + model_options
    embed_model_options = (
        "<option value=''>— from config —</option>"
        "<option value='local:cpu'>local · CPU (bge-small)</option>"
        "<option value='local:gpu'>local · GPU (bge-large)</option>" + model_options
    )
    profiles, profile_default = slo_profile_options(config_path)
    profile_options = "".join(
        f"<option value='{escape(p)}'{' selected' if p == profile_default else ''}>{escape(p)}</option>"
        for p in profiles
    )
    prompt_options = "<option value=''>built-in default</option>" + "".join(
        f"<option value='{escape(Path(f).stem)}'>{escape(Path(f).stem)}</option>"
        for f in list_prompt_files(prompts_dir)
    )
    body = (
        "<h1>Run a benchmark</h1>"  # nosec B608  (HTML string building, not SQL; '<select>' trips the heuristic)
        "<form id='runform' onsubmit='startRun(event)'>"
        f"<div class='fld'>{_label('Model')}<select name='model'>{model_options}</select></div>"
        f"<div class='fld'>{_label('Mode')}"
        "<div class='radios'>"
        "<label class='cb'><input type='radio' name='mode' value='closed' checked onchange='toggleMode()'>closed</label>"
        "<label class='cb'><input type='radio' name='mode' value='open' onchange='toggleMode()'>open</label>"
        "</div></div>"
        f"<div class='fld' id='closed-load'>{_label('Concurrency')}"
        f"{_checkbox_grid('c', _CONCURRENCY_PRESETS, _CONCURRENCY_DEFAULT)}"
        "<input name='c_manual' placeholder='extra, comma-separated e.g. 12,24' size='28'></div>"
        f"<div class='fld' id='open-load' style='display:none'>{_label('Arrival rate')}"
        f"{_checkbox_grid('r', _RATE_PRESETS, _RATE_DEFAULT)}"
        "<input name='r_manual' placeholder='extra req/s, comma-separated e.g. 7,15' size='28'></div>"
        f"<div class='fld'>{_label('Duration')}{_duration_select()}</div>"
        f"<div class='fld'>{_label('Prompts')}<select name='prompts'>{prompt_options}</select></div>"
        f"<div class='fld'>{_label('Quality eval')}<select name='eval_method' onchange='toggleEval()'>"
        "<option value=''>none</option><option value='embedding'>embedding (cosine)</option>"
        "<option value='judge'>judge (model)</option></select></div>"
        "<div class='row' id='eval-judge' style='display:none'>"
        f"<div class='fld'>{_label('Judge model')}<select name='judge_model'>{eval_model_options}</select></div>"
        f"<div class='fld'>{_label('Judge rubric')}<select name='judge_rubric'>"
        "<option value=''>— from config —</option><option value='score'>score (0..1)</option>"
        "<option value='three_level'>three_level</option><option value='binary'>binary</option></select></div>"
        "</div>"
        "<div class='row' id='eval-embedding' style='display:none'>"
        f"<div class='fld'>{_label('Embedding model')}<select name='embedding_model'>{embed_model_options}</select></div>"
        "</div>"
        "<div class='row'>"
        f"<div class='fld'>{_label('Max tokens')}<input name='max_tokens' type='number' min='1' placeholder='config' size='8'></div>"
        f"<div class='fld'>{_label('Temperature')}<input name='temperature' type='number' min='0' step='0.1' placeholder='config' size='8'></div>"
        f"<div class='fld'>{_label('SLO profile')}<select name='slo_profile'>{profile_options}</select></div>"
        f"<div class='fld'>{_label('Seed')}<input name='seed' type='number' placeholder='config' size='8'></div>"
        "</div>"
        "<button class='btn' id='startbtn' type='submit'>Start run</button>"
        "<span class='note' id='startmsg' style='margin-left:1rem'></span>"
        "</form>"
        "<h2>Runs this session</h2>"
        "<div id='jobs'><p class='empty'>No runs launched yet.</p></div>"
        f"<script>{_RUN_JS}</script>"
    )
    return _shell("run", body, runs_count)


_RUN_JS = """
function toggleMode(){
  const open = document.querySelector('input[name=mode]:checked').value === 'open';
  document.getElementById('open-load').style.display = open ? 'block' : 'none';
  document.getElementById('closed-load').style.display = open ? 'none' : 'block';
}
function toggleEval(){
  const m = document.querySelector('select[name=eval_method]').value;
  document.getElementById('eval-judge').style.display = m === 'judge' ? 'flex' : 'none';
  document.getElementById('eval-embedding').style.display = m === 'embedding' ? 'flex' : 'none';
}
async function startRun(ev){
  ev.preventDefault();
  const msg = document.getElementById('startmsg'); msg.textContent = 'Launching…';
  const data = new URLSearchParams(new FormData(document.getElementById('runform')));
  try {
    const j = await (await fetch('/run/start', {method:'POST', body:data})).json();
    if (j.error) { msg.textContent = j.error; return; }
    msg.textContent = '';
    refreshJobs();
  } catch (e) { msg.textContent = String(e); }
}
function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function jobRow(j){
  let html = "<div class='jobrow'><div class='jobhead'><b>" + esc(j.model||'run') + "</b> ";
  if(j.state === 'running'){
    const ev = j.phase === 'eval';
    html += ev ? "scoring quality…" : ("running · " + Math.round(j.pct||0) + "% · " + (j.elapsed||0) + "s/~" + (j.estimate||0) + "s");
    html += "</div>";
    html += "<div class='barlabel'>Load</div><div class='bar'><span style='width:" + (j.pct||0) + "%'></span></div>";
    if(ev){ html += "<div class='barlabel'>Quality eval</div><div class='bar'><span class='indet'></span></div>"; }
  } else if(j.state === 'done'){
    html += "done · <a href='/?run=" + encodeURIComponent(j.run) + "'>open in Dashboards</a></div>";
  } else if(j.state === 'failed'){
    html += "<span style='color:#da1e28'>failed: " + esc(j.message || ('exit ' + j.code)) + "</span></div>";
  } else { html += esc(j.state||'') + "</div>"; }
  return html + "</div>";
}
async function refreshJobs(){
  try {
    const r = await (await fetch('/run/jobs')).json();
    const jobs = r.jobs || [];
    const box = document.getElementById('jobs');
    box.innerHTML = jobs.length ? jobs.map(jobRow).join('') : "<p class='empty'>No runs launched yet.</p>";
    if(jobs.some(x => x.state === 'running')) setTimeout(refreshJobs, 1500);
  } catch (e) {}
}
refreshJobs();
"""


def build_prompts_page(prompts_dir: Path | None, runs_count: int) -> str:
    """Render the Prompts tab: a structured per-prompt editor (no raw YAML)."""
    if prompts_dir is None:
        inner = "<p class='empty'>No prompts directory configured.</p>"
        return _shell("prompts", f"<h1>Prompts</h1>{inner}", runs_count)

    stems = [Path(f).stem for f in list_prompt_files(prompts_dir)]
    options = "".join(f"<option value='{escape(s)}'>{escape(s)}</option>" for s in stems)
    hint = escape(str(prompts_dir))
    body = (
        "<h1>Prompts</h1>"
        f"<p class='note'>Profiles in <code>{hint}</code>. Pick one to edit it as a form (one card per prompt, "
        "add/remove prompts and messages), or type a new name. Save rebuilds and validates the YAML.</p>"
        "<div class='row'>"
        "<div class='fld'><label>Choose a Prompt Profile</label><select id='pfile' onchange='loadPrompt()'>"
        f"<option value=''>— choose a profile —</option>{options}</select></div>"
        "<div class='fld'><label>New profile name</label>"
        "<input id='pnew' placeholder='e.g. tools' size='18'></div>"
        "</div>"
        "<div id='cards'></div>"
        "<button class='mini' type='button' onclick='addPrompt()'>+ Add prompt</button>"
        "<div style='margin-top:1rem'><button class='btn' type='button' onclick='savePrompt()'>Save</button>"
        "<span class='note' id='pmsg' style='margin-left:1rem'></span></div>"
        f"<script>{_PROMPTS_JS}</script>"
    )
    return _shell("prompts", body, runs_count)


_PROMPTS_JS = """
const CATS=['coding','synthesis','tool-use','vision','general'], ISL=['short','medium','long'],
      ROLES=['system','user','assistant'];
function opts(vals,sel){ return vals.map(v=>"<option"+(v===sel?" selected":"")+">"+v+"</option>").join(''); }
function msgRow(m){
  m=m||{};
  const d=document.createElement('div'); d.className='msg';
  d.innerHTML="<select class='m-role'>"+opts(ROLES,m.role||'user')+"</select>"
    +"<textarea class='m-content' rows='2'></textarea>"
    +"<button type='button' class='mini rm-msg' title='remove message'>x</button>";
  const ta=d.querySelector('.m-content'); ta.value=m.content||'';
  if(m.json){ ta.dataset.json='1'; ta.placeholder='(structured JSON content - edit with care)'; }
  return d;
}
function card(p){
  p=p||{};
  const d=document.createElement('div'); d.className='card';
  d.innerHTML="<div class='crow'><label>id</label><input class='p-id'>"
    +"<label>category</label><select class='p-cat'>"+opts(CATS,p.category||'general')+"</select>"
    +"<label>length</label><select class='p-isl'>"+opts(ISL,p.isl_bucket||'short')+"</select>"
    +"<button type='button' class='rm-card' title='Remove prompt' aria-label='Remove prompt'>"
    +"<svg viewBox='0 0 32 32' width='18' height='18' fill='currentColor' aria-hidden='true'>"
    +"<path d='M12 12h2v12h-2zM18 12h2v12h-2z'/>"
    +"<path d='M4 6v2h2v20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8h2V6zm4 22V8h16v20zM12 2h8v2h-8z'/></svg></button></div>"
    +"<div class='msgs'></div>"
    +"<button type='button' class='mini add-msg'>+ message</button>"
    +"<label class='adv'>expected_output (optional)</label><textarea class='p-exp' rows='2'></textarea>"
    +"<details><summary>advanced: tools / tool_results (raw YAML)</summary>"
    +"<label class='adv'>tools</label><textarea class='p-tools' rows='3'></textarea>"
    +"<label class='adv'>tool_results</label><textarea class='p-tr' rows='2'></textarea></details>";
  d.querySelector('.p-id').value=p.id||'';
  d.querySelector('.p-exp').value=p.expected_output||'';
  d.querySelector('.p-tools').value=p.tools||'';
  d.querySelector('.p-tr').value=p.tool_results||'';
  const msgs=d.querySelector('.msgs');
  (p.messages&&p.messages.length?p.messages:[{role:'user',content:''}]).forEach(m=>msgs.appendChild(msgRow(m)));
  return d;
}
function render(list){ const c=document.getElementById('cards'); c.innerHTML=''; (list||[]).forEach(p=>c.appendChild(card(p))); }
function addPrompt(){ document.getElementById('cards').appendChild(card({})); }
document.addEventListener('click', e=>{
  const rmCard=e.target.closest('.rm-card');
  if(rmCard){ if(confirm('Remove this prompt? This cannot be undone until you reload without saving.')){ rmCard.closest('.card').remove(); } return; }
  if(e.target.closest('.add-msg')){ e.target.closest('.card').querySelector('.msgs').appendChild(msgRow({})); return; }
  const rmMsg=e.target.closest('.rm-msg');
  if(rmMsg){ rmMsg.closest('.msg').remove(); }
});
async function loadPrompt(){
  const name=document.getElementById('pfile').value, msg=document.getElementById('pmsg'); msg.textContent='';
  if(!name){ render([]); return; }
  document.getElementById('pnew').value='';
  try {
    const j=await (await fetch('/prompts/load?file='+encodeURIComponent(name))).json();
    if(j.error){ msg.textContent=j.error; return; }
    render(j.prompts);
  } catch(e){ msg.textContent=String(e); }
}
function collect(){
  const out=[];
  document.querySelectorAll('#cards .card').forEach(card=>{
    const msgs=[];
    card.querySelectorAll('.msg').forEach(row=>{
      const ta=row.querySelector('.m-content');
      msgs.push({role: row.querySelector('.m-role').value, content: ta.value, json: ta.dataset.json==='1'});
    });
    out.push({
      id: card.querySelector('.p-id').value.trim(),
      category: card.querySelector('.p-cat').value,
      isl_bucket: card.querySelector('.p-isl').value,
      messages: msgs,
      expected_output: card.querySelector('.p-exp').value,
      tools: card.querySelector('.p-tools').value,
      tool_results: card.querySelector('.p-tr').value
    });
  });
  return out;
}
async function savePrompt(){
  const msg=document.getElementById('pmsg');
  const name=document.getElementById('pnew').value.trim()||document.getElementById('pfile').value;
  if(!name){ msg.textContent='Choose a file or enter a new name.'; return; }
  const data=new URLSearchParams({file:name, payload:JSON.stringify(collect())});
  try {
    const j=await (await fetch('/prompts/save',{method:'POST',body:data})).json();
    if(j.error){ msg.textContent='Not saved: '+j.error; return; }
    msg.textContent='Saved '+j.file;
    setTimeout(()=>location.search='', 700);
  } catch(e){ msg.textContent=String(e); }
}
"""


def _options(values: tuple[str, ...] | list[str], selected: str, *, blank: str | None = None) -> str:
    """Render <option> tags, marking ``selected`` (with an optional blank first option)."""
    opts = [f"<option value=''>{escape(blank)}</option>"] if blank is not None else []
    opts += [f"<option{' selected' if v == selected else ''}>{escape(v)}</option>" for v in values]
    return "".join(opts)


def build_dashboards_page(
    dashboards_dir: Path | None, runs_dir: Path, file: str | None, run: str | None, runs_count: int
) -> str:
    """Render the Dashboards tab: file + run pickers, computed panels, and an editor."""
    if dashboards_dir is None:
        return _shell("dashboards", "<h1>Dashboards</h1><p class='empty'>No dashboards directory.</p>", runs_count)

    stems = [Path(f).stem for f in list_prompt_files(dashboards_dir)]
    runs = [r.name for r in iter_runs(runs_dir)]
    # Sensible defaults so the home page just shows something: the 'default'
    # dashboard (or the first) over the newest run.
    if not file:
        file = "default" if "default" in stems else (stems[0] if stems else None)
    if not run:
        run = runs[0] if runs else None

    file_sel = "".join(
        f"<option value='{escape(s)}'{' selected' if s == file else ''}>{escape(s)}</option>" for s in stems
    )
    run_sel = "".join(
        f"<option value='{escape(r)}'{' selected' if r == run else ''}>{escape(r)}</option>" for r in runs
    )
    pickers = (
        "<div class='row'>"
        f"<div class='fld'><label>Dashboard</label><select id='dfile' onchange='nav()'>"
        f"<option value=''>— choose —</option>{file_sel}</select></div>"
        f"<div class='fld'><label>Run</label><select id='drun' onchange='nav()'>"
        f"<option value=''>— choose —</option>{run_sel}</select></div>"
        "<div class='fld'><label>New name</label><input id='dnew' placeholder='e.g. tokens' size='14'></div>"
        "</div>"
    )

    panels_html = "<p class='empty'>Pick a dashboard and a run to render it.</p>"
    editor_form = ""
    if file and file in stems:
        try:
            text = read_prompt_file(dashboards_dir, file)
        except ReportServeError as exc:
            return _shell(
                "dashboards", f"<h1>Dashboards</h1>{pickers}<p class='empty'>{escape(str(exc))}</p>", runs_count
            )
        if run and run in runs:
            try:
                records = read_steady_records(runs_dir / run)
                panels_html = render_dashboard(records, text) or "<p class='empty'>This dashboard has no panels.</p>"
            except DashboardError as exc:
                panels_html = f"<p class='empty'>{escape(str(exc))}</p>"
        else:
            panels_html = "<p class='empty'>No run yet. Launch one in the Run tab.</p>"
        editor_form = (
            "<details><summary>Edit this dashboard</summary>"
            "<div id='panels'></div>"
            "<button class='mini' type='button' onclick='addPanel()'>+ Add panel</button>"
            "<div style='margin-top:.6rem'><button class='btn' type='button' onclick='saveDash()'>Save</button>"
            "<span class='note' id='dmsg' style='margin-left:1rem'></span></div></details>"
        )

    controls = f"<div class='helprow'>{editor_form}{_dashboard_help_html()}</div>"
    body = f"<h1>Dashboards</h1>{pickers}{panels_html}{controls}<script>{_DASH_JS}</script>"
    return _shell("dashboards", body, runs_count)


def _help_table(caption: str, items: dict[str, str]) -> str:
    """Render a two-column reference table (name -> meaning)."""
    rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(text)}</td></tr>" for name, text in items.items()
    )
    return f"<h3>{escape(caption)}</h3><table class='help'><tbody>{rows}</tbody></table>"


def _dashboard_help_html() -> str:
    """Render the 'metrics & dimensions' reference shown beside the editor."""
    dims = _help_table("Dimensions (x / group)", {name: DIMENSION_HELP.get(name, "") for name in DIMENSIONS})
    metrics = _help_table("Metrics (values)", {name: METRIC_HELP.get(name, "") for name in METRICS})
    aggs = _help_table("Aggregations", dict(AGG_HELP))
    note = (
        "<p class='note'>A panel = one chart. <b>x</b> is the axis dimension, <b>group</b> optionally splits each "
        "value into one series per value, and <b>values</b> are the metrics plotted. Charts use the steady, "
        "successful requests of the chosen run. Chart type is auto (line for a numeric x, bars for a categorical "
        "one). Click any point or bar to read its exact value.</p>"
    )
    return (
        "<details class='help-box'><summary>What do the metrics and dimensions mean?</summary>"
        f"{note}{dims}{metrics}{aggs}</details>"
    )


_DASH_JS = (
    "const DIMS="
    + json.dumps(["", *DIMENSIONS])
    + ", METRICS="
    + json.dumps(list(METRICS))
    + ", AGGS="
    + json.dumps(list(AGGS))
    + ", CHARTS=['auto','line','bar'];\n"
    + """
function opt(vals,sel){ return vals.map(v=>"<option"+(v===sel?" selected":"")+">"+v+"</option>").join(''); }
function nav(){
  const f=document.getElementById('dfile').value, r=document.getElementById('drun').value;
  location.search='?file='+encodeURIComponent(f)+'&run='+encodeURIComponent(r);
}
function valRow(v){
  v=v||{};
  const d=document.createElement('div'); d.className='msg';
  d.innerHTML="<select class='v-metric'>"+opt(METRICS,v.metric||'e2e')+"</select>"
    +"<select class='v-agg'>"+opt(AGGS,v.agg||'p50')+"</select>"
    +"<button type='button' class='mini rm-val'>x</button>";
  return d;
}
function panelCard(p){
  p=p||{};
  const d=document.createElement('div'); d.className='card';
  d.innerHTML="<div class='crow'><label>title</label><input class='p-title'>"
    +"<label>x</label><select class='p-x'>"+opt(DIMS,p.x||'level_or_rate')+"</select>"
    +"<label>group</label><select class='p-group'>"+opt(DIMS,p.group||'')+"</select>"
    +"<label>chart</label><select class='p-chart'>"+opt(CHARTS,p.chart||'auto')+"</select>"
    +"<button type='button' class='rm-card' title='Remove panel' aria-label='Remove panel'>"
    +"<svg viewBox='0 0 32 32' width='18' height='18' fill='currentColor' aria-hidden='true'>"
    +"<path d='M12 12h2v12h-2zM18 12h2v12h-2z'/>"
    +"<path d='M4 6v2h2v20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8h2V6zm4 22V8h16v20zM12 2h8v2h-8z'/></svg></button></div>"
    +"<div class='vals'></div><button type='button' class='mini add-val'>+ value</button>";
  d.querySelector('.p-title').value=p.title||'';
  const vals=d.querySelector('.vals');
  (p.values&&p.values.length?p.values:[{metric:'e2e',agg:'p50'}]).forEach(v=>vals.appendChild(valRow(v)));
  return d;
}
function renderPanels(list){ const c=document.getElementById('panels'); if(!c)return; c.innerHTML=''; (list||[]).forEach(p=>c.appendChild(panelCard(p))); }
function addPanel(){ const c=document.getElementById('panels'); if(c) c.appendChild(panelCard({})); }
document.addEventListener('click', e=>{
  const rc=e.target.closest('.rm-card');
  if(rc && rc.closest('#panels')){ if(confirm('Remove this panel?')) rc.closest('.card').remove(); return; }
  if(e.target.closest('.add-val')){ e.target.closest('.card').querySelector('.vals').appendChild(valRow({})); return; }
  const rv=e.target.closest('.rm-val');
  if(rv){ rv.closest('.msg').remove(); }
});
function collectDash(){
  const out=[];
  document.querySelectorAll('#panels .card').forEach(card=>{
    const values=[];
    card.querySelectorAll('.vals .msg').forEach(row=>{
      values.push({metric:row.querySelector('.v-metric').value, agg:row.querySelector('.v-agg').value});
    });
    out.push({title:card.querySelector('.p-title').value.trim(), x:card.querySelector('.p-x').value,
              group:card.querySelector('.p-group').value, chart:card.querySelector('.p-chart').value, values:values});
  });
  return out;
}
async function saveDash(){
  const msg=document.getElementById('dmsg');
  const name=document.getElementById('dnew').value.trim()||document.getElementById('dfile').value;
  if(!name){ msg.textContent='Choose a file or enter a new name.'; return; }
  const data=new URLSearchParams({file:name, payload:JSON.stringify(collectDash())});
  try {
    const j=await (await fetch('/dashboards/save',{method:'POST',body:data})).json();
    if(j.error){ msg.textContent='Not saved: '+j.error; return; }
    msg.textContent='Saved '+j.file;
    const f=document.getElementById('dfile').value, r=document.getElementById('drun').value;
    setTimeout(()=>location.search='?file='+encodeURIComponent(j.file)+'&run='+encodeURIComponent(r), 700);
  } catch(e){ msg.textContent=String(e); }
}
async function loadDash(file){
  if(!file) return;
  try {
    const j=await (await fetch('/dashboards/load?file='+encodeURIComponent(file))).json();
    if(!j.error) renderPanels(j.panels);
  } catch(e){}
}
loadDash(document.getElementById('dfile') ? document.getElementById('dfile').value : '');
"""
)


def _merge_load(checked: list[str], manual: str, *, integers: bool) -> str:
    """Merge preset + manual load values into a sorted, de-duplicated comma list.

    Each token must be a positive number (a positive integer when ``integers``).
    Raises :class:`ReportServeError` on an invalid token.
    """
    numbers: list[float] = []
    for token in [*checked, *manual.split(",")]:
        item = token.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            raise ReportServeError(f"invalid value: {item!r}") from None
        if value <= 0:
            raise ReportServeError(f"value must be > 0: {item!r}")
        if integers and not value.is_integer():
            raise ReportServeError(f"concurrency must be a whole number: {item!r}")
        numbers.append(value)
    ordered = sorted(set(numbers))
    return ",".join(str(int(n)) if n.is_integer() else str(n) for n in ordered)


def parse_run_form(form: dict[str, list[str]], known_models: set[str], prompts_dir: Path | None = None) -> RunRequest:
    """Validate the Run-tab form into a :class:`RunRequest` (raises on bad input)."""
    model = form.get("model", [""])[0]
    if model not in known_models:
        raise ReportServeError(f"unknown model: {model!r}")
    mode = form.get("mode", ["closed"])[0]
    if mode not in {"closed", "open"}:
        raise ReportServeError(f"invalid mode: {mode!r}")

    if mode == "open":
        load = _merge_load(form.get("r", []), form.get("r_manual", [""])[0], integers=False)
        if not load:
            raise ReportServeError("select at least one arrival rate")
    else:
        load = _merge_load(form.get("c", []), form.get("c_manual", [""])[0], integers=True)
        if not load:
            raise ReportServeError("select at least one concurrency level")

    duration_choice = form.get("duration", [""])[0]
    duration = form.get("duration_other", [""])[0].strip() if duration_choice == "other" else duration_choice
    if duration:
        try:
            parse_duration(duration)
        except (ValueError, TypeError):
            raise ReportServeError(f"invalid duration: {duration!r}") from None

    prompts_name = form.get("prompts", [""])[0].strip()
    prompts_path = ""
    if prompts_name:
        if prompts_dir is None:
            raise ReportServeError("no prompts directory configured")
        target = safe_prompt_path(prompts_dir, prompts_name)
        if not target.is_file():
            raise ReportServeError(f"prompts file not found: {prompts_name}")
        prompts_path = str(target)

    eval_method = form.get("eval_method", [""])[0].strip()
    if eval_method and eval_method not in {"embedding", "judge"}:
        raise ReportServeError(f"invalid eval method: {eval_method!r}")
    # eval model overrides only apply to the matching method.
    judge_model = form.get("judge_model", [""])[0].strip() if eval_method == "judge" else ""
    judge_rubric = form.get("judge_rubric", [""])[0].strip() if eval_method == "judge" else ""
    embedding_model = form.get("embedding_model", [""])[0].strip() if eval_method == "embedding" else ""

    return RunRequest(
        model=model,
        mode=mode,
        load=load,
        duration=duration,
        max_tokens=form.get("max_tokens", [""])[0].strip(),
        temperature=form.get("temperature", [""])[0].strip(),
        slo_profile=form.get("slo_profile", [""])[0].strip(),
        seed=form.get("seed", [""])[0].strip(),
        prompts=prompts_path,
        eval_method=eval_method,
        judge_model=judge_model,
        judge_rubric=judge_rubric,
        embedding_model=embedding_model,
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


def _make_handler(
    runs_dir: Path,
    config_path: Path | None,
    jobs: JobRegistry,
    prompts_dir: Path | None,
    dashboards_dir: Path | None,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to the runs dir, config, jobs, and content dirs."""

    known_models = {entry["name"] for entry in list_config_models(config_path)}

    class _Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            params = parse_qs(parts.query)
            runs_count = len(iter_runs(runs_dir))
            if parts.path in {"/", "/dashboards"}:
                page = build_dashboards_page(
                    dashboards_dir, runs_dir, params.get("file", [None])[0], params.get("run", [None])[0], runs_count
                )
                self._send(page.encode("utf-8"))
            elif parts.path == "/run":
                self._send(build_run_page(config_path, prompts_dir, runs_count).encode("utf-8"))
            elif parts.path == "/run/status":
                self._json(jobs.status(params.get("job", [""])[0]))
            elif parts.path == "/run/jobs":
                self._json({"jobs": jobs.list_jobs()})
            elif parts.path == "/prompts":
                self._send(build_prompts_page(prompts_dir, runs_count).encode("utf-8"))
            elif parts.path == "/prompts/load":
                self._load_prompt(params.get("file", [""])[0])
            elif parts.path == "/dashboards/load":
                self._load_dashboard(params.get("file", [""])[0])
            else:
                self.send_error(404)

        def _load_prompt(self, name: str) -> None:
            if prompts_dir is None:
                self._json({"error": "no prompts directory"}, status=400)
                return
            try:
                content = read_prompt_file(prompts_dir, name)
                self._json({"content": content, "prompts": prompts_to_form(content)})
            except ReportServeError as exc:
                self._json({"error": str(exc)}, status=400)

        def _load_dashboard(self, name: str) -> None:
            if dashboards_dir is None:
                self._json({"error": "no dashboards directory"}, status=400)
                return
            try:
                content = read_prompt_file(dashboards_dir, name)
                self._json({"panels": dashboard_to_form(content)})
            except (ReportServeError, DashboardError) as exc:
                self._json({"error": str(exc)}, status=400)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode("utf-8"))
            if path == "/run/start":
                self._start_run(form)
            elif path == "/prompts/save":
                self._save_prompt(form)
            elif path == "/dashboards/save":
                self._save_dashboard(form)
            else:
                self.send_error(404)

        def _start_run(self, form: dict[str, list[str]]) -> None:
            try:
                request = parse_run_form(form, known_models, prompts_dir)
            except ReportServeError as exc:
                self._json({"error": str(exc)}, status=400)
                return
            try:
                job_id = jobs.start(request)
            except OSError as exc:
                self._json({"error": f"could not launch run: {exc}"}, status=500)
                return
            self._json({"job": job_id})

        def _save_prompt(self, form: dict[str, list[str]]) -> None:
            if prompts_dir is None:
                self._json({"error": "no prompts directory"}, status=400)
                return
            try:
                content = _prompts_content_from_form(form)
                saved = save_prompt_file(prompts_dir, form.get("file", [""])[0], content)
            except ReportServeError as exc:
                self._json({"error": str(exc)}, status=400)
                return
            self._json({"file": saved})

        def _save_dashboard(self, form: dict[str, list[str]]) -> None:
            if dashboards_dir is None:
                self._json({"error": "no dashboards directory"}, status=400)
                return
            try:
                content = _dashboard_content_from_form(form)
                saved = save_dashboard_file(dashboards_dir, form.get("file", [""])[0], content)
            except (ReportServeError, DashboardError) as exc:
                self._json({"error": str(exc)}, status=400)
                return
            self._json({"file": saved})

        def log_message(self, *_args: Any) -> None:
            """Silence the default per-request stderr logging."""

    return _Handler


def _dashboard_content_from_form(form: dict[str, list[str]]) -> str:
    """Turn a dashboard save form into YAML: ``payload`` (editor) or raw ``content``."""
    payload = form.get("payload", [""])[0]
    if payload:
        try:
            items = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ReportServeError(f"malformed dashboard payload: {exc}") from exc
        return build_dashboard_yaml(items)
    return form.get("content", [""])[0]


def make_server(
    runs_dir: Path,
    config_path: Path | None = None,
    jobs: JobRegistry | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    prompts_dir: Path | None = None,
    dashboards_dir: Path | None = None,
) -> ThreadingHTTPServer:
    """Create (but do not start) a report server. ``port=0`` picks a free port."""
    registry = jobs if jobs is not None else JobRegistry(config_path)
    return ThreadingHTTPServer(
        (host, port), _make_handler(runs_dir, config_path, registry, prompts_dir, dashboards_dir)
    )


def serve_reports(
    run: str | None,
    *,
    runs_dir: Path,
    config_path: Path | None = None,
    prompts_dir: Path | None = None,
    dashboards_dir: Path | None = None,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    host: str = DEFAULT_HOST,
) -> None:
    """Resolve ``run``, start the report server, optionally open the browser, and block.

    Blocks until interrupted (Ctrl-C).
    """
    selected: str | None = None
    listing_dir = runs_dir
    if run is not None:
        resolved = resolve_run(run, runs_dir)
        listing_dir = resolved.parent
        selected = resolved.name

    server = make_server(listing_dir, config_path, None, host, port, prompts_dir, dashboards_dir)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    if selected is not None:
        url += f"?run={selected}"
    print(f"serving reports from {listing_dir} at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.shutdown()
        server.server_close()
