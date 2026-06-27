"""llm-bench command-line entry point.

Defines the Typer ``app`` exposing the ``run``, ``serve``, ``models``, and
``init`` commands (plus a hidden ``analyze``). ``serve`` launches the local web
UI (Dashboards / Run / Prompts).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys

# Typer evaluates these parameter annotations at runtime to build the CLI, so Path
# must stay a runtime import (it cannot move into a TYPE_CHECKING block).
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated

import typer

from llm_bench.analyze import AnalyzeError, run_query
from llm_bench.config import (
    REDACTED,
    ConfigError,
    ConfigValidationError,
    EmbeddingConfig,
    EvaluationConfig,
    JudgeConfig,
    JudgeModel,
    MissingEnvVarError,
    apply_eval_method,
    apply_slo_overrides,
    default_config_file,
    default_dashboards_dir,
    default_prompts_dir,
    default_prompts_path,
    default_run_dir,
    default_runs_dir,
    load_config,
    scaffold_config,
)
from llm_bench.prompts import EmptyPromptSetError, PromptError, PromptLibrary, builtin_library, load_prompts
from llm_bench.runner import (
    PreflightAbort,
    RunAbort,
    ValidationAbort,
    parse_duration,
    preflight_check,
    run_benchmark,
    write_resolved_config,
)
from llm_bench.serve import DEFAULT_PORT, ReportServeError, serve_reports
from llm_bench.version import __version__

if TYPE_CHECKING:
    from llm_bench.config import BenchConfig, ModelRegistryEntry

_APP_HELP = """\
Benchmark the performance and output quality of OpenAI-compatible LLM endpoints.

WORKFLOW

- llm-bench init — scaffold ~/.config/llm-bench (config.yaml + prompts/ + dashboards/).
- Edit config.yaml — register your model(s) under `models:`; put secrets in env vars via $ENV:.
- llm-bench models — list registered models (no endpoint contacted).
- llm-bench run -m NAME --preflight — validate config and check the endpoint answers; no load.
- llm-bench run -m NAME — run the benchmark; artifacts land in ~/.local/share/llm-bench/runs/<timestamp>/.
- llm-bench serve — open a local browser view of your runs, with a drop-down to pick any run.

LOAD MODES (closed is the default)

- closed: N parallel clients, each sending the next request on completion. Set N with
  `concurrency_levels` in config, or on the CLI with --concurrency 1,4,16 and --duration 30s.
  Good for per-user latency/throughput and capacity sizing; the tail is optimistic under saturation.
- open: requests arrive at a fixed Poisson rate regardless of server speed, via
  --request-rate 5 --request-rate 20 (repeatable; implies --mode open). Honest p99 for SLO checks.

QUALITY (optional, runs asynchronously and never perturbs the timing)

- Add an `evaluation` block to the config, then pass --eval-method embedding (cosine similarity
  vs each prompt's expected_output; needs a threshold) or --eval-method judge (LLM-as-judge rubric).
  Read the `eval` block in summary.json for coverage and scores.

Metrics: TTFT, TPOT, ITL, end-to-end latency, throughput, goodput (SLO-passing rate), reliability,
and cost. Run `llm-bench run --help` for the full flag list.
"""

app = typer.Typer(
    name="llm-bench",
    help=_APP_HELP,
    no_args_is_help=True,
    add_completion=False,
)

_NOT_IMPLEMENTED = "not implemented yet (scaffolding); implemented in a later task"


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"llm-bench {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the llm-bench version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """llm-bench: a duration-based load and quality benchmark for LLM endpoints."""


@app.command()
def run(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml (default: ~/.config/llm-bench/config.yaml)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model registry entry to benchmark."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Run artifacts directory (default: ~/.local/share/llm-bench/runs/<timestamp>/).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and resolve the config only; do not contact the SUT."),
    ] = False,
    preflight: Annotated[
        bool,
        typer.Option(
            "--preflight",
            help="Validate the config AND verify the endpoint answers a single request, then stop.",
        ),
    ] = False,
    raw_itl: Annotated[
        bool,
        typer.Option("--raw-itl", help="Persist the full per-request inter-token-latency list."),
    ] = False,
    prompts: Annotated[
        Path | None,
        typer.Option("--prompts", help="External prompt set (YAML list) overriding the built-in library."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Master seed for reproducible prompt selection."),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Load mode override: 'closed' (concurrency) or 'open' (arrival rate)."),
    ] = None,
    request_rate: Annotated[
        list[float] | None,
        typer.Option("--request-rate", help="Open-loop arrival rate(s) in req/s (repeatable); sets mode to open."),
    ] = None,
    concurrency: Annotated[
        str | None,
        typer.Option(
            "--concurrency",
            help="Closed-loop concurrency level(s), comma-separated (e.g. 1,2,4); overrides concurrency_levels.",
        ),
    ] = None,
    duration: Annotated[
        str | None,
        typer.Option("--duration", help="Per-level run duration override (e.g. 30s, 2m); applies to both modes."),
    ] = None,
    slo: Annotated[
        list[str] | None,
        typer.Option("--slo", help="Override an active-profile SLO threshold as key=value (e.g. ttft_ms=300)."),
    ] = None,
    slo_profile: Annotated[
        str | None,
        typer.Option("--slo-profile", help="Select the active SLO profile by name (e.g. interactive, relaxed)."),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="Override the per-request output token cap (run.max_tokens)."),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", help="Override the sampling temperature (run.temperature)."),
    ] = None,
    log_format: Annotated[
        str,
        typer.Option("--log-format", help="Log output format on stderr: 'text' (default) or 'json'."),
    ] = "text",
    eval_method: Annotated[
        str | None,
        typer.Option("--eval-method", help="Quality evaluation method: 'embedding' or 'judge'."),
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option(
            "--judge-model", help="Registry model to use as the LLM judge (overrides evaluation.judge.model)."
        ),
    ] = None,
    judge_rubric: Annotated[
        str | None,
        typer.Option("--judge-rubric", help="Judge rubric: 'score' (0..1), 'three_level', or 'binary'."),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model", help="Registry model to use as the embeddings endpoint (must serve /v1/embeddings)."
        ),
    ] = None,
) -> None:
    """Run a benchmark sweep against one model and write a run directory.

    Closed-loop (default): one sweep per concurrency level in `concurrency_levels`
    (override with --concurrency 1,4,16), each for --duration. Open-loop: pass
    --request-rate (repeatable) for fixed Poisson arrival rates. Use --preflight
    to validate + probe the endpoint without load, or --dry-run to resolve the
    config only. Optional quality scoring via --eval-method embedding|judge.

    Artifacts (in --out, default ~/.local/share/llm-bench/runs/<timestamp>/):
    summary.json, raw.jsonl, rollup.parquet, plus resolved-config and trace
    snapshots. Browse them with `llm-bench serve`.
    """
    bench_config = _load_or_exit(config if config is not None else default_config_file())
    _apply_cli_overrides(
        bench_config,
        mode=mode,
        request_rate=request_rate,
        concurrency=concurrency,
        duration=duration,
        slo=slo,
        slo_profile=slo_profile,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    _apply_eval_overrides_or_exit(
        bench_config, judge_model=judge_model, judge_rubric=judge_rubric, embedding_model=embedding_model
    )
    _apply_eval_method_or_exit(bench_config, eval_method)
    _configure_run_logging(log_format)
    _warn_self_preference(bench_config, model)

    if dry_run:
        _emit_dry_run(bench_config, model, out)
        raise typer.Exit(0)

    if preflight:
        # --preflight is a superset of --dry-run: emit the resolved endpoint and
        # redacted key (and the snapshot if --out), then verify the endpoint answers.
        _emit_dry_run(bench_config, model, out)
        _run_preflight_or_exit(bench_config, model)
        raise typer.Exit(0)

    effective_seed = seed if seed is not None else bench_config.run.seed
    library = _load_library_or_exit(prompts, effective_seed)
    out_dir = out if out is not None else default_run_dir(_effective_model_name(bench_config, model))
    typer.echo(f"writing run artifacts to {out_dir}")

    exit_code = _execute_run(bench_config, model, out_dir, raw_itl=raw_itl, library=library, seed=seed)
    raise typer.Exit(exit_code)


def _run_preflight_or_exit(bench_config: BenchConfig, model: str | None) -> None:
    """Validate the config and verify the endpoint answers one request (--preflight).

    On success prints the reachable endpoint and returns; on failure echoes the
    diagnostic on stderr and exits non-zero. Writes no run data.
    """
    try:
        asyncio.run(preflight_check(bench_config, model))
    except ValidationAbort as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except (RunAbort, ConfigError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo("pre-flight OK: endpoint answered")


def _apply_eval_method_or_exit(bench_config: BenchConfig, eval_method: str | None) -> None:
    """Select the eval method from ``--eval-method``, aborting on invalid config (FR-003).

    An embedding method without a threshold aborts at the config stage with the
    explanatory message and exit code 1 before any endpoint is contacted (FR-003).
    """
    if eval_method is not None and eval_method not in {"embedding", "judge"}:
        typer.echo(f"invalid --eval-method: {eval_method} (expected 'embedding' or 'judge')", err=True)
        raise typer.Exit(2)
    try:
        apply_eval_method(bench_config, eval_method)
    except ConfigValidationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


_DEFAULT_EMBED_THRESHOLD = 0.8
_RUBRICS: frozenset[str] = frozenset({"binary", "three_level", "score"})


def _apply_eval_overrides_or_exit(
    bench_config: BenchConfig, *, judge_model: str | None, judge_rubric: str | None, embedding_model: str | None
) -> None:
    """Point the judge / embedding at a registry model (and pick the judge rubric).

    Lets the UI choose evaluators without hand-editing the config: a named
    registry entry supplies the judge's or embedding's url / model / api_key, and
    an evaluation block is synthesised when the config has none. Exits non-zero on
    an unknown model name or rubric.
    """
    if not (judge_model or judge_rubric or embedding_model):
        return
    if judge_rubric is not None and judge_rubric not in _RUBRICS:
        typer.echo(f"invalid --judge-rubric: {judge_rubric} (expected {sorted(_RUBRICS)})", err=True)
        raise typer.Exit(2)
    evaluation = bench_config.evaluation or EvaluationConfig()

    if judge_model or judge_rubric:
        entry = _eval_entry_or_exit(bench_config, judge_model) if judge_model else None
        existing = evaluation.judge
        model = (
            JudgeModel(
                url=entry.base_url,
                model=entry.model,
                api_key=entry.api_key,
                prompt=existing.model.prompt if existing else None,
            )
            if entry is not None
            else (existing.model if existing else None)
        )
        if model is None:
            typer.echo(
                "--judge-rubric needs a judge model (pass --judge-model or add an evaluation.judge block)", err=True
            )
            raise typer.Exit(2)
        rubric = judge_rubric or (existing.rubric if existing else "score")
        evaluation.judge = JudgeConfig(model=model, rubric=rubric)  # type: ignore[arg-type]

    if embedding_model:
        entry = _eval_entry_or_exit(bench_config, embedding_model)
        existing_embed = evaluation.embedding
        evaluation.embedding = EmbeddingConfig(
            url=entry.base_url,
            model=entry.model,
            api_key=entry.api_key,
            threshold=existing_embed.threshold
            if existing_embed and existing_embed.threshold is not None
            else _DEFAULT_EMBED_THRESHOLD,
            rate_limit=existing_embed.rate_limit if existing_embed else None,
        )

    bench_config.evaluation = evaluation


def _eval_entry_or_exit(bench_config: BenchConfig, name: str) -> ModelRegistryEntry:
    """Resolve a registry entry by name for an eval override, aborting if unknown."""
    try:
        return bench_config.model_entry(name)
    except (ConfigError, ConfigValidationError, KeyError, ValueError) as exc:
        typer.echo(f"unknown model for evaluation: {name!r} ({exc})", err=True)
        raise typer.Exit(2) from exc


def _warn_self_preference(bench_config: BenchConfig, model: str | None) -> None:
    """Warn when the judge model family matches the SUT family (FR-044, E2E-110).

    Self-preference bias is a known LLM-as-judge failure mode; the run is allowed
    to proceed but the operator is warned. Inert unless judge evaluation is active.
    """
    evaluation = bench_config.evaluation
    if evaluation is None or evaluation.method != "judge" or evaluation.judge is None:
        return
    sut_family = _model_family(bench_config.model_entry(model).model)
    judge_family = _model_family(evaluation.judge.model.model)
    if sut_family == judge_family:
        logging.getLogger("llm_bench").warning("judge model family matches SUT (self-preference bias risk)")


def _model_family(model: str) -> str:
    """Return a model's provider/family token (the segment before the first ``/``)."""
    return model.split("/", 1)[0]


def _load_library_or_exit(prompts: Path | None, seed: int) -> PromptLibrary:
    """Load the built-in or external prompt library, aborting on empty/invalid sets.

    With no ``--prompts``, the default ``~/.config/llm-bench/prompts/short.yaml``
    (or the legacy top-level ``prompts.yaml``) is used when present; otherwise the
    built-in library is used. An empty external set aborts with ``no prompts
    loaded from <file>`` and a non-zero exit (FR-036); a malformed set aborts with
    a descriptive message.
    """
    if prompts is None:
        default_prompts = default_prompts_path()
        if default_prompts is None:
            return builtin_library(seed)
        prompts = default_prompts
    try:
        return load_prompts(prompts, seed)
    except EmptyPromptSetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except PromptError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _effective_model_name(bench_config: BenchConfig, model: str | None) -> str | None:
    """Return the registry entry name that will be benchmarked (for the run label).

    ``--model`` wins; otherwise the first registry entry is the default. Returns
    ``None`` when the registry is empty so the run dir falls back to a bare stamp.
    """
    if model is not None:
        return model
    return bench_config.models[0].name if bench_config.models else None


_SLO_KEYS: frozenset[str] = frozenset({"ttft_ms", "tpot_ms", "e2e_ms"})


def _apply_cli_overrides(
    bench_config: BenchConfig,
    *,
    mode: str | None,
    request_rate: list[float] | None,
    concurrency: str | None,
    duration: str | None,
    slo: list[str] | None,
    slo_profile: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> None:
    """Fold load-shaping CLI flags onto the config (FR-016/030).

    ``--request-rate`` sets the open-loop arrival rates and implies open mode;
    ``--mode`` overrides the load mode; ``--concurrency`` is closed-loop-only
    (rejected in open mode); ``--duration`` applies in both modes. ``--slo-profile``
    selects the active profile, ``--slo key=value`` overrides one threshold, and
    ``--max-tokens`` / ``--temperature`` tune generation. An unparseable or invalid
    flag exits non-zero.
    """
    if request_rate:
        bench_config.run.request_rates = list(request_rate)
        bench_config.run.mode = "open"
    if mode is not None:
        if mode not in {"closed", "open"}:
            typer.echo(f"invalid --mode: {mode} (expected 'closed' or 'open')", err=True)
            raise typer.Exit(2)
        # mypy cannot narrow str to Literal["closed", "open"] from the set guard above.
        bench_config.run.mode = mode  # type: ignore[assignment]
    # --concurrency is meaningless under open arrivals; reject rather than ignore.
    if concurrency is not None and bench_config.run.mode != "closed":
        typer.echo(f"--concurrency only applies in closed mode (current mode: {bench_config.run.mode})", err=True)
        raise typer.Exit(2)
    if concurrency is not None:
        bench_config.run.concurrency_levels = _parse_concurrency(concurrency)
    if duration is not None:
        bench_config.run.duration = _validate_duration(duration)
    _apply_generation_overrides(bench_config, slo_profile=slo_profile, max_tokens=max_tokens, temperature=temperature)
    apply_slo_overrides(bench_config, _parse_slo_overrides(slo))


def _apply_generation_overrides(
    bench_config: BenchConfig, *, slo_profile: str | None, max_tokens: int | None, temperature: float | None
) -> None:
    """Apply the ``--slo-profile`` / ``--max-tokens`` / ``--temperature`` overrides."""
    if slo_profile is not None:
        allowed = set(bench_config.slo_profiles) | {"interactive", "relaxed"}
        if slo_profile not in allowed:
            typer.echo(f"unknown --slo-profile: {slo_profile!r} (known: {sorted(allowed)})", err=True)
            raise typer.Exit(2)
        bench_config.run.slo_profile = slo_profile
    if max_tokens is not None:
        if max_tokens < 1:
            typer.echo(f"invalid --max-tokens: {max_tokens} (must be >= 1)", err=True)
            raise typer.Exit(2)
        bench_config.run.max_tokens = max_tokens
    if temperature is not None:
        if temperature < 0:
            typer.echo(f"invalid --temperature: {temperature} (must be >= 0)", err=True)
            raise typer.Exit(2)
        bench_config.run.temperature = temperature


def _parse_concurrency(value: str) -> list[int]:
    """Parse a comma-separated ``--concurrency`` list (e.g. ``"1,2,4"``) into levels.

    Each level must be a positive integer. An empty, non-integer, or below-one
    value exits non-zero with a descriptive message.
    """
    levels: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            level = int(item)
        except ValueError:
            typer.echo(f"invalid --concurrency: {item!r} is not an integer", err=True)
            raise typer.Exit(2) from None
        if level < 1:
            typer.echo(f"invalid --concurrency: level must be >= 1 (got {level})", err=True)
            raise typer.Exit(2)
        levels.append(level)
    if not levels:
        typer.echo("invalid --concurrency: no levels given", err=True)
        raise typer.Exit(2)
    return levels


def _validate_duration(value: str) -> str:
    """Validate a ``--duration`` override (e.g. ``"30s"``, ``"2m"``), returning it as-is.

    The value is kept as a string (``run.duration`` is parsed later) but checked
    now so a malformed duration fails fast with exit code 2.
    """
    try:
        parse_duration(value)
    except ValueError:
        typer.echo(f"invalid --duration: {value!r} (expected e.g. 30s, 500ms, 2m)", err=True)
        raise typer.Exit(2) from None
    return value


def _parse_slo_overrides(slo: list[str] | None) -> dict[str, float]:
    """Parse ``--slo key=value`` pairs into a threshold override mapping (FR-030)."""
    if not slo:
        return {}
    overrides: dict[str, float] = {}
    for item in slo:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if sep != "=" or key not in _SLO_KEYS:
            typer.echo(f"invalid --slo override: {item!r} (expected one of {sorted(_SLO_KEYS)} as key=value)", err=True)
            raise typer.Exit(2)
        try:
            overrides[key] = float(raw.strip())
        except ValueError as exc:
            typer.echo(f"invalid --slo value: {item!r} ({exc})", err=True)
            raise typer.Exit(2) from exc
    return overrides


class _JsonLogFormatter(logging.Formatter):
    """Render log records as single-line JSON objects (FR-056).

    Every line carries at least ``{timestamp, level, event}``; ``event`` is the
    structured event name when supplied via ``extra={"event": ...}`` and falls
    back to the logger function name otherwise. Prompts, responses, and secrets
    are never logged, so no message payload can leak them (FR-057).
    """

    _RESERVED = frozenset(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        """Serialize one record to a compact JSON line."""
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": getattr(record, "event", record.funcName),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and key != "event":
                payload[key] = value
        return json.dumps(payload, default=str)


def _configure_run_logging(log_format: str = "text") -> None:
    """Attach a stderr handler to the package logger for run warnings/events.

    The handler is bound to the *current* ``sys.stderr`` so warnings (min-samples,
    429 rate, event-loop lag) and lifecycle events are visible on the terminal and
    captured by tests. With ``log_format == "json"`` records are emitted as JSON
    objects (FR-056); otherwise a compact ``LEVEL message`` text format is used.
    Any previously attached run handler is replaced so successive invocations bind
    to the active stream.
    """
    package_logger = logging.getLogger("llm_bench")
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    for existing in [h for h in package_logger.handlers if getattr(h, "_llm_bench_run", False)]:
        package_logger.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    if log_format == "json":
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler._llm_bench_run = True  # type: ignore[attr-defined]
    package_logger.addHandler(handler)


def _execute_run(
    bench_config: BenchConfig,
    model: str | None,
    out: Path | None,
    *,
    raw_itl: bool,
    library: PromptLibrary,
    seed: int | None,
) -> int:
    """Drive the closed-loop run, mapping engine aborts to CLI exit codes.

    A validation abort exits 2 (FR-005/006); a pre-flight failure or disk-write
    failure exits 1 with a descriptive message; SIGINT yields exit 130 via the
    runner's graceful shutdown (FR-015).
    """
    try:
        return asyncio.run(_run_with_signal(bench_config, model, out, raw_itl=raw_itl, library=library, seed=seed))
    except ValidationAbort as exc:
        typer.echo(str(exc), err=True)
        return 2
    except PreflightAbort as exc:
        typer.echo(str(exc), err=True)
        return 1
    except RunAbort as exc:
        typer.echo(str(exc), err=True)
        return 1


async def _run_with_signal(
    bench_config: BenchConfig,
    model: str | None,
    out: Path | None,
    *,
    raw_itl: bool,
    library: PromptLibrary,
    seed: int | None,
) -> int:
    """Run the benchmark, cancelling it cleanly on SIGINT for a graceful flush."""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(run_benchmark(bench_config, model, out, raw_itl=raw_itl, library=library, seed=seed))
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, task.cancel)
    return await task


def _load_or_exit(config: Path) -> BenchConfig:
    """Load and validate a config, mapping config errors to CLI exit codes.

    A missing env var exits with code 2 (FR-002); any other config error exits
    with code 1. In all cases no run data is written before this returns.
    """
    try:
        return load_config(config, dict(os.environ))
    except MissingEnvVarError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


def _emit_dry_run(bench_config: BenchConfig, model: str | None, out: Path | None) -> None:
    """Print the resolved endpoint and redacted key; persist snapshots if asked."""
    entry = bench_config.model_entry(model)
    api_key = REDACTED if entry.api_key else "none"
    typer.echo(f"base_url: {entry.base_url}")
    typer.echo(f"api_key: {api_key}")
    if out is not None:
        write_resolved_config(bench_config, out)


@app.command()
def serve(
    run: Annotated[
        str | None,
        typer.Argument(help="Run to open (full path or a bare name under the runs dir); omit for the default."),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", help="Port for the local web server."),
    ] = DEFAULT_PORT,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Do not open the browser automatically; just print the URL."),
    ] = False,
) -> None:
    """Open the llm-bench web UI (Dashboards / Run / Prompts) in your browser.

    A local server with three tabs: Dashboards (build custom charts over any run;
    the home page), Run (launch a benchmark from the config with a live progress
    bar), and Prompts (edit the prompt-library files under
    ``~/.config/llm-bench/prompts/``). Pass a run (full path or bare name) to open
    it directly. The Dashboards/Prompts tabs contact nothing; stop with Ctrl-C.
    """
    config_path = default_config_file()
    try:
        serve_reports(
            run,
            runs_dir=default_runs_dir(),
            config_path=config_path if config_path.is_file() else None,
            prompts_dir=default_prompts_dir(),
            dashboards_dir=default_dashboards_dir(),
            port=port,
            open_browser=not no_open,
        )
    except ReportServeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except OSError as exc:
        typer.echo(f"could not start the web server: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command(hidden=True)
def analyze(
    data: Annotated[
        Path,
        typer.Argument(help="Path to a run's raw.jsonl or rollup.parquet file."),
    ],
    sql: Annotated[
        str,
        typer.Option("--sql", help="DuckDB SQL query to execute against the data file."),
    ],
) -> None:
    """Run an ad-hoc DuckDB query against a run's data (table name ``data``).

    The data file is read in place by DuckDB (JSONL or Parquet, no ETL). A
    missing file or invalid SQL aborts non-zero with a clean message and no
    Python traceback (EXC-006a/EXC-006b).
    """
    try:
        rendered = run_query(data, sql)
    except AnalyzeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(rendered)


@app.command()
def models(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config.yaml (default: ~/.config/llm-bench/config.yaml)."),
    ] = None,
) -> None:
    """List the models registered in the config (no endpoint is contacted).

    Resolves the config (including ``$ENV:`` references) and prints one line per
    registry entry with its model id, resolved ``base_url``, redacted key, and
    capability flags. The first entry is the default selected when ``--model`` is
    omitted.
    """
    config_path = config if config is not None else default_config_file()
    bench_config = _load_or_exit(config_path)
    entries = bench_config.models
    typer.echo(f"{len(entries)} model(s) registered in {config_path} (* = default for --model):")
    for index, entry in enumerate(entries):
        typer.echo(_format_model_line(entry, default=index == 0))


@app.command()
def init() -> None:
    """Scaffold ~/.config/llm-bench/ (config.yaml + prompts/ + dashboards/) and runs dir.

    Idempotent: existing files are kept, never overwritten. Run this once after
    install, then edit the config and benchmark. ``prompts/long.yaml`` holds
    instruction-heavy long-input prompts; select either file with ``--prompts``
    or in the Prompts tab of ``serve``.
    """
    created, skipped = scaffold_config()
    for path in created:
        typer.echo(f"created: {path}")
    for path in skipped:
        typer.echo(f"kept existing: {path}")
    typer.echo("Next: edit the config, then run 'llm-bench models' or 'llm-bench run -m <name>'.")


def _format_model_line(entry: ModelRegistryEntry, *, default: bool) -> str:
    """Render one registry entry as a single descriptive line (secrets redacted)."""
    marker = "*" if default else " "
    key = REDACTED if entry.api_key else "none"
    caps = f"tools:{'yes' if entry.supports_tools else 'no'} vision:{'yes' if entry.supports_vision else 'no'}"
    price = ""
    if entry.price_input is not None and entry.price_output is not None:
        price = f"  price:{entry.price_input}/{entry.price_output}"
    return f"{marker} {entry.name}  [{entry.model}]  {entry.base_url}  key:{key}  {caps}{price}"


if __name__ == "__main__":
    app()
