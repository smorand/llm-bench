# llm-bench

## Overview
`llm-bench` is a Python 3.13 asyncio CLI that benchmarks the performance and optionally the output quality of OpenAI-compatible LLM endpoints. It measures latency (TTFT, TPOT, ITL, E2E), throughput, reliability, goodput, and cost, with closed-loop and open-loop load models, and an asynchronous quality-evaluation pipeline (embedding cosine, optional LLM-as-judge). Results are persisted as JSONL/Parquet and rendered as a rich terminal summary and a local browser view (`serve`).

Tech stack: Python 3.13, uv, Typer, httpx (async streaming), pydantic / pydantic-settings, DuckDB, pyarrow, stdlib http.server (serve), fastembed (local eval embeddings), rich, numpy, OpenTelemetry.

## Key Commands
```
make sync          # Install dependencies (uv sync)
make run           # Run the CLI
make test          # Run the test suite
make check         # Full quality gate (lint, format-check, typecheck, security, test-cov >= 80%)
make docker-build  # Build the Docker image
```
NEVER run `uv run pytest`/`ruff`/`mypy` directly; always use the `make` targets.

## Project Structure
- `src/llm_bench/llm_bench.py` : Typer CLI entry point (`run`, `serve`, `models`, `init`; `analyze` hidden)
- `src/llm_bench/config.py` : configuration schema, `$ENV:` interpolation, model registry, SLO profiles, evaluation block + `--eval-method` selection (FR-003)
- `src/llm_bench/runner.py` : closed/open-loop load engine, per-request records, artifact persistence, eval enqueue + concurrent draining (eval pool `start()`s before the sweep, `finish()`es after) + backfill orchestration
- `src/llm_bench/evaluation.py` : async eval pipeline (bounded queue, rate-limited worker pool draining concurrently with the load via `start()`/`finish()`, embedding cosine / judge rubric incl. `score` 0..1, unified `quality_score`, global-timeout tail coverage) joined on `request_id` (SC-004, FR-040..047)
- `src/llm_bench/local_embed.py` : built-in local embeddings (fastembed/ONNX) for `embedding.local: cpu|gpu` (no embeddings server); lazy, memoised
- `src/llm_bench/metrics.py` : per-request metrics, percentiles, throughput, goodput, cost, `cosine_similarity`
- `src/llm_bench/prompts.py` : built-in + external prompt library, seeded selection, cache-busting, capability gating (FR-033..039)
- `src/llm_bench/serve.py` : local web server (`serve`), IBM Carbon UI, three tabs - Dashboards (home `/`; dashboard-file + run pickers, renders custom panels as interactive SVG charts where clicking a point/bar shows its value; form editor; defaults to `default` over newest run; file names shown without `.yaml`), Run (model + mode + load presets + prompts-file + tuning form -> launches a benchmark subprocess with a live progress bar), Prompts (structured per-prompt form editor — cards with add/remove prompts/messages — `prompts_to_form`/`build_prompts_yaml`, validated on save); `_svg_line_chart`/`_svg_bar_chart`, `RunRequest`/`parse_run_form`, `JobRegistry`
- `src/llm_bench/dashboards.py` : custom dashboard engine - parse/validate panels, pivot a run's steady `raw.jsonl` records (numpy) by x/group into `{metric, agg}` series (per-request metrics + derived `rps`/`system_tok_s` over the steady window), form<->YAML (`dashboard_to_form`/`build_dashboard_yaml`), `STARTER_DASHBOARD`
- `src/llm_bench/__main__.py` : `python -m llm_bench` entry point (used by the Run-tab subprocess launcher)
- `src/llm_bench/analyze.py` : ad-hoc DuckDB SQL over `raw.jsonl` / `rollup.parquet` (table `data`, FR-052; CLI hidden)
- `src/llm_bench/logging_config.py` : rich + file logging setup
- `src/llm_bench/tracing.py` : OpenTelemetry JSONL tracing
- `src/llm_bench/version.py` : build version (injected at build time)
- `tests/conftest.py` : test harness (FakeSUT + FakeEval in-process OpenAI-compatible servers, cfg_base)

Run artifacts (`--out`): `raw.jsonl`, `rollup.parquet`, `summary.json`, `traces.jsonl`, `resolved_config.json`, `env_snapshot.json` (plus `tool_calls.jsonl` when a tool round-trip ran).

## Conventions
- Async-first (asyncio + httpx); `time.monotonic()` for all duration measurements.
- Config via `pydantic-settings` / pydantic models, never `os.environ` directly; secrets via `$ENV:` interpolation, never persisted in clear.
- Metrics come from the server `usage` field, never from `max_tokens`.
- The 112 E2E tests in `specs/` are the acceptance contract. Two extreme-scale `heavy` tests are excluded by default; run them with `uv run pytest -m heavy`.

## Quality Gate
Run `make check` before every commit. It runs: lint, format-check, typecheck (mypy strict), security (bandit), test-cov (>= 80% coverage).

## Auto-Evaluation Checklist
Before considering any task complete:
- [ ] `make check` passes
- [ ] No sync blocking calls in async code
- [ ] All external (LLM) calls traced with OpenTelemetry (model, tokens, duration, cost; never prompts/responses/secrets)
- [ ] No forbidden practices (bare except, print, mutable defaults, .format(), assert in production)
- [ ] Config via Settings/pydantic, not os.environ
- [ ] Dependencies injected, not created inline
- [ ] Test coverage >= 80%

## Coding Standards
This project follows the `python` skill. Reload it for the full coding standards reference.

## Documentation Index
- `.agent_docs/architecture.md` : module map, data flow, eval side-channel, FakeSUT/FakeEval harness
- `.agent_docs/metrics.md` : exact metric definitions, conventions, coordinated-omission caveat
- `.agent_docs/configuration.md` : full config schema, `$ENV:` interpolation, annotated example
- `.agent_docs/evaluation.md` : async eval pipeline, embedding vs judge, `eval_status` values, no-leakage guarantee
- `.agent_docs/python.md` : Python coding standards
- `.agent_docs/makefile.md` : Makefile documentation
- `docs/operator-guide.md` : reading results, closed vs open loop, goodput, heavy tests
- `docs/metrics-glossary.md` : operator-facing glossary (Section 16)
- `specs/` : project specification and E2E test contract
