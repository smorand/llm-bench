# llm-bench

`llm-bench` is a Python 3.13 asyncio command-line tool that benchmarks the performance and, optionally, the output quality of Large Language Model endpoints exposed through an OpenAI-compatible streaming API (`/v1/chat/completions`). It targets both self-hosted servers (vLLM, TGI, SGLang, llama.cpp) and hosted APIs.

Runs are duration-based: each concurrency level (closed loop) or arrival rate (open loop) holds load for a fixed wall-clock window rather than a fixed request count. The tool measures latency (TTFT, TPOT, ITL, E2E), throughput (per-user and system token rates, RPS), reliability (success, 429, timeout, error), goodput (SLO-aware throughput), and cost, with an asynchronous quality-evaluation pipeline (embedding cosine similarity by default, optional LLM-as-judge).

## Audience

Operators and engineers who run an OpenAI-compatible LLM endpoint and want honest, reproducible numbers: capacity planning, regression tracking, SLO verification, cost-per-request estimates, and output-quality spot checks.

## Two-phase model

`llm-bench` separates measurement from interpretation:

1. **`run` produces data.** A run contacts the SUT, drives the load, and writes a self-contained run directory (`raw.jsonl`, `rollup.parquet`, `summary.json`, plus snapshots and traces) to `--out`. With no `--out`, it defaults to a timestamped directory under `~/.local/share/llm-bench/runs/`, and prints the path it wrote. The endpoint is contacted only in this phase.
2. **`serve` shows the data.** It reads your stored runs and never contacts the SUT: it opens a local browser view (Dashboards home) where you build custom panels (pick x / group dimensions and `{metric, agg}` values) over any run, plus tabs to launch runs and edit prompt libraries.

This means you can browse any run offline, long after the benchmark finished, and share a run directory as a single artifact.

## Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)

## Install

```bash
make sync          # uv sync: create the venv and install dependencies
make install       # install llm-bench as a uv tool (system-wide) and scaffold config dirs
```

`make install` also runs `make init`, which scaffolds `~/.config/llm-bench/` with a starter `config.yaml` and a `prompts/` directory holding `short.yaml` (a copy of the built-in prompt library) and `long.yaml` (instruction-heavy long-input prompts for large-prefill / TTFT stress; select it with `--prompts ~/.config/llm-bench/prompts/long.yaml`), a `dashboards/` directory with a `default.yaml`, and the results directory `~/.local/share/llm-bench/runs/`. Run `make init` (or `llm-bench init`) on its own at any time; it is idempotent and never overwrites an existing file. You can manage the prompt and dashboard files visually in the **Prompts** and **Dashboards** tabs of `serve`.

## Key commands

```bash
make run ARGS='--help'   # run the CLI through uv
make test                # run the test suite (heavy tests excluded by default)
make check               # full quality gate: lint, format-check, typecheck, security, tests + coverage
```

Always drive the tools through `make`; do not call `uv run pytest` / `ruff` / `mypy` directly.

## Quick start

There is no `config.yaml` in the repository; `make init` / `llm-bench init` writes a starter `config.yaml` and a `prompts/` directory (`short.yaml`, `long.yaml`) to `~/.config/llm-bench/` for you to edit (or create your own). With no `-c/--config`, `llm-bench` reads `~/.config/llm-bench/config.yaml` (and, with no `--prompts`, `~/.config/llm-bench/prompts/short.yaml` when present, otherwise the built-in prompt library, which `short.yaml` mirrors). Pass `-c <path>` to use a config elsewhere (for example a `./config.yaml` in the current directory).

Create a `config.yaml`. The example below registers four models under test (IBM ICA Claude Haiku, Mistral Medium and Qwen behind the EI gateway, and a keyless local llama.cpp server), an embedding endpoint for quality evaluation, and the IBM ICA model as the judge, all wired through `$ENV:` interpolation. `--model <name>` selects which registry entry to benchmark; the first entry (`ibm-haiku`) is the default. The `local` entry shows the keyless case: with no `api_key`, no `Authorization` header is sent:

```yaml
models:
  - name: ibm-haiku
    base_url: $ENV:IBM_ICA_BASE_URL   # IBM ICA gateway, OpenAI-compatible
    model: claude-haiku-4-5
    api_key: $ENV:IBM_ICA_API_KEY     # resolved at load time; never persisted in clear
    supports_vision: true
    supports_tools: true
    # pricing omitted on purpose: token usage is unlimited on the IBM ICA gateway,
    # so cost metrics are simply skipped (add price_input/price_output to enable them).
  - name: ei-mistral-medium35
    base_url: $ENV:EI_MODEL_MISTRAL_MEDIUM35_URL
    model: mistral-medium-3.5         # set to the exact model id your gateway expects
    api_key: $ENV:EI_MODEL_MISTRAL_MEDIUM35_API_KEY
    supports_vision: false
    supports_tools: true
  - name: ei-qwen36
    base_url: $ENV:EI_MODEL_QWEN36_URL
    model: qwen3                      # set to the exact model id your gateway expects
    api_key: $ENV:EI_MODEL_QWEN36_API_KEY
    supports_vision: false
    supports_tools: true
  - name: local
    base_url: http://localhost:8080/v1   # llama.cpp server (llama-server), OpenAI-compatible
    model: local-model                   # any string: llama.cpp serves whatever model is loaded
    # no api_key: llama.cpp needs no auth, so no Authorization header is sent
    supports_vision: false
    supports_tools: false

run:
  mode: closed                        # closed (concurrency) or open (arrival rate)
  duration: 30s
  warmup: 5s
  cooldown: 5s
  min_samples: 30
  concurrency_levels: [1, 4, 16, 64]
  max_tokens: 128
  ignore_eos: true
  temperature: 0.0
  cache_busting: true
  timeout: 30s
  seed: 0
  slo_profile: interactive

slo_profiles:
  interactive: { ttft_ms: 500, tpot_ms: 50, e2e_ms: 5000 }
  relaxed:     { ttft_ms: 2000, tpot_ms: 200, e2e_ms: 30000 }

evaluation:
  method: none                        # none | embedding | judge (or pick via --eval-method)
  global_timeout: 60s
  embedding:
    local: cpu                        # built-in local embedder: cpu (bge-small) or gpu (bge-large); no server to run
    # url: http://localhost:8001/v1   # or an OpenAI-style /v1/embeddings endpoint (then set model:)
    # model: text-embedding-3-small
    threshold: 0.80                   # mandatory for the embedding method
    rate_limit: 20                    # eval calls/s to the embedding provider
  judge:
    rubric: three_level               # binary | three_level | score (model returns 0..1 -> quality_score)
    model:
      url: $ENV:IBM_ICA_BASE_URL
      api_key: $ENV:IBM_ICA_API_KEY
      model: claude-haiku-4-5
```

Then run a closed-loop sweep and browse the results in your browser:

```bash
make run ARGS='run --config config.yaml --model ibm-haiku'
make run ARGS='serve'   # opens a local page with a run picker
```

## Usage examples

List the models registered in the config (resolves `$ENV:`, redacts keys, contacts nothing; `*` marks the default):

```bash
make run ARGS='models -c config.yaml'
```

Validate and resolve a config without contacting the SUT (prints the resolved endpoint and a redacted key):

```bash
make run ARGS='run -c config.yaml -m ibm-haiku --dry-run'
```

Check that the endpoint actually answers, without running the benchmark. `--preflight` is a superset of `--dry-run`: it validates and resolves the config (printing the resolved `base_url` and a redacted `api_key`, and writing `resolved_config.json` when `--out` is given), then issues a single pre-flight request. It exits 0 with `pre-flight OK: endpoint answered` when reachable, or non-zero with `pre-flight verification failed` otherwise:

```bash
make run ARGS='run -c config.yaml -m ibm-haiku --preflight'
```

Closed-loop concurrency sweep with the full ITL list persisted and a reproducible seed:

```bash
make run ARGS='run -c config.yaml -m ibm-haiku --out runs/sweep --raw-itl --seed 42'
```

In closed mode the number of parallel clients is the `concurrency_levels` list (each level is run sequentially for `duration`). Override both on the command line without editing the config; `--concurrency` takes a comma-separated list and both flags are closed-mode-only (rejected with `--mode open`):

```bash
make run ARGS='run -c config.yaml -m ibm-haiku --out runs/sweep \
  --concurrency 1,4,16,64 --duration 30s'
```

Open-loop Poisson arrivals (repeat `--request-rate` for several rates; this implies `--mode open`):

```bash
make run ARGS='run -c config.yaml -m ibm-haiku --out runs/open \
  --request-rate 5 --request-rate 20 --request-rate 50'
```

Override active SLO thresholds and switch the load mode on the command line:

```bash
make run ARGS='run -c config.yaml --mode closed --slo ttft_ms=300 --slo e2e_ms=4000'
```

Quality evaluation, selecting the method at runtime (an embedding method without a threshold aborts before any endpoint is contacted):

```bash
make run ARGS='run -c config.yaml -m ibm-haiku --out runs/q --eval-method embedding'
make run ARGS='run -c config.yaml -m ibm-haiku --out runs/q --eval-method judge'
```

Browse your stored runs in the browser. `serve` starts a small local web server (IBM Carbon styling) and opens it, with three tabs:

- **Dashboards** (the home page) - build your own charts. A dashboard (a YAML file under `~/.config/llm-bench/dashboards/`) is a list of **panels**; each panel pivots a run's steady requests by an **x** dimension (e.g. `level_or_rate`, `osl_bucket`, `model`), an optional **group** dimension (one series per value), and one or more **`{metric, agg}`** values (`ttft`/`e2e`/`input_tokens`/`output_tokens`/`rps`/`system_tok_s`/`quality_score`/`cost_usd`… × `p50`/`p99`/`mean`/…). Chart type is `line` for a numeric x, `bar` for a categorical one (override with `chart:`). Pick a dashboard + a run (defaults to `default` over the newest run); **click a point/bar to read its value**; edit panels with a form (add/remove panels and values). `init` scaffolds `dashboards/default.yaml` (latency, throughput, latency-by-output-length, tokens, quality_score).
- **Run** - pick a model and **mode** (closed / open), choose the load (closed: concurrency-level checkboxes with presets + manual extras; open: arrival-rate req/s checkboxes + manual), a duration, a **prompts file** (or the built-in default), a **quality eval** (none / embedding / judge), and optional tuning (`max_tokens`, `temperature`, SLO profile, seed), then launch a benchmark with a **live progress bar**; the run executes in the background and lands in the runs directory. When you pick a quality eval you also choose the **judge model + rubric** (or **embedding model**) right from the registry, so you can grade with any reachable model without editing the config. Embedding can run **fully locally** (`local · CPU` / `local · GPU`, built-in fastembed: no embeddings server to run; the model downloads once on first use), or against any `/v1/embeddings` endpoint. Quality eval fills the `quality_score` (0..1) metric: embedding cosine, or a judge model (rubric `binary`/`three_level`, or `score` where the model returns a 0..1 compliance number). The same is on the CLI: `--eval-method`, `--judge-model NAME`, `--judge-rubric score`, `--embedding-model NAME`.
- **Prompts** - choose a prompt-library file from `~/.config/llm-bench/prompts/` and edit it as a **form** (one card per prompt, with add/remove buttons for prompts and messages, category/length selectors, and an advanced raw-YAML escape hatch for `tools`/`tool_results`), or create a new one. Save serialises the whole form back to YAML server-side and validates it, so a broken file is never written; multimodal (image) message content round-trips intact.

Every form field has a small **(i)** you can click for a one-line explanation. The same knobs are available on the CLI: `--mode`, `--concurrency 1,4,16` / `--request-rate`, `--duration` (both modes), `--max-tokens`, `--temperature`, `--slo-profile`, `--seed`.

`--no-open` just prints the URL; `--port` sets the port. Stop it with Ctrl-C:

```bash
make run ARGS='serve'                       # Dashboards home + Run + Prompts tabs
make run ARGS='serve 2026-06-25_22-41-52'   # open one run directly (bare name resolved in the runs dir)
make run ARGS='serve --port 9000 --no-open' # custom port, do not auto-open the browser
```

Each run directory is named `<timestamp>_<model>` (e.g. `2026-06-26_08-00-05_ibm-haiku`) so runs are identifiable at a glance.

The terminal also prints a per-level summary table at the end of each run; for the raw numbers read `summary.json` (per-level `ttft`/`tpot`/`e2e` percentiles, `rps`, `goodput_attainment`, plus the `eval` block when quality scoring ran).

## Metrics cheat-sheet

| Metric | One-line definition |
|--------|---------------------|
| TTFT | Time To First Token: send to the first content-bearing chunk, ignoring an initial role-only chunk. |
| TPOT | Time Per Output Token: `(E2E - TTFT) / (output_tokens - 1)`, request-weighted; null for single-token responses. |
| ITL | Inter-Token Latency: token-weighted distribution of inter-token gaps, excluding the TTFT gap. |
| E2E | End-to-end latency: send to the last token. |
| Goodput | Throughput counting only requests that meet every active SLO threshold (TTFT, TPOT, E2E). |
| Per-user throughput | Output tokens/s during one request's generation window, averaged across requests (what a single client feels). |
| System throughput | Aggregate output tokens/s across all concurrent requests over the steady window (server capacity). |

Tokens come from the server `usage` field, never from `max_tokens`. Percentiles are `mean/min/max/std/p50/p90/p95/p99` (plus `p99.9` once a level has at least ~1000 steady samples). See `docs/metrics-glossary.md` and `.agent_docs/metrics.md` for full definitions.

## Configuration reference

The config has four top-level blocks (see `.agent_docs/configuration.md` for the annotated schema):

- **`models`** (registry): one entry per benchmarkable model with `name`, `base_url`, `model`, optional `api_key`, `tokenizer`, `supports_vision` / `supports_tools` capability flags, and optional `price_input` / `price_output` (USD per 1M tokens). `--model` selects an entry by `name`; the first entry is the default.
- **`run`**: `mode`, `duration`, `warmup`, `cooldown`, `min_samples`, `concurrency_levels`, `request_rates`, `burstiness`, `max_outstanding`, `max_tokens`, `ignore_eos`, `temperature`, `cache_busting`, `retries`, `timeout`, `seed`, `slo_profile`, plus the tuning keys `eval_queue_maxsize` and `event_loop_lag_threshold_ms`.
- **`slo_profiles`**: named threshold sets (`ttft_ms`, `tpot_ms`, `e2e_ms`); built-in `interactive` and `relaxed` apply when the block is omitted. `run.slo_profile` selects the active one; goodput counts only requests meeting all three.
- **`evaluation`**: `method` (`none` / `embedding` / `judge`), an `embedding` block (`url`, `model`, `threshold`, `rate_limit`) and/or a `judge` block (`rubric`, `model.{url, api_key, model, prompt}`), and a `global_timeout`.

**`$ENV:` interpolation.** Any string value may be `$ENV:VAR` (whole-value) or contain `${VAR}` tokens; both are resolved against the process environment at load time, and a missing variable aborts the load with a clear message. Secrets (`api_key`) are **never** written in clear: every persisted snapshot redacts them to `***`.

## Run artifacts

A run directory (`--out`) holds:

| File | Contents |
|------|----------|
| `raw.jsonl` | One JSON record per measured request (steady/warmup/cooldown). `itl_list` is omitted unless `--raw-itl` is set. |
| `rollup.parquet` | Columnar roll-up of `raw.jsonl` (same row count), nested objects stored as JSON strings, DuckDB/pyarrow-readable. |
| `summary.json` | Per-level/per-rate aggregates: reliability, latency percentiles, throughput, goodput, cost, eval coverage, and `status`. Rendered by `serve`. |
| `traces.jsonl` | OpenTelemetry JSONL spans for the internal LLM calls (model, tokens, duration only; never prompts, responses, or secrets). |
| `resolved_config.json` | The fully resolved config with secrets redacted and the active SLO thresholds recorded. |
| `env_snapshot.json` | Tool version, UTC timestamp, Python version, and platform. |
| `tool_calls.jsonl` | One line per deterministic mock tool invocation, written only when a tool round-trip occurred. |

## Testing

The end-to-end suite runs fully offline against an in-process fake OpenAI-compatible server (`FakeSUT`) plus fake embedding/judge endpoints (`FakeEval`); see `tests/conftest.py`. Two extreme-scale `heavy` tests (1000-virtual-user saturation and a large-percentile recompute) are excluded from the default run; run them explicitly with:

```bash
uv run pytest -m heavy
```

## Documentation

- `docs/operator-guide.md`: how to read results, choosing closed vs open loop, interpreting goodput, the heavy tests.
- `docs/metrics-glossary.md`: operator-facing glossary of every term.
- `.agent_docs/`: contributor docs (architecture, metrics, configuration, evaluation).

## License

MIT. Copyright (c) 2026 Sebastien MORAND.
