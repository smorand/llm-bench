# Architecture

A modular monolith: layered modules behind small interfaces, all driven by the Typer CLI. Async-first (asyncio + httpx); `time.monotonic()` for every duration.

## Module map

| Module | Role |
|--------|------|
| `llm_bench.py` | Typer CLI: `run`, `serve`, `analyze`. Loads config, folds CLI overrides (`--mode` / `--request-rate` / `--slo` / `--eval-method`), configures logging (text/json), maps engine aborts to exit codes (2 validation, 1 preflight/disk, 130 SIGINT), warns on judge self-preference. |
| `config.py` | Config schema (pydantic): `models` registry, `run`, `slo_profiles`, `evaluation`. `$ENV:` / `${...}` interpolation, semantic validation (embedding needs a threshold), redacted `resolved_config.json` + `env_snapshot.json` writers. |
| `prompts.py` | Built-in prompt library + external `--prompts` YAML loader; seeded per-request selection (`PromptLibrary.select`); UUID cache-busting prefix; capability flags (`requires_tools` / `requires_vision`). |
| `runner.py` | The load engine. Closed-loop (concurrency) and open-loop (Poisson arrivals) sweeps, pre-flight, SSE stream consumption, outcome classification, per-request `RequestRecord`, eval enqueue + backfill, artifact persistence, terminal/guard summaries, event-loop lag monitor. |
| `metrics.py` | Pure, side-effect-free: per-request derived metrics (TTFT/TPOT/ITL/E2E/normalized), token-weighted ITL pooling, percentile objects, throughput, goodput, cost, `cosine_similarity`. |
| `evaluation.py` | Async eval pipeline: bounded queue, rate-limited worker pool, embedding cosine / judge rubric scoring, global-timeout draining, result join by `request_id`. |
| `serve.py` | Local HTTP server (`serve`), IBM Carbon UI, self-contained HTML. Dashboards tab (home): renders a dashboard's panels as interactive SVG charts (click a point/bar for its value) over a chosen run; form editor. Run tab: lists config models (raw YAML, no `$ENV:`) and launches a benchmark as a `python -m llm_bench run` subprocess (`JobRegistry`), polled for a live progress bar. Prompts tab: structured prompt-library editor. |
| `__main__.py` | `python -m llm_bench` entry point so the Run tab can launch a benchmark subprocess. |
| `dashboards.py` | Custom dashboards: parse/validate panels, pivot a run's steady `raw.jsonl` records (numpy) by x/group into `{metric, agg}` chart series, and form<->YAML for the Dashboards-tab editor. |
| `analyze.py` | Registers the data file as a DuckDB view named `data` (`read_json_auto` / `read_parquet`, no ETL) and runs the operator's SQL, rendering `key=value` rows. |
| `tracing.py` | OpenTelemetry JSONL span export to `traces.jsonl` (model/tokens/duration only). |
| `logging_config.py` | rich + file logging setup. |
| `version.py` | Build version (injected at build time). |

## Data flow

```
config.yaml ──load_config──> BenchConfig ──┐
--prompts / built-in ──> PromptLibrary ─────┤
                                            v
                                  runner.run_benchmark
                                            │
                  ┌─────────────────────────┼───────────────────────────┐
                  v                          v                           v
          closed loop                  open loop                    pre-flight
       (_run_level: N workers,   (_run_rate: gamma/Poisson      (one tagged request;
        next request on           arrivals, max_outstanding      failure aborts before
        completion)               semaphore guard)               any data is written)
                  └─────────────────────────┬───────────────────────────┘
                                            v
                               RequestRecord per request
                       (metrics.* fills TTFT/TPOT/ITL/E2E/cost/buckets)
                                            │
                       eligible records ────┼──> eval queue (side channel)
                                            v
                              build_summary (per-level/rate
                              reliability + percentiles + goodput)
                                            │
        ┌────────────────┬─────────────────┼──────────────────┬───────────────┐
        v                v                 v                  v               v
   raw.jsonl       rollup.parquet     summary.json       traces.jsonl   resolved_config.json
                                                                        + env_snapshot.json
                                            │
                                  serve.py / analyze.py
                                   (consume the run dir; never touch the SUT)
```

### Order of operations at run end

The perf summary is published **first** (terminal table + guard summary), then the eval queue is drained and its scores are backfilled onto the records by `request_id`, then the joined artifacts are written. This keeps perf timing independent of eval latency and guarantees perf data is valid even if the eval provider is unreachable.

## Eval side channel

The eval pipeline is decoupled from the load generator (see `evaluation.md`):
- The load generator enqueues a lightweight `EvalRecord` non-blocking for each eligible request (succeeded + non-empty `expected_output`). A full bounded queue drops the item and bumps a counter; it never blocks the load.
- A separate worker pool with its own rate limiter (distinct from SUT concurrency) drains the queue under a global timeout.
- Records with no usable expected reference are tagged `skipped_no_expected` and excluded from coverage.

## Test harness (FakeSUT / FakeEval)

The E2E suite (`tests/conftest.py`) runs fully offline. Every endpoint is replaced by a local, deterministic, in-process `aiohttp` server:
- `SUTController` / `fake_sut`: a scriptable OpenAI-compatible streaming chat-completions server (the system under test).
- `EvalController` / `fake_eval`: fake `/v1/embeddings` and judge `/v1/chat/completions` endpoints with deterministic outputs.
- `cfg_base`: writes a `config.yaml` wired to the running `FakeSUT` and sets `SUT_API_KEY`.

**Background-thread server model.** Each fake server runs on its own dedicated thread with its own event loop, because the production CLI runs the benchmark under its own `asyncio.run(...)` loop. If the fakes shared the test's loop it would be blocked inside `asyncio.run` for the whole run and the sockets would never be serviced (every real request would time out). The listening socket is bound before the thread starts so the chosen port is known race-free, then handed to aiohttp's `SockSite` inside the server loop.
