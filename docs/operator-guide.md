# Operator guide

How to run `llm-bench`, choose a load model, and read the results. This guide is operator-facing; for metric math see `docs/metrics-glossary.md` and `.agent_docs/metrics.md`.

## The two-phase workflow

1. `llm-bench run ... --out runs/r1` contacts the SUT and writes the run directory.
2. `llm-bench report runs/r1` and `llm-bench analyze runs/r1/raw.jsonl --sql "..."` consume that directory and never contact the SUT.

A run directory is a complete, shareable artifact. Re-render or re-query it any time, offline.

## Choosing closed-loop vs open-loop

| | Closed-loop (`--mode closed`, default) | Open-loop (`--mode open` / `--request-rate`) |
|---|---|---|
| Knob | `concurrency_levels` (N virtual users; CLI `--concurrency 1,4,16`) | `request_rates` (req/s, Poisson; CLI `--request-rate`) |
| Mental model | "N clients, each sends the next request on completion." | "Traffic arrives at a fixed rate regardless of how fast the server answers." |
| Tail latency | Optimistic under saturation (coordinated omission). | Honest under saturation. |
| Use it for | Per-user generation rate, simple capacity sizing. | SLO verification, realistic p99 under load, finding the rate where latency falls off a cliff. |

**Coordinated omission.** When you push closed-loop into saturation, the reported p99 is too good: during a server stall no new requests are issued, so the slow window is undersampled. If your question is "what latency will real users see at X req/s", run open-loop. See the metrics glossary entry for the full explanation.

A common pattern: a closed-loop sweep (`concurrency_levels: [1, 4, 16, 64]`) to find peak system throughput, then an open-loop run at rates just under that peak to confirm tail latency holds. The closed-loop levels and per-level duration can be set without editing the config via `--concurrency 1,4,16,64 --duration 30s` (both flags are closed-mode-only).

## Reading the terminal output

`run` prints a rich table, one row per concurrency level (or arrival rate):
- `samples`: steady-phase request count (warmup/cooldown excluded). If this is below `min_samples` you will see a warning; lengthen `duration` or lower `min_samples`.
- `e2e_p50` / `e2e_p99`, `ttft_p50` / `ttft_p99`: latency percentiles in ms.
- `success%` and `rate_limited%`: reliability. A steady 429 rate above 1% is flagged ("rate limiting detected"); you are hitting a provider limit, not measuring the server.

Open-loop runs additionally print `max_outstanding reached N times` when the in-flight guard engaged: the server could not keep up with the arrival rate, so arrivals were paused. That is itself a finding (the rate exceeds capacity).

If you see `event loop lag ... (client saturation)`, the **benchmark client** is the bottleneck, not the server; reduce concurrency or run on a bigger box. Enable it with `run.event_loop_lag_threshold_ms`.

## Interpreting goodput

Latency percentiles tell you how fast requests were; **goodput** tells you how many were fast enough. A request counts toward goodput only if it meets every active SLO threshold (TTFT, TPOT, E2E) at once.

- `goodput_attainment` is the fraction of steady-success requests that met the SLO. `1.0` means every served request was within budget; `0.6` means 40% blew at least one threshold.
- `goodput_rps` is useful goodput per second: the rate at which the server delivered SLO-compliant responses. This is the number to size capacity against, not raw RPS.

Pick the profile with `run.slo_profile` (`interactive` is strict, `relaxed` is lenient) or override a single threshold on the command line, e.g. `--slo ttft_ms=300`. The thresholds actually applied are recorded in `resolved_config.json` under the `slo` block.

Per-user vs system throughput answer different questions: `per_user_tok_s` is what one client feels, `system_tok_s` is total server capacity. They diverge as concurrency rises (per-user drops while system climbs, then plateaus).

## Quality evaluation

Add `--eval-method embedding` or `--eval-method judge` (the config must carry an `evaluation` block). Evaluation runs asynchronously and is joined back after the perf summary, so it never perturbs the load timing. Read `summary.json`'s `eval` block: `coverage = judged / eligible`. Records can be `eval_skipped` (timeout or endpoint down) or `eval_dropped` (eval queue overflowed); both leave the perf data fully valid. The judge default is `claude-haiku-4-5` via `IBM_ICA_BASE_URL` / `IBM_ICA_API_KEY`.

## Reports and ad-hoc analysis

`report runs/r1` writes a self-contained `report.html` (six standard charts, plotly.js inlined, fully offline) you can email as a single file. An empty or interrupted run renders a banner instead of failing.

`analyze` runs DuckDB SQL straight over `raw.jsonl` or `rollup.parquet` (registered as the table `data`), with no import step. Example: recompute a percentile, group reliability by level, or compute cost per category. Use the Parquet rollup for large runs (columnar, faster scans).

## The heavy tests

The default test run excludes two extreme-scale `heavy` tests (a 1000-virtual-user closed-loop saturation test and a large-percentile recompute). They are slow and resource-hungry, so they are opt-in:

```bash
uv run pytest -m heavy
```

Run them when validating behavior at high concurrency or after touching the engine's concurrency or percentile paths.
