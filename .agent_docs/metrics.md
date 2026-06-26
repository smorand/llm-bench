# Metrics

All metric math lives in `metrics.py`; it is pure and side-effect free. The engine feeds it monotonic arrival offsets and the parsed server `usage` object; it returns derived latencies and the per-level percentile / throughput / goodput objects. Per-request latencies are stored in **seconds**; SLO thresholds are in **milliseconds**.

## Per-request metrics

| Metric | Definition | Edge cases |
|--------|------------|------------|
| TTFT | Time from request send to the first content-bearing chunk. An initial role-only chunk (delta with no `content`) is ignored. | A request that streams only a role chunk has no TTFT. |
| time-to-second-token (`tt2t`) | Gap from the first content token to the second. | Null when fewer than two content tokens. |
| TPOT | `(E2E - TTFT) / (output_tokens - 1)`, request-weighted. | Null when `output_tokens <= 1` (no division guard needed). |
| ITL | Inter-token gaps between successive content arrivals, excluding the TTFT gap. | Empty for a single-token response, so `itl_summary` is `None` (no spurious zeros). |
| E2E | Send to last token (full round-trip wall time). | |
| normalized latency | `E2E / output_tokens`. | Null when `output_tokens <= 0`. |

`itl_summary` is `{mean, p50, p95, p99, max}`. The full per-request `itl_list` is kept in memory for level pooling regardless, but only persisted to `raw.jsonl` when `--raw-itl` is set.

## Token accounting

- `output_tokens`, `prompt_tokens`, `cached_tokens`, `reasoning_tokens` come from the server `usage` field, never from `max_tokens`. Cached/reasoning tokens are read from `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`.
- If a stream ends without a `usage` object, the record is flagged `usage_incomplete` and `output_tokens` falls back to the count of content deltas seen.
- ISL/OSL buckets: input `< 256` short, `< 1024` medium, else long; output `< 32` short, `< 256` medium, else long.

## Per-level aggregation

Computed over **steady-phase, success** records only (warmup and cooldown are excluded).

- Percentile objects: `{mean, min, max, std, p50, p90, p95, p99}`, plus `p999` only when a level has at least 1000 steady samples (else the key is omitted, never null). All percentiles use `numpy.percentile` with linear interpolation so the suite can recompute them from `raw.jsonl` within a ±2% tolerance.
- ITL at the level is **token-weighted**: every inter-token gap from every steady-success record is pooled into one distribution (not an average of per-request summaries).
- TPOT at the level is **request-weighted**: the per-request TPOT values (first token excluded by the formula) are aggregated.

## Throughput (per-user vs system)

| Field | Meaning |
|-------|---------|
| `per_user_tok_s` | Mean over requests of `output_tokens / (E2E - TTFT)` (the generation window of one request). What a single client experiences. |
| `system_tok_s` | `sum(output_tokens) / steady_window`. Aggregate server output capacity. |
| `total_tok_s` | `sum(output_tokens + prompt_tokens) / steady_window`. |
| `rps` | `completed / steady_window`. |

The steady window is `max(t_start + e2e) - min(t_start)` across the level's records.

## Goodput vs SLO

`goodput` counts steady-success requests that meet **every** active threshold (`ttft_ms`, `tpot_ms`, `e2e_ms`). A `None` threshold means no bound on that dimension; a missing latency on a bounded dimension fails the threshold. Reported as:
- `goodput_count`: requests meeting all thresholds,
- `goodput_attainment`: that count over the steady-success total (`0.0` when none),
- `goodput_rps`: count over the steady window (`0.0` when the window is non-positive).

The active thresholds come from `run.slo_profile` (built-in `interactive` / `relaxed` when `slo_profiles` is omitted) and any `--slo key=value` overrides; both the resolved thresholds and goodput observe the override.

## Cost

When the model entry defines both `price_input` and `price_output` (USD per 1M tokens), each record carries `cost_usd = (prompt_tokens * price_input + output_tokens * price_output) / 1e6`. Levels add `total_cost_usd` and `cost_per_1k_requests`; the run summary aggregates across levels. Absent pricing, cost fields are simply omitted.

## Coordinated-omission caveat

Closed-loop holds N virtual users, each sending its next request only on completion. Under saturation this **understates tail latency**: when the server stalls, in-flight requests just wait, but no new requests are issued during the stall, so the slow period is sampled by far fewer requests than a real fixed-rate workload would send. The measured p99 is therefore optimistic.

Open-loop issues arrivals on a Poisson schedule **independent of completion times** (the engine records each request's scheduled arrival offset as `t_start`, not its dispatch instant). A stalled server still accrues scheduled arrivals, so the tail is sampled honestly. Use open-loop (`--mode open` / `--request-rate`) when you care about realistic tail latency under load; closed-loop matches the simpler N-users mental model and the per-user generation-rate question.
