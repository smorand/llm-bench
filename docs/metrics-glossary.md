# Metrics glossary

Operator-facing definitions of every term `llm-bench` reports. Mirrors Section 16 of the specification.

| Term | Definition | Context |
|------|------------|---------|
| TTFT | Time To First Token: time from request send to the first content chunk (an initial role-only chunk is ignored). | Metrics |
| TPOT | Time Per Output Token: `(E2E - TTFT) / (output_tokens - 1)`, request-weighted. Null for a single-token response. | Metrics |
| ITL | Inter-Token Latency: token-weighted distribution of inter-token gaps, excluding the TTFT gap. | Metrics |
| E2E | End-to-end latency: send to the last token. | Metrics |
| normalized latency | `E2E / output_tokens`: end-to-end time per output token. | Metrics |
| Goodput | Throughput counting only requests that meet every active SLO threshold (TTFT, TPOT, E2E). | Metrics |
| per-user tok/s | Output tokens per second during one request's generation window, averaged across requests. What a single client feels. | Metrics |
| system tok/s | Aggregate output tokens per second across all concurrent requests over the steady window. Server capacity. | Metrics |
| RPS | Requests per second completed over the steady window. | Metrics |
| Closed-loop | Fixed number of virtual users, each sending the next request on completion. | Load model |
| Open-loop | Fixed arrival rate (Poisson), independent of completion times. | Load model |
| Coordinated omission | Underestimation of tail latency in closed-loop under saturation: during a server stall no new requests are issued, so the slow window is undersampled and the reported p99 is optimistically low. Open-loop, which issues arrivals on a fixed schedule regardless of completion, avoids this and gives honest tails. | Load model |
| Warmup / steady / cooldown | Per-level phases; metrics are computed over the steady phase only. | Methodology |
| Cache busting | A unique per-request prefix added to avoid prefix-cache bias. A cache hit despite busting is warned and counted. | Anti-cache |
| ISL / OSL | Input / Output Sequence Length buckets (short / medium / long). | Dimensions |
| SUT | System Under Test: the model endpoint being benchmarked. | Benchmark |
| SLO profile | A named set of latency thresholds (`interactive`, `relaxed`) used to compute goodput. | Configuration |
| LLM-as-judge | Scoring output quality with a separate judge model: a `binary`/`three_level` verdict, or a `score` rubric where the model returns a 0..1 compliance number. | Evaluation |
| Embedding cosine | Semantic similarity of expected vs actual output via embedding vectors, compared against an inclusive threshold. | Evaluation |
| quality_score | Unified 0..1 quality metric (embedding cosine, the judge's score, or its verdict mapped to 0..1); charted in the Dashboards tab. | Evaluation |
| Coverage | Fraction of eligible requests actually evaluated (`judged / eligible`). | Evaluation |

## Percentiles and reliability

- **Percentile set.** Latency metrics report `mean`, `min`, `max`, `std`, `p50`, `p90`, `p95`, `p99`. `p99.9` is added only once a level has at least ~1000 steady samples; below that it is omitted, never shown as null.
- **Tokens.** All token counts come from the server `usage` field, never from `max_tokens`.
- **Reliability outcomes.** Each request is classified as `success`, `rate_limited` (HTTP 429), `timeout`, `malformed_stream` (interrupted or unparseable SSE), or `connection_error`. A steady 429 rate above 1% is flagged.

## Cost

When the model entry defines `price_input` and `price_output` (USD per 1M tokens), each request gets `cost_usd`, each level reports `total_cost_usd` and `cost_per_1k_requests`, and the run summary aggregates across levels. Without pricing, cost fields are omitted.
