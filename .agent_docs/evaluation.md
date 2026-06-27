# Evaluation pipeline

The async, decoupled output-quality pipeline lives in `evaluation.py` (`EvalPipeline`), driven from `runner.py`. It scores how close a model's output is to a prompt's `expected_output`, without perturbing the load test's timing.

## Lifecycle

1. **Enqueue (during the load test, non-blocking).** On each completed request that carries a non-empty `expected_output` and `outcome == success`, the runner builds a lightweight `EvalRecord(request_id, expected, actual)` and calls `pipeline.enqueue`. The queue is bounded (`run.eval_queue_maxsize`, default 10000). If it is full, the item is dropped and an aggregate `dropped` counter is bumped; the load generator is **never** blocked.
2. **Drain (after the perf summary is published).** The runner publishes perf metrics first, then awaits `pipeline.drain`. Drain spawns the worker pool (4 workers), waits for the backlog to clear or the `global_timeout` to fire, then marks every still-pending record skipped.
3. **Backfill (join by `request_id`).** Results are joined onto the in-memory records by `request_id`, then the joined records are written to `raw.jsonl` / `rollup.parquet`. A coverage line (`Eval coverage: judged/eligible`) is printed and an `eval` block is added to `summary.json`.

## Rate limiting and isolation

The worker pool has its own `asyncio.Semaphore`-style pacing (`_RateLimiter`) against the embedding/judge provider, **distinct from the SUT concurrency**. For the embedding method, the limiter uses `evaluation.embedding.rate_limit` (calls/s); a non-positive or absent rate disables pacing. This keeps eval traffic from competing with or distorting the load test.

## Methods

- **Embedding (default).** Both `expected` and `actual` are embedded, cosine similarity (`metrics.cosine_similarity`, `0.0` on a zero vector) is compared against the inclusive `threshold`, and the record gets `sim_score`, `quality_score = sim`, and `quality_pass = sim >= threshold`. Two backends: `embedding.local: cpu|gpu` runs the vectors in-process via fastembed (`local_embed.py`; bge-small on CPU, bge-large + CUDA when available; no embeddings server, model downloaded once) on a thread so the event loop is never blocked; otherwise `embedding.url` POSTs to an OpenAI-style `/v1/embeddings`. The threshold is mandatory; its absence aborts at config stage.
- **Judge (LLM-as-judge).** A grading request goes to `<judge.model.url>/chat/completions`. Two rubric families:
  - `binary` / `three_level` ask for a JSON `{verdict, reason}`; the verdict is normalized (`pass`/`fail` or `correct`/`partial`/`incorrect`) and stored as `judge_verdict` + `judge_reason`, and also mapped to a 0..1 `quality_score` (pass/correct = 1, partial = 0.5, fail/incorrect = 0).
  - `score` asks for a JSON `{score, reason}` where the model returns a compliance number in 0..1 directly; it is clamped and stored as `quality_score`.
- **`quality_score` (0..1)** is the unified quality metric across methods (charted in the Dashboards tab); `sim_score` keeps the raw embedding cosine.

## `eval_status` values

Each record carries an `eval_status`:

| Status | Meaning |
|--------|---------|
| `judged` | Successfully scored (embedding or judge). Counts toward coverage. |
| `eval_skipped` | Eligible but not scored: the global timeout fired before it drained, or the eval endpoint was unreachable. |
| `eval_dropped` | Eligible but its `EvalRecord` was spilled on a full bounded queue (FR-041). |
| `skipped_no_expected` | The request had no usable `expected_output` (or did not succeed); excluded from coverage. |

Records with `None` or `skipped_no_expected` status are **not** counted as eligible. Coverage = `judged / eligible`.

## Failure handling

- An unreachable embedding/judge endpoint is detected on the first `HTTPError`, warned **once** (method-specific message), and every affected record is marked `eval_skipped`. The perf data stays valid.
- The global timeout bounds total drain time; whatever has not been scored is marked `eval_skipped` with a count in the warning.

## No-leakage guarantee

Only the embedding/judge calls are traced, and only with model / input-count / rubric / duration attributes. Prompts, responses, and secrets are never logged or traced. The transient join inputs on a record (`expected_output`, `output_text`) are stripped before `raw.jsonl` is written, so reference and model text never reach the persisted artifacts.
