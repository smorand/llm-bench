# Configuration

Config is a single `config.yaml` loaded by `config.load_config`, interpolated, then validated against the pydantic schema in `config.py`. Four top-level blocks: `models`, `run`, `slo_profiles`, `evaluation`.

## `$ENV:` interpolation

Two token forms are resolved against the process environment at load time:
- whole-value `$ENV:NAME` (the entire string is the variable), and
- inline `${NAME}` (one or more tokens embedded in a larger string).

A referenced variable that is unset aborts the load with `environment variable not set: NAME` (CLI exit code 2). Resolution is recursive across nested mappings and lists.

**Secrets are never persisted in clear.** Any field named `api_key` is redacted to `***` in every snapshot (`resolved_config.json`). The literal secret never reaches disk, logs, or traces.

## Schema

### `models` (registry, required)

One entry per benchmarkable model. `extra` keys are forbidden.

| Field | Type | Notes |
|-------|------|-------|
| `name` | str | Selected via `--model`; first entry is the default. |
| `base_url` | str | Endpoint base, e.g. `http://localhost:8000/v1`. |
| `model` | str | Model id sent in the request body. |
| `api_key` | str? | Optional; use `$ENV:...`. Sent as a Bearer token when set. |
| `tokenizer` | str? | Optional tokenizer id. |
| `supports_vision` | bool | Capability gate; vision prompts are skipped when false. |
| `supports_tools` | bool | Capability gate; tool prompts are skipped when false. |
| `price_input` | float? | USD per 1M input tokens (enables cost). |
| `price_output` | float? | USD per 1M output tokens (enables cost). |

### `run`

`extra` keys are **allowed** (so tuning keys load even if not first-class fields).

| Field | Default | Notes |
|-------|---------|-------|
| `mode` | `closed` | `closed` (concurrency) or `open` (arrival rate). |
| `duration` | `2s` | Per-level/per-rate hold time. Durations accept `ms`/`s`/`m`/`h`. |
| `warmup` | `0.5s` | Leading phase, excluded from metrics. |
| `cooldown` | `0.5s` | Trailing phase, excluded from metrics. `warmup + cooldown` must not exceed `duration`. |
| `min_samples` | `30` | Warn when a level under-collects steady samples. |
| `concurrency_levels` | `[1]` | Closed-loop levels, run sequentially. |
| `request_rates` | `[]` | Open-loop arrival rates (req/s), run sequentially. |
| `burstiness` | `1.0` | Gamma shape for inter-arrivals; `1.0` is exponential (Poisson). |
| `max_outstanding` | `1000` | Open-loop in-flight guard; arrivals pause and a counter bumps when reached. |
| `max_tokens` | `8` | Requested output length (with `ignore_eos` for deterministic length). |
| `ignore_eos` | `true` | Force the SUT to emit `max_tokens` tokens. |
| `temperature` | `0.0` | Sampling temperature. |
| `cache_busting` | `true` | Prepend a unique prefix to the last user message; warns if a cache hit still occurs. |
| `retries` | `0` | Per-request retry budget. |
| `timeout` | `5s` | Per-request timeout. |
| `seed` | `0` | Master seed for reproducible prompt selection and arrival schedules. |
| `slo_profile` | `interactive` | Active profile name for goodput. |
| `eval_queue_maxsize` | (extra) | Bounded eval-queue size; default 10000 when omitted. |
| `event_loop_lag_threshold_ms` | (extra) | Client-saturation warning threshold; monitor off when omitted. |

### `slo_profiles`

A mapping of profile name to `{ttft_ms, tpot_ms, e2e_ms}` (extra threshold keys tolerated). When the block is omitted, the built-in defaults apply:
- `interactive`: `ttft_ms 500`, `tpot_ms 50`, `e2e_ms 5000`
- `relaxed`: `ttft_ms 2000`, `tpot_ms 200`, `e2e_ms 30000`

`run.slo_profile` selects the active one. `--slo key=value` overrides a threshold in place; the override is reflected in `resolved_config.json` (under the `slo` block) and in goodput.

### `evaluation`

| Field | Notes |
|-------|-------|
| `method` | `none` / `embedding` / `judge`. `--eval-method` overrides it (requires an `evaluation` block present). |
| `global_timeout` | Duration string bounding the post-run eval drain. |
| `embedding.url` | Embeddings endpoint; omit for a local endpoint. |
| `embedding.model` | Embedding model id. |
| `embedding.threshold` | **Mandatory** when the embedding method is active; a missing threshold aborts at config stage (exit 1). |
| `embedding.rate_limit` | Eval calls/s to the embedding provider (separate from SUT concurrency). |
| `judge.rubric` | `binary` (pass/fail) or `three_level` (correct/partial/incorrect); never a numeric score. |
| `judge.model.{url, api_key, model, prompt}` | Judge coordinates; default model `claude-haiku-4-5` via `IBM_ICA_BASE_URL` / `IBM_ICA_API_KEY`. `prompt` overrides the grading instruction. |

A judge model family matching the SUT family triggers a self-preference-bias warning (the run still proceeds).

## Annotated example

```yaml
models:
  - name: local-vllm
    base_url: http://localhost:8000/v1
    model: meta-llama/Llama-3.1-8B-Instruct
    api_key: $ENV:SUT_API_KEY        # resolved at load; redacted in snapshots
    supports_tools: true             # tool-use prompts are kept
    supports_vision: false           # vision prompts are skipped with a warning
    price_input: 0.20                # enables per-request and run cost
    price_output: 0.60

run:
  mode: open                         # honest tail latency under load
  duration: 30s
  warmup: 5s
  cooldown: 5s
  request_rates: [5, 20, 50]         # req/s, run sequentially
  burstiness: 1.0                    # Poisson arrivals
  max_outstanding: 500               # pause arrivals beyond 500 in flight
  max_tokens: 128
  timeout: 30s
  seed: 42
  slo_profile: interactive
  event_loop_lag_threshold_ms: 50    # warn on client-side saturation

evaluation:
  method: judge
  global_timeout: 60s
  judge:
    rubric: three_level
    model:
      url: $ENV:IBM_ICA_BASE_URL
      api_key: $ENV:IBM_ICA_API_KEY
      model: claude-haiku-4-5
```
