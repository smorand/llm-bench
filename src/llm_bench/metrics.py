"""Per-request metrics, token accounting, cost, and percentile aggregation.

This module layers the SC-001 METRICS / TOKEN-ACCOUNTING / COST contexts on top
of the closed-loop engine (FR-018..032, FR-058). It is intentionally pure and
side-effect free: the engine feeds it monotonic arrival offsets and the parsed
``usage`` object, and it returns derived latencies, an inter-token-latency (ITL)
summary, normalized latency, optional per-request cost, and the per-level
percentile / throughput objects rendered into ``summary.json``.

All percentiles use :func:`numpy.percentile` with linear interpolation so the
acceptance suite can recompute them from ``raw.jsonl`` and match within its ±2 %
tolerance. No prompts, responses, or secrets are handled here (FR-057).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from llm_bench.config import ModelRegistryEntry
    from llm_bench.runner import RequestRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTCOME_SUCCESS = "success"

# Threshold above which the p99.9 percentile is emitted per level (FR-024).
_P999_SAMPLE_THRESHOLD = 1000

# Per-million-token pricing divisor for cost computation (FR-031).
_PRICE_PER_MILLION = 1_000_000.0

# Input/output length bucket thresholds (token counts).
_ISL_MEDIUM = 256
_ISL_LONG = 1024
_OSL_MEDIUM = 32
_OSL_LONG = 256


# ---------------------------------------------------------------------------
# Per-request derived metrics
# ---------------------------------------------------------------------------


def itl_from_arrivals(arrival_times: Sequence[float]) -> list[float]:
    """Inter-token latencies from content arrival offsets, excluding TTFT (FR-020)."""
    return [arrival_times[i] - arrival_times[i - 1] for i in range(1, len(arrival_times))]


def itl_summary(itl_list: Sequence[float]) -> dict[str, float] | None:
    """Summarize an ITL list as ``{mean,p50,p95,p99,max}`` (FR-020).

    Returns ``None`` when the list is empty (a single-token response has no
    inter-token gaps), so a record carries no spurious zero-valued summary.
    """
    if not itl_list:
        return None
    arr = np.asarray(itl_list, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def tpot(ttft: float | None, e2e: float | None, output_tokens: int) -> float | None:
    """Per-output-token latency ``(e2e - ttft)/(output_tokens - 1)`` (FR-019).

    Null for single-token (or empty) responses so there is no division guard.
    """
    if ttft is None or e2e is None or output_tokens <= 1:
        return None
    return (e2e - ttft) / (output_tokens - 1)


def normalized_latency(e2e: float | None, output_tokens: int) -> float | None:
    """End-to-end latency per output token ``e2e / output_tokens`` (FR-021)."""
    if e2e is None or output_tokens <= 0:
        return None
    return e2e / output_tokens


def cost_usd(
    entry: ModelRegistryEntry,
    prompt_tokens: int,
    output_tokens: int,
) -> float | None:
    """Per-request cost from per-million pricing, or ``None`` when absent (FR-031/032)."""
    if entry.price_input is None or entry.price_output is None:
        return None
    return (prompt_tokens * entry.price_input + output_tokens * entry.price_output) / _PRICE_PER_MILLION


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two embedding vectors (FR-043).

    Returns ``0.0`` when either vector is the zero vector (no defined direction)
    so a degenerate embedding never raises and simply fails any positive
    threshold. The two vectors must share the same dimensionality.

    Args:
        a: First embedding vector.
        b: Second embedding vector.
    """
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def isl_bucket(prompt_tokens: int) -> str:
    """Bucket an input length into ``short``/``medium``/``long`` (FR-037)."""
    if prompt_tokens < _ISL_MEDIUM:
        return "short"
    if prompt_tokens < _ISL_LONG:
        return "medium"
    return "long"


def osl_bucket(output_tokens: int) -> str:
    """Bucket an output length into ``short``/``medium``/``long`` (FR-037)."""
    if output_tokens < _OSL_MEDIUM:
        return "short"
    if output_tokens < _OSL_LONG:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Per-level aggregation
# ---------------------------------------------------------------------------


def _percentile_object(values: Sequence[float]) -> dict[str, float] | None:
    """Build a ``{mean,min,max,std,p50,p90,p95,p99[,p999]}`` object (FR-023/024).

    The ``p999`` key is added only when the sample count reaches the documented
    threshold (FR-024); below it the key is omitted, never null.
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    obj: dict[str, float] = {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    if arr.size >= _P999_SAMPLE_THRESHOLD:
        obj["p999"] = float(np.percentile(arr, 99.9))
    return obj


def _steady_window_seconds(records: Sequence[RequestRecord]) -> float:
    """Wall time of the steady window from record start/end offsets (FR-058)."""
    if not records:
        return 0.0
    starts = [r.t_start for r in records]
    ends = [r.t_start + (r.e2e or 0.0) for r in records]
    window = max(ends) - min(starts)
    return window if window > 0 else 0.0


def _collect(records: Sequence[RequestRecord], attr: str) -> list[float]:
    """Collect a non-null float metric from steady-success records."""
    out: list[float] = []
    for record in records:
        value = getattr(record, attr)
        if value is not None:
            out.append(float(value))
    return out


def level_metrics(
    steady_success: Sequence[RequestRecord],
    *,
    entry: ModelRegistryEntry,
    cache_busting: bool = False,
) -> dict[str, Any]:
    """Build the per-level metric block over steady success records (FR-018..032).

    Args:
        steady_success: Steady-phase records with ``outcome == "success"``.
        entry: The model entry, consulted for pricing presence (FR-031/032).
        cache_busting: Whether cache busting is enabled; a cache-bias violation
            is only meaningful when it is (FR-028).

    Returns:
        A mapping of percentile objects, throughput rates, accounting counts,
        and (when pricing is defined) cost aggregates for one concurrency level.
    """
    window = _steady_window_seconds(steady_success)
    cache_violations = sum(1 for r in steady_success if r.cached_tokens > 0) if cache_busting else 0
    block: dict[str, Any] = {
        "latency_sample_count": len(steady_success),
        "steady_window_s": window,
        "steady_window": window,
        "ttft": _percentile_object(_collect(steady_success, "ttft")),
        "tpot": _percentile_object(_collect(steady_success, "tpot")),
        "e2e": _percentile_object(_collect(steady_success, "e2e")),
        "itl": _percentile_object(_itl_pool(steady_success)),
        "input_tokens": _percentile_object(_collect(steady_success, "prompt_tokens")),
        "output_tokens": _percentile_object(_collect(steady_success, "output_tokens")),
        "usage_incomplete_count": sum(1 for r in steady_success if r.usage_incomplete),
        "cache_busting_violations": cache_violations,
    }
    block.update(_throughput(steady_success, window))
    _add_cost(block, steady_success, entry)
    return block


def _itl_pool(records: Sequence[RequestRecord]) -> list[float]:
    """Pool every inter-token gap across steady-success records (token-weighted)."""
    pool: list[float] = []
    for record in records:
        if record.itl_list:
            pool.extend(record.itl_list)
    return pool


def _throughput(records: Sequence[RequestRecord], window: float) -> dict[str, float]:
    """Per-user / system / total tok/s and RPS over the steady window (FR-022)."""
    per_user = _per_user_tok_s(records)
    sum_output = sum(r.output_tokens for r in records)
    sum_total = sum(r.output_tokens + r.prompt_tokens for r in records)
    completed = len(records)
    if window <= 0:
        return {
            "per_user_tok_s": per_user,
            "system_tok_s": 0.0,
            "total_tok_s": 0.0,
            "rps": 0.0,
            "completed": completed,
        }
    return {
        "per_user_tok_s": per_user,
        "system_tok_s": sum_output / window,
        "total_tok_s": sum_total / window,
        "rps": completed / window,
        "completed": completed,
    }


def _per_user_tok_s(records: Sequence[RequestRecord]) -> float:
    """Mean per-request output tokens per second over generation time (FR-022)."""
    rates: list[float] = []
    for record in records:
        if record.ttft is None or record.e2e is None:
            continue
        generation = record.e2e - record.ttft
        if generation > 0 and record.output_tokens > 0:
            rates.append(record.output_tokens / generation)
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def _meets_slo(record: RequestRecord, slo: Mapping[str, float | None]) -> bool:
    """Return whether one record meets every active SLO threshold (FR-029).

    Record latencies are stored in seconds; SLO thresholds are in milliseconds.
    A ``None`` threshold is treated as "no bound" for that dimension. A latency
    that is missing (``None``) on a bounded dimension fails the threshold.
    """
    checks = (("ttft_ms", record.ttft), ("tpot_ms", record.tpot), ("e2e_ms", record.e2e))
    for key, value in checks:
        threshold = slo.get(key)
        if threshold is None:
            continue
        if value is None or value * 1000.0 > threshold:
            return False
    return True


def goodput(
    steady_success: Sequence[RequestRecord],
    slo: Mapping[str, float | None],
    window: float,
) -> dict[str, Any]:
    """Compute goodput count, attainment, and rate against the active SLO (FR-029).

    Args:
        steady_success: Steady-phase records with ``outcome == "success"``.
        slo: Active SLO thresholds (``ttft_ms``/``tpot_ms``/``e2e_ms``, in ms).
        window: Steady-window wall time in seconds (the goodput rate divisor).

    Returns:
        ``goodput_count`` (requests meeting all thresholds), ``goodput_attainment``
        (that count over the steady-success total, ``0.0`` when none), and
        ``goodput_rps`` (count over the steady window, ``0.0`` when the window is
        non-positive).
    """
    count = sum(1 for record in steady_success if _meets_slo(record, slo))
    total = len(steady_success)
    attainment = count / total if total else 0.0
    rps = count / window if window > 0 else 0.0
    return {
        "goodput_count": count,
        "goodput_attainment": attainment,
        "goodput_rps": rps,
    }


def _add_cost(
    block: dict[str, Any],
    records: Sequence[RequestRecord],
    entry: ModelRegistryEntry,
) -> None:
    """Fold per-level cost aggregates into ``block`` when pricing is defined (FR-031)."""
    if entry.price_input is None or entry.price_output is None:
        return
    costs = [cost_usd(entry, r.prompt_tokens, r.output_tokens) for r in records]
    total = sum(c for c in costs if c is not None)
    count = len(records)
    block["total_cost_usd"] = total
    block["cost_per_1k_requests"] = (total / count * 1000.0) if count else 0.0
