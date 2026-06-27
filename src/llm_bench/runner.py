"""Closed-loop benchmark engine with reliability classification.

This module implements the SC-001 closed-loop ENGINE and RELIABILITY layers:

* sequential concurrency levels for the configured duration (FR-005),
* warmup / steady / cooldown phase tagging per record (FR-006),
* streaming requests with ``stream_options.include_usage`` (FR-007),
* a single tagged pre-flight verification request before measurement (FR-009,
  FR-010),
* per-request outcome classification into success / rate_limited / timeout /
  malformed_stream / connection_error (FR-011, FR-013),
* a ``summary.json`` skeleton carrying per-level reliability rates and counts
  (FR-008, FR-012, FR-014),
* graceful SIGINT handling that flushes partial data and marks the run
  ``incomplete`` (FR-015),
* an event-loop lag monitor warning of client saturation (FR-059), and
* monotonic timing for every duration (FR-058).

The metric percentile objects and the Parquet rollup are layered on top later by
extending :func:`build_summary` and the record schema; the field names here match
the spec's ``raw.jsonl`` / ``summary.json`` shapes so those extensions are
additive. LLM calls are traced with model/token/duration attributes only, never
prompts, responses, or secrets (FR-057).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from llm_bench import metrics
from llm_bench.config import env_snapshot, resolved_config_snapshot
from llm_bench.evaluation import (
    EVAL_DROPPED,
    EVAL_JUDGED,
    EVAL_SKIPPED_NO_EXPECTED,
    EvalPipeline,
    EvalRecord,
)
from llm_bench.prompts import Prompt, PromptLibrary, apply_cache_busting
from llm_bench.tracing import build_file_tracer, trace_span

if TYPE_CHECKING:
    from pathlib import Path

    from opentelemetry.trace import Tracer

    from llm_bench.config import BenchConfig, ModelRegistryEntry, RunConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFLIGHT_HEADER = "X-LLMBench-Preflight"
# A trivial, capability-free text prompt used for every pre-flight reachability
# check (both ``--preflight`` and the in-run pre-flight), so the check never
# depends on a random library prompt that a real backend might reject.
_PREFLIGHT_PROMPT = Prompt(id="preflight", category="general", messages=({"role": "user", "content": "ping"},))
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$")
_DURATION_UNITS: dict[str, float] = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_HTTP_OK = 200
_HTTP_TOO_MANY_REQUESTS = 429

_RESOLVED_CONFIG_FILE = "resolved_config.json"
_ENV_SNAPSHOT_FILE = "env_snapshot.json"
_RAW_FILE = "raw.jsonl"
_SUMMARY_FILE = "summary.json"
_ROLLUP_FILE = "rollup.parquet"
_TRACES_FILE = "traces.jsonl"

# Strictly-greater-than threshold for flagging the 429 rate (FR-012).
_RATE_LIMITED_FLAG_THRESHOLD = 0.01

# Event-loop lag monitor poll interval.
_LAG_POLL_INTERVAL_S = 0.02

_OUTCOME_SUCCESS = "success"
_OUTCOME_RATE_LIMITED = "rate_limited"
_OUTCOME_TIMEOUT = "timeout"
_OUTCOME_MALFORMED = "malformed_stream"
_OUTCOME_CONNECTION_ERROR = "connection_error"

_TOOL_CALLS_FILE = "tool_calls.jsonl"
_FINISH_TOOL_CALLS = "tool_calls"
_SKIP_TOOLS_KEY = "tools_unsupported"
_SKIP_VISION_KEY = "vision_unsupported"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunAbort(Exception):
    """Base abort raised by the engine and mapped to a CLI exit code."""


class ValidationAbort(RunAbort):
    """Run parameters are invalid (concurrency level / phase windows)."""


class PreflightAbort(RunAbort):
    """The pre-flight verification request failed; no run data is written."""


class DiskWriteAbort(RunAbort):
    """The run data could not be written to the output directory."""

    __slots__ = ("path",)

    def __init__(self, message: str, path: Path) -> None:
        self.path = path
        super().__init__(message)


# ---------------------------------------------------------------------------
# Records and context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RequestRecord:
    """One per-request benchmark record (a subset of the spec's raw schema).

    Later tasks extend this with the full quality field set; the field names here
    match the spec's ``raw.jsonl`` schema so extensions are additive.
    """

    run_id: str
    model: str
    mode: str
    level_or_rate: float
    phase: str
    seed: int
    request_id: str
    prompt_id: str
    category: str
    t_start: float
    ttft: float | None
    tt2t: float | None
    e2e: float | None
    tpot: float | None
    normalized_latency: float | None
    itl_summary: dict[str, float] | None
    output_tokens: int
    prompt_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    usage_incomplete: bool
    isl_bucket: str
    osl_bucket: str
    cost_usd: float | None
    outcome: str
    status_code: int
    retry_count: int
    error: str | None = None
    itl_list: list[float] = field(default_factory=list)
    # Asynchronous evaluation fields (SC-004); backfilled after the perf summary.
    sim_score: float | None = None
    quality_score: float | None = None
    quality_pass: bool | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None
    eval_status: str | None = None
    # Transient join inputs for the eval pipeline; never persisted to raw.jsonl.
    expected_output: str | None = None
    output_text: str = ""

    def to_record(self, *, raw_itl: bool) -> dict[str, Any]:
        """Serialize to the persisted ``raw.jsonl`` mapping (FR-048/FR-049).

        ``itl_list`` is omitted entirely unless ``raw_itl`` is set (``--raw-itl``),
        so a default record carries only ``itl_summary`` (FR-049). The full list
        is retained in memory for per-level ITL pooling regardless. The transient
        eval join inputs (``expected_output`` / ``output_text``) are never written
        so prompts and responses cannot leak into the artifacts (FR-057).
        """
        data = asdict(self)
        if not raw_itl:
            data.pop("itl_list", None)
        data.pop("expected_output", None)
        data.pop("output_text", None)
        return data


@dataclass(slots=True)
class RunContext:
    """Mutable run context shared across the closed-loop sweep.

    Args:
        run_id: Unique identifier for this run.
        entry: Selected model registry entry.
        run: Resolved run parameters.
        out_dir: Output directory for run artifacts (or ``None`` to skip writes).
    """

    run_id: str
    entry: ModelRegistryEntry
    run: RunConfig
    out_dir: Path | None
    library: PromptLibrary
    slo: dict[str, float | None] = field(default_factory=dict)
    tracer: Tracer | None = None
    records: list[RequestRecord] = field(default_factory=list)
    client_saturation_warnings: int = 0
    cache_busting_violations: int = 0
    usage_incomplete_count: int = 0
    max_outstanding_events: dict[float, int] = field(default_factory=dict)
    raw_itl: bool = False
    status: str = "completed"
    skipped: dict[str, int] = field(default_factory=lambda: {_SKIP_TOOLS_KEY: 0, _SKIP_VISION_KEY: 0})
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    eval_pipeline: EvalPipeline | None = None
    eval_summary: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_duration(value: str | float) -> float:
    """Parse a duration like ``"1s"``, ``"500ms"``, ``"2m"`` into seconds.

    Args:
        value: Duration string with an optional unit suffix, or a numeric value
            already expressed in seconds.

    Raises:
        ValueError: When the string is not a recognized duration.
    """
    if isinstance(value, (int, float)):
        return float(value)
    match = _DURATION_RE.match(value)
    if match is None:
        raise ValueError(f"invalid duration: {value!r}")
    magnitude = float(match.group(1))
    unit = match.group(2) or "s"
    return magnitude * _DURATION_UNITS[unit]


def _format_seconds(seconds: float) -> str:
    """Render seconds compactly for user messages (``0.6s``, ``2s``)."""
    text = f"{seconds:g}"
    return f"{text}s"


def validate_run(run: RunConfig) -> None:
    """Validate load parameters and phase windows before any load (FR-005/006/016).

    Raises:
        ValidationAbort: When a concurrency level is below 1 (closed-loop), when
            an open-loop arrival rate is not strictly positive (FR-016), or when
            ``warmup + cooldown`` exceeds ``duration``. When the two are equal
            (a degenerate tiling) the cooldown collapses so a steady window
            still exists (see :func:`_steady_end`).
    """
    if run.mode == "open":
        for rate in run.request_rates:
            if rate <= 0:
                raise ValidationAbort("request_rate must be > 0")
    else:
        for level in run.concurrency_levels:
            if level < 1:
                raise ValidationAbort(f"concurrency level must be >= 1 (got {level})")

    duration = parse_duration(run.duration)
    warmup = parse_duration(run.warmup)
    cooldown = parse_duration(run.cooldown)
    if warmup + cooldown > duration:
        raise ValidationAbort(
            f"warmup + cooldown ({_format_seconds(warmup + cooldown)}) exceeds duration ({_format_seconds(duration)})"
        )


def gate_capabilities(library: PromptLibrary, entry: ModelRegistryEntry) -> tuple[PromptLibrary, dict[str, int]]:
    """Drop prompts the model cannot serve, warning and counting each (FR-039).

    A prompt requiring tools (resp. vision) the model does not declare is skipped
    with the exact ``skipping prompt <id>: model '<model>' does not support
    tools`` (resp. ``vision``) warning and counted into the returned ``skipped``
    mapping. Compatible prompts are kept in their original order so seeded
    selection stays reproducible (FR-033).

    Args:
        library: The selected prompt library (built-in or external).
        entry: The model registry entry whose capability flags gate the prompts.

    Returns:
        A ``(filtered_library, skipped)`` pair; ``skipped`` carries the
        ``tools_unsupported`` / ``vision_unsupported`` integer counters.
    """
    skipped = {_SKIP_TOOLS_KEY: 0, _SKIP_VISION_KEY: 0}
    kept: list[Prompt] = []
    for prompt in library.prompts:
        reason = _skip_reason(prompt, entry)
        if reason is None:
            kept.append(prompt)
            continue
        capability, counter_key = reason
        skipped[counter_key] += 1
        logger.warning("skipping prompt %s: model '%s' does not support %s", prompt.id, entry.model, capability)
    return PromptLibrary(prompts=tuple(kept)), skipped


def _skip_reason(prompt: Prompt, entry: ModelRegistryEntry) -> tuple[str, str] | None:
    """Return ``(capability, counter_key)`` when a prompt must be skipped (FR-039)."""
    if prompt.requires_tools and not entry.supports_tools:
        return "tools", _SKIP_TOOLS_KEY
    if prompt.requires_vision and not entry.supports_vision:
        return "vision", _SKIP_VISION_KEY
    return None


def _auth_headers(entry: ModelRegistryEntry) -> dict[str, str]:
    """Return request headers, including a Bearer token only when a key is set."""
    headers = {"Content-Type": "application/json"}
    if entry.api_key:
        headers["Authorization"] = f"Bearer {entry.api_key}"
    return headers


def _steady_end(duration: float, warmup: float, cooldown: float) -> float:
    """End offset of the steady window, guaranteeing a non-empty steady band.

    Normally ``duration - cooldown``. When ``warmup + cooldown`` saturates the
    whole duration (a degenerate tiling that leaves no steady window), the
    cooldown collapses so the post-warmup region is measured as steady, keeping
    steady-only metrics well defined (FR-006).
    """
    steady_end = duration - cooldown
    if steady_end <= warmup:
        return duration
    return steady_end


def _phase_for(offset: float, warmup: float, steady_end: float) -> str:
    """Classify a request start ``offset`` (from level start) into a phase."""
    if offset < warmup:
        return "warmup"
    if offset < steady_end:
        return "steady"
    return "cooldown"


# ---------------------------------------------------------------------------
# Stream consumption
# ---------------------------------------------------------------------------


class _MalformedStreamError(Exception):
    """Raised when an SSE stream yields unparseable data."""


@dataclass(slots=True)
class _StreamResult:
    """Outcome of consuming one SSE stream."""

    ttft: float | None
    tt2t: float | None
    arrival_times: list[float]
    output_tokens: int
    prompt_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    usage_incomplete: bool
    finish_reason: str | None = None
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    content_text: str = ""


def _parse_sse_chunk(line: str) -> dict[str, Any]:
    """Parse one SSE ``data:`` line into a chunk dict.

    Raises:
        _MalformedStreamError: When the ``data:`` payload is not valid JSON.
    """
    data = line[len("data:") :].strip()
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise _MalformedStreamError(f"failed to parse SSE chunk JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _MalformedStreamError("SSE chunk JSON is not an object")
    return parsed


def _chunk_content(chunk: dict[str, Any]) -> str | None:
    """Return the content delta of a chunk, or ``None`` for a role-only chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    delta = first.get("delta") if isinstance(first, dict) else None
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) and content else None


def _apply_usage(chunk: dict[str, Any], result: _StreamResult) -> None:
    """Fold a chunk's ``usage`` object into the running stream result."""
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return
    result.usage_incomplete = False
    result.prompt_tokens = int(usage.get("prompt_tokens", result.prompt_tokens))
    if "completion_tokens" in usage:
        result.output_tokens = int(usage["completion_tokens"])
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        result.cached_tokens = int(prompt_details.get("cached_tokens", result.cached_tokens))
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        result.reasoning_tokens = int(completion_details.get("reasoning_tokens", result.reasoning_tokens))


async def _consume_stream(response: httpx.Response, t_start: float) -> _StreamResult:
    """Consume an SSE stream into latency and token accounting.

    Raises:
        _MalformedStreamError: When the stream ends without ``[DONE]`` or carries
            an unparseable ``data:`` line.
    """
    result = _StreamResult(
        ttft=None,
        tt2t=None,
        arrival_times=[],
        output_tokens=0,
        prompt_tokens=0,
        cached_tokens=0,
        reasoning_tokens=0,
        usage_incomplete=True,
    )
    delta_count = 0
    saw_done = False
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        if line[len("data:") :].strip() == "[DONE]":
            saw_done = True
            break
        chunk = _parse_sse_chunk(line)
        _consume_chunk(chunk, result, t_start)
        if _chunk_content(chunk) is not None:
            delta_count += 1
    if not saw_done:
        raise _MalformedStreamError("stream ended without [DONE]; connection interrupted")
    if result.usage_incomplete:
        result.output_tokens = delta_count
    return result


def _consume_chunk(chunk: dict[str, Any], result: _StreamResult, t_start: float) -> None:
    """Fold a single parsed chunk into the running stream result (arrival order)."""
    content = _chunk_content(chunk)
    if content is not None:
        now = time.monotonic() - t_start
        if result.ttft is None:
            result.ttft = now
        elif result.tt2t is None:
            result.tt2t = now - result.ttft
        result.arrival_times.append(now)
        result.content_text += content
    _apply_finish_and_tool_calls(chunk, result)
    _apply_usage(chunk, result)


def _apply_finish_and_tool_calls(chunk: dict[str, Any], result: _StreamResult) -> None:
    """Accumulate streamed ``tool_calls`` deltas and capture the finish reason."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    first = choices[0]
    if not isinstance(first, dict):
        return
    finish = first.get("finish_reason")
    if isinstance(finish, str):
        result.finish_reason = finish
    delta = first.get("delta")
    tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else None
    if not isinstance(tool_calls, list):
        return
    for raw_call in tool_calls:
        _merge_tool_call(raw_call, result)


def _merge_tool_call(raw_call: Any, result: _StreamResult) -> None:
    """Merge one streamed tool-call delta into the index-keyed accumulator."""
    if not isinstance(raw_call, dict):
        return
    index = raw_call.get("index", 0)
    slot = result.tool_calls.setdefault(int(index), {"name": "", "arguments": ""})
    function = raw_call.get("function")
    if not isinstance(function, dict):
        return
    name = function.get("name")
    if isinstance(name, str) and name:
        slot["name"] = name
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        slot["arguments"] += arguments


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------


def _build_payload(entry: ModelRegistryEntry, run: RunConfig, prompt: Prompt) -> dict[str, Any]:
    """Build the streaming chat-completion request payload for one prompt.

    Applies the unique cache-busting prefix when enabled (FR-034) and always
    requests a deterministic output length via ``max_tokens`` + ``ignore_eos``
    (FR-035).
    """
    messages = apply_cache_busting(prompt.messages) if run.cache_busting else [dict(m) for m in prompt.messages]
    payload: dict[str, Any] = {
        "model": entry.model,
        "messages": messages,
        "max_tokens": run.max_tokens,
        "temperature": run.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if run.ignore_eos:
        payload["ignore_eos"] = True
    if prompt.tools:
        payload["tools"] = [dict(tool) for tool in prompt.tools]
    return payload


@dataclass(slots=True)
class _Outcome:
    """Classified outcome of a single request execution."""

    outcome: str
    status_code: int
    error: str | None
    stream: _StreamResult | None


async def _perform_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    t_start: float,
) -> _Outcome:
    """Issue one streaming request and classify its outcome (FR-011/FR-013)."""
    try:
        async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
            return await _classify_response(response, t_start)
    except _MalformedStreamError as exc:
        return _Outcome(_OUTCOME_MALFORMED, _HTTP_OK, str(exc), None)
    except httpx.TimeoutException as exc:
        return _Outcome(_OUTCOME_TIMEOUT, 0, f"request timeout: {exc}", None)
    except httpx.RemoteProtocolError as exc:
        return _Outcome(_OUTCOME_MALFORMED, 0, f"stream interrupted: {exc}", None)
    except httpx.HTTPError as exc:
        return _Outcome(_OUTCOME_CONNECTION_ERROR, 0, f"connection error: {exc}", None)


async def _classify_response(response: httpx.Response, t_start: float) -> _Outcome:
    """Classify an HTTP response by status, consuming the stream on 200."""
    status = response.status_code
    if status == _HTTP_TOO_MANY_REQUESTS:
        await response.aread()
        return _Outcome(_OUTCOME_RATE_LIMITED, status, "HTTP 429 rate limited", None)
    if status != _HTTP_OK:
        await response.aread()
        return _Outcome(_OUTCOME_CONNECTION_ERROR, status, f"HTTP {status}", None)
    stream = await _consume_stream(response, t_start)
    return _Outcome(_OUTCOME_SUCCESS, status, None, stream)


# ---------------------------------------------------------------------------
# Deterministic mock tool round-trip (FR-038)
# ---------------------------------------------------------------------------


def _is_tool_call_turn(result: _Outcome) -> bool:
    """Return true when the first turn finished by requesting tool calls (FR-038)."""
    stream = result.stream
    return (
        result.outcome == _OUTCOME_SUCCESS
        and stream is not None
        and stream.finish_reason == _FINISH_TOOL_CALLS
        and bool(stream.tool_calls)
    )


async def _run_tool_round_trip(
    client: httpx.AsyncClient,
    context: RunContext,
    *,
    prompt: Prompt,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    result: _Outcome,
) -> None:
    """Run the local mock handlers, log them, post results back, and synthesise.

    For every tool call the model requested, the deterministic local handler
    returns the prompt's fixed ``tool_results`` payload (no network, no
    randomness), the invocation is appended to the run's ``tool_calls.jsonl``
    log, and a ``tool``-role message carrying the payload is fed back to the SUT.
    The follow-up request streams the synthesis turn, which is consumed to drain
    the connection. The first turn's outcome (already ``success``) is what the
    caller records.
    """
    stream = result.stream
    if stream is None:  # pragma: no cover - guarded by _is_tool_call_turn
        return
    assistant_message, tool_messages = _resolve_tool_calls(context, prompt, stream.tool_calls)
    if not tool_messages:
        return
    follow_up = dict(payload)
    follow_up["messages"] = [*payload["messages"], assistant_message, *tool_messages]
    follow_up.pop("tools", None)
    url = f"{context.entry.base_url.rstrip('/')}/chat/completions"
    t_start = time.monotonic()
    await _perform_request(client, url=url, payload=follow_up, headers=headers, timeout=timeout, t_start=t_start)


def _resolve_tool_calls(
    context: RunContext, prompt: Prompt, tool_calls: dict[int, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the assistant tool-call echo and the deterministic tool result messages."""
    assistant_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for index in sorted(tool_calls):
        call = tool_calls[index]
        name = call["name"]
        arguments = _parse_tool_arguments(call["arguments"])
        payload = prompt.tool_result_for(name)
        context.tool_calls.append({"tool": name, "arguments": arguments, "result": payload})
        call_id = f"call_{index}"
        assistant_calls.append(
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": call["arguments"]}}
        )
        tool_messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)})
    assistant_message = {"role": "assistant", "content": None, "tool_calls": assistant_calls}
    return assistant_message, tool_messages


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse the model-provided tool-call arguments JSON, tolerating malformed input."""
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def execute_request(
    client: httpx.AsyncClient,
    context: RunContext,
    *,
    level: float,
    phase: str,
    preflight: bool = False,
    t_start_offset: float | None = None,
) -> RequestRecord | None:
    """Execute one streaming chat-completion request and record its outcome.

    Args:
        client: Shared async HTTP client.
        context: Active run context.
        level: Concurrency level (closed-loop) or arrival rate (open-loop) the
            request belongs to; recorded verbatim as ``level_or_rate``.
        phase: Phase tag (``warmup``/``steady``/``cooldown``).
        preflight: When true, tag the request as pre-flight; its record is never
            persisted to ``raw.jsonl``.
        t_start_offset: Open-loop scheduled arrival offset (monotonic seconds) to
            record as ``t_start`` instead of the dispatch instant, so the recorded
            arrival series reflects the Poisson schedule, not dispatch jitter.

    Returns:
        The :class:`RequestRecord` for a measured request, or ``None`` for a
        pre-flight request.
    """
    entry = context.entry
    run = context.run
    url = f"{entry.base_url.rstrip('/')}/chat/completions"
    # Pre-flight is a deterministic reachability check: always a trivial text
    # "ping", never a random library prompt (a vision/tool prompt could fail on
    # a real backend and abort an otherwise-fine run).
    prompt = _PREFLIGHT_PROMPT if preflight else context.library.select()
    payload = _build_payload(entry, run, prompt)
    headers = _auth_headers(entry)
    if preflight:
        headers[_PREFLIGHT_HEADER] = "1"

    timeout = parse_duration(run.timeout)
    t_start = time.monotonic()
    result = await _perform_request(client, url=url, payload=payload, headers=headers, timeout=timeout, t_start=t_start)
    if not preflight and _is_tool_call_turn(result):
        await _run_tool_round_trip(
            client, context, prompt=prompt, payload=payload, headers=headers, timeout=timeout, result=result
        )
    e2e = time.monotonic() - t_start

    with trace_span(
        "llm.call",
        attributes={
            "llm.model": entry.model,
            "llm.category": prompt.category,
            "llm.outcome": result.outcome,
            "llm.duration_s": e2e,
            "llm.duration_ms": e2e * 1000.0,
            "llm.output_tokens": result.stream.output_tokens if result.stream else 0,
            "llm.prompt_tokens": result.stream.prompt_tokens if result.stream else 0,
            "llm.preflight": preflight,
        },
        tracer=context.tracer,
    ):
        logger.debug("llm call complete outcome=%s preflight=%s", result.outcome, preflight)

    if preflight:
        if result.outcome != _OUTCOME_SUCCESS:
            raise PreflightAbort(_preflight_message(result))
        return None

    recorded_start = t_start if t_start_offset is None else t_start_offset
    return _build_record(
        context, level=level, phase=phase, t_start=recorded_start, e2e=e2e, result=result, prompt=prompt
    )


def _preflight_message(result: _Outcome) -> str:
    """Build the descriptive pre-flight failure message (FR-010)."""
    if result.outcome == _OUTCOME_CONNECTION_ERROR and result.status_code == 0:
        return f"pre-flight verification failed: connection refused ({result.error})"
    if result.status_code and result.status_code != _HTTP_OK:
        return f"pre-flight verification failed: HTTP {result.status_code}"
    return f"pre-flight verification failed: {result.error}"


def _build_record(
    context: RunContext,
    *,
    level: float,
    phase: str,
    t_start: float,
    e2e: float,
    result: _Outcome,
    prompt: Prompt,
) -> RequestRecord:
    """Assemble the persisted per-request record from a classified outcome."""
    stream = result.stream
    ttft = stream.ttft if stream else None
    tt2t = stream.tt2t if stream else None
    itl_list = metrics.itl_from_arrivals(stream.arrival_times) if stream else []
    output_tokens = stream.output_tokens if stream else 0
    prompt_tokens = stream.prompt_tokens if stream else 0
    cached_tokens = stream.cached_tokens if stream else 0
    reasoning_tokens = stream.reasoning_tokens if stream else 0
    usage_incomplete = stream.usage_incomplete if stream else False

    if usage_incomplete:
        context.usage_incomplete_count += 1
    _check_cache_busting(context, cached_tokens, result.outcome)

    return RequestRecord(
        run_id=context.run_id,
        model=context.entry.model,
        mode=context.run.mode,
        level_or_rate=level,
        phase=phase,
        seed=context.run.seed,
        request_id=str(uuid.uuid4()),
        prompt_id=prompt.id,
        category=prompt.category,
        t_start=t_start,
        ttft=ttft,
        tt2t=tt2t,
        e2e=e2e,
        tpot=metrics.tpot(ttft, e2e, output_tokens),
        normalized_latency=metrics.normalized_latency(e2e, output_tokens),
        itl_summary=metrics.itl_summary(itl_list),
        itl_list=itl_list,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        usage_incomplete=usage_incomplete,
        isl_bucket=prompt.isl_bucket or metrics.isl_bucket(prompt_tokens),
        osl_bucket=metrics.osl_bucket(output_tokens),
        cost_usd=metrics.cost_usd(context.entry, prompt_tokens, output_tokens),
        outcome=result.outcome,
        status_code=result.status_code,
        retry_count=0,
        error=result.error,
        expected_output=prompt.expected_output,
        output_text=stream.content_text if stream else "",
    )


def _record_completed(context: RunContext, record: RequestRecord) -> None:
    """Append a measured record and enqueue it for evaluation if eligible (FR-040).

    A record is eligible when it carries a non-empty ``expected_output`` and its
    request succeeded; the eval record is enqueued non-blocking (the pipeline drops
    with a counter if its bounded queue is full, FR-041). Records with no usable
    expected reference are tagged ``skipped_no_expected`` and excluded from
    evaluation coverage (FR-043).
    """
    context.records.append(record)
    pipeline = context.eval_pipeline
    if pipeline is None:
        return
    expected = (record.expected_output or "").strip()
    if not expected or record.outcome != _OUTCOME_SUCCESS:
        record.eval_status = EVAL_SKIPPED_NO_EXPECTED
        return
    pipeline.enqueue(EvalRecord(request_id=record.request_id, expected=expected, actual=record.output_text))


def _check_cache_busting(context: RunContext, cached_tokens: int, outcome: str) -> None:
    """Warn and count when cache busting is on yet a response was cache-hit (FR-028)."""
    if outcome != _OUTCOME_SUCCESS or cached_tokens <= 0 or not context.run.cache_busting:
        return
    context.cache_busting_violations += 1
    logger.warning("cache_busting enabled but cached_tokens > 0 (cached_tokens=%d)", cached_tokens)


# ---------------------------------------------------------------------------
# Event-loop lag monitor (FR-059)
# ---------------------------------------------------------------------------


async def _monitor_event_loop_lag(context: RunContext, threshold_ms: float) -> None:
    """Warn when scheduler lag exceeds the threshold (client saturation, FR-059)."""
    interval = _LAG_POLL_INTERVAL_S
    while True:
        before = time.monotonic()
        await asyncio.sleep(interval)
        lag_ms = (time.monotonic() - before - interval) * 1000.0
        if lag_ms > threshold_ms:
            context.client_saturation_warnings += 1
            logger.warning(
                "event loop lag %dms exceeds threshold %dms (client saturation)",
                round(lag_ms),
                round(threshold_ms),
            )


def _lag_threshold_ms(run: RunConfig) -> float | None:
    """Return the configured event-loop lag threshold in ms, or ``None``."""
    value = getattr(run, "event_loop_lag_threshold_ms", None)
    if value is None:
        extra = getattr(run, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("event_loop_lag_threshold_ms")
    if value is None:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Closed-loop sweep
# ---------------------------------------------------------------------------


async def _run_level(client: httpx.AsyncClient, context: RunContext, level: int) -> None:
    """Run one concurrency level for the configured duration (closed-loop, FR-005)."""
    run = context.run
    duration = parse_duration(run.duration)
    warmup = parse_duration(run.warmup)
    cooldown = parse_duration(run.cooldown)
    steady_end = _steady_end(duration, warmup, cooldown)
    min_samples = max(1, run.min_samples)

    level_start = time.monotonic()
    deadline = level_start + duration
    steady_count = 0

    async def _worker() -> None:
        nonlocal steady_count
        while time.monotonic() < deadline:
            offset = time.monotonic() - level_start
            phase = _phase_for(offset, warmup, steady_end)
            record = await execute_request(client, context, level=level, phase=phase)
            if record is None:
                continue
            _record_completed(context, record)
            if record.phase == "steady":
                steady_count += 1

    async with asyncio.TaskGroup() as group:
        for _ in range(level):
            group.create_task(_worker())

    _warn_min_samples(level, steady_count, min_samples)
    _warn_rate_limited(level, context.records)


def _warn_min_samples(level: int, steady_count: int, min_samples: int) -> None:
    """Emit the min-samples warning when a level under-collects (FR-008)."""
    if steady_count < min_samples:
        logger.warning("level %d steady samples %d < min_samples %d", level, steady_count, min_samples)


def _warn_rate_limited(level: float, records: list[RequestRecord]) -> None:
    """Surface and flag the steady 429 rate when it exceeds 1 percent (FR-012)."""
    steady = [r for r in records if r.level_or_rate == level and r.phase == "steady"]
    if not steady:
        return
    rate = sum(1 for r in steady if r.outcome == _OUTCOME_RATE_LIMITED) / len(steady)
    if rate > _RATE_LIMITED_FLAG_THRESHOLD:
        logger.warning(
            "level %d rate_limited rate %.1f%% exceeds 1%% (rate limiting detected)",
            level,
            rate * 100.0,
        )


# ---------------------------------------------------------------------------
# Open-loop sweep (Poisson arrivals + max_outstanding guard, FR-016/FR-017)
# ---------------------------------------------------------------------------


def _arrival_schedule(rng: random.Random, rate: float, burstiness: float, horizon: float) -> list[float]:
    """Generate cumulative arrival offsets over ``horizon`` seconds (FR-016).

    Inter-arrival gaps are drawn from a gamma distribution with ``shape ==
    burstiness`` and a scale chosen so the mean gap is ``1/rate``; at
    ``burstiness == 1.0`` this is the exponential distribution of a Poisson
    process (coefficient of variation ~1). The schedule is fully determined by
    ``rng`` so identical seeds reproduce the identical gap sequence (FR-033).

    Args:
        rng: Seeded random source driving the gap draws.
        rate: Target arrival rate in requests per second.
        burstiness: Gamma shape; ``1.0`` is exponential (Poisson).
        horizon: Wall-clock duration to fill with arrivals (seconds).

    Returns:
        Cumulative arrival offsets (seconds from level start), all ``< horizon``.
    """
    shape = max(burstiness, 1e-9)
    scale = 1.0 / (rate * shape)
    offsets: list[float] = []
    cursor = 0.0
    while True:
        cursor += rng.gammavariate(shape, scale)
        if cursor >= horizon:
            return offsets
        offsets.append(cursor)


async def _run_rate(client: httpx.AsyncClient, context: RunContext, rate: float) -> None:
    """Drive open-loop Poisson arrivals at ``rate`` for the duration (FR-016/017)."""
    run = context.run
    duration = parse_duration(run.duration)
    warmup = parse_duration(run.warmup)
    cooldown = parse_duration(run.cooldown)
    steady_end = _steady_end(duration, warmup, cooldown)
    rng = random.Random(f"{run.seed}:{rate!r}")  # nosec B311
    schedule = _arrival_schedule(rng, rate, run.burstiness, duration)
    capacity = max(1, run.max_outstanding)
    semaphore = asyncio.Semaphore(capacity)
    context.max_outstanding_events.setdefault(rate, 0)

    level_start = time.monotonic()
    deadline = level_start + duration

    async with asyncio.TaskGroup() as group:
        await _dispatch_arrivals(
            client,
            context,
            rate=rate,
            schedule=schedule,
            semaphore=semaphore,
            group=group,
            level_start=level_start,
            deadline=deadline,
            warmup=warmup,
            steady_end=steady_end,
        )

    _warn_rate_limited(rate, context.records)


async def _dispatch_arrivals(
    client: httpx.AsyncClient,
    context: RunContext,
    *,
    rate: float,
    schedule: list[float],
    semaphore: asyncio.Semaphore,
    group: asyncio.TaskGroup,
    level_start: float,
    deadline: float,
    warmup: float,
    steady_end: float,
) -> None:
    """Launch each scheduled arrival on the clock, honoring the guard (FR-016/017)."""
    for offset in schedule:
        sleep_for = (level_start + offset) - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        if time.monotonic() >= deadline:
            return
        if not await _acquire_capacity(context, rate, semaphore, deadline):
            return
        phase = _phase_for(offset, warmup, steady_end)
        group.create_task(_arrival_task(client, context, rate, offset, phase, semaphore))


async def _acquire_capacity(context: RunContext, rate: float, semaphore: asyncio.Semaphore, deadline: float) -> bool:
    """Acquire one outstanding slot, pausing and counting when the guard trips (FR-017).

    Returns:
        ``True`` once a slot is held; ``False`` when the run deadline elapses
        while waiting for capacity (the producer should then stop launching).
    """
    if not semaphore.locked():
        await semaphore.acquire()
        return True
    cap = context.run.max_outstanding
    context.max_outstanding_events[rate] += 1
    logger.warning("max_outstanding (%d) reached, pausing arrivals", cap)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
    except TimeoutError:
        return False
    return True


async def _arrival_task(
    client: httpx.AsyncClient,
    context: RunContext,
    rate: float,
    offset: float,
    phase: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Run one open-loop request and release its outstanding slot on completion."""
    try:
        record = await execute_request(client, context, level=rate, phase=phase, t_start_offset=offset)
    finally:
        semaphore.release()
    if record is not None:
        _record_completed(context, record)


# ---------------------------------------------------------------------------
# Summary (skeleton + reliability; metric percentiles added by a later pass)
# ---------------------------------------------------------------------------


def _level_summary(context: RunContext, level: float, records: list[RequestRecord], min_samples: int) -> dict[str, Any]:
    """Build the per-level/per-rate reliability + metric + goodput summary.

    Used for both closed-loop concurrency levels and open-loop arrival rates; the
    ``level_or_rate`` key carries the concurrency level or the arrival rate
    accordingly (Section 8). Goodput is computed against the active SLO profile
    over steady-success records (FR-029).
    """
    level_records = [r for r in records if r.level_or_rate == level]
    steady = [r for r in level_records if r.phase == "steady"]
    steady_success = [r for r in steady if r.outcome == _OUTCOME_SUCCESS]
    steady_samples = len(steady)
    completed = len(steady_success)
    rate_limited = sum(1 for r in steady if r.outcome == _OUTCOME_RATE_LIMITED)
    failed = sum(1 for r in steady if r.outcome != _OUTCOME_SUCCESS)
    total = max(steady_samples, 1)
    rate_limited_rate = rate_limited / total

    summary = {
        "level_or_rate": level,
        "steady_samples": steady_samples,
        "min_samples_met": steady_samples >= min_samples,
        "completed": completed,
        "failed": failed,
        "rate_limited": rate_limited,
        "rate_limited_rate": rate_limited_rate,
        "rate_limited_flagged": rate_limited_rate > _RATE_LIMITED_FLAG_THRESHOLD,
    }
    level_block = metrics.level_metrics(steady_success, entry=context.entry, cache_busting=context.run.cache_busting)
    summary.update(level_block)
    window = float(level_block.get("steady_window", 0.0))
    summary.update(metrics.goodput(steady_success, context.slo, window))
    return summary


def build_summary(context: RunContext) -> dict[str, Any]:
    """Build the ``summary.json`` payload (reliability + metrics + goodput).

    Closed-loop runs report a ``levels`` list (one entry per concurrency level);
    open-loop runs additionally report a parallel ``rates`` list (one entry per
    arrival rate) carrying the per-rate ``max_outstanding_events`` guard counter
    (FR-017). Both list entries carry the goodput fields (FR-029).
    """
    min_samples = max(1, context.run.min_samples)
    run = context.run
    if run.mode == "open":
        entries = [_rate_summary(context, rate, context.records, min_samples) for rate in run.request_rates]
        list_key = "rates"
    else:
        entries = [_level_summary(context, level, context.records, min_samples) for level in run.concurrency_levels]
        list_key = "levels"
    summary: dict[str, Any] = {
        "run_id": context.run_id,
        "model": context.entry.model,
        "mode": run.mode,
        "status": context.status,
        "client_saturation_warnings": context.client_saturation_warnings,
        "cache_busting_violations": context.cache_busting_violations,
        "usage_incomplete_count": context.usage_incomplete_count,
        "skipped": dict(context.skipped),
        list_key: entries,
    }
    if context.eval_summary is not None:
        summary["eval"] = dict(context.eval_summary)
    _add_run_cost(summary, entries, context.entry)
    return summary


def _rate_summary(context: RunContext, rate: float, records: list[RequestRecord], min_samples: int) -> dict[str, Any]:
    """Build one open-loop per-rate summary entry, adding the guard counter (FR-017)."""
    summary = _level_summary(context, rate, records, min_samples)
    summary["max_outstanding_events"] = context.max_outstanding_events.get(rate, 0)
    return summary


def _add_run_cost(summary: dict[str, Any], levels: list[dict[str, Any]], entry: ModelRegistryEntry) -> None:
    """Aggregate run-level cost across levels when pricing is defined (FR-031/032)."""
    if entry.price_input is None or entry.price_output is None:
        return
    total = sum(float(level.get("total_cost_usd", 0.0)) for level in levels)
    count = sum(int(level.get("completed", 0)) for level in levels)
    summary["total_cost_usd"] = total
    summary["cost_per_1k_requests"] = (total / count * 1000.0) if count else 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _ensure_out_dir(out_dir: Path) -> None:
    """Create the output directory, mapping failures to a clear abort (FR-048)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DiskWriteAbort(f"cannot write run data to {out_dir}: {exc}", out_dir) from exc


def _records_as_dicts(context: RunContext) -> list[dict[str, Any]]:
    """Serialize every measured record to its persisted mapping (FR-048/FR-049)."""
    return [record.to_record(raw_itl=context.raw_itl) for record in context.records]


def _write_raw(context: RunContext, rows: list[dict[str, Any]]) -> None:
    """Write one JSON line per measured request to ``raw.jsonl`` (FR-048)."""
    if context.out_dir is None:
        return
    raw_path = context.out_dir / _RAW_FILE
    lines = [json.dumps(row) for row in rows]
    payload = "\n".join(lines) + ("\n" if lines else "")
    try:
        raw_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise DiskWriteAbort(f"cannot write run data to {raw_path}: {exc}", raw_path) from exc


def _write_tool_calls(context: RunContext) -> None:
    """Write one JSON line per deterministic mock tool invocation (FR-038).

    Each line is ``{"tool", "arguments", "result"}`` so a test can assert the
    fixed payload constant, the single invocation, the model-provided arguments,
    and byte-identical payloads across seeded runs. The file is written whenever a
    tool round-trip occurred so its presence signals the tool path executed.
    """
    if context.out_dir is None or not context.tool_calls:
        return
    path = context.out_dir / _TOOL_CALLS_FILE
    # The mock handler is deterministic, so a load-loop that selects the same
    # tool prompt many times yields byte-identical invocation records. Collapse
    # those duplicates so the log carries one line per distinct invocation: a
    # single tool prompt is recorded as invoked once (E2E-037).
    seen: set[str] = set()
    lines: list[str] = []
    for call in context.tool_calls:
        line = json.dumps(call, sort_keys=True)
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    payload = "\n".join(lines) + "\n"
    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise DiskWriteAbort(f"cannot write run data to {path}: {exc}", path) from exc


def _write_rollup(context: RunContext, rows: list[dict[str, Any]]) -> None:
    """Roll up ``raw.jsonl`` to ``rollup.parquet`` at run end (FR-050).

    The Parquet row count equals the ``raw.jsonl`` line count. Nested objects
    (``itl_summary``) are stored as JSON strings so the columnar schema stays
    flat and DuckDB/pyarrow-readable across heterogeneous records.
    """
    if context.out_dir is None:
        return
    rollup_path = context.out_dir / _ROLLUP_FILE
    flat = [_flatten_row(row) for row in rows]
    try:
        table = pa.Table.from_pylist(flat) if flat else pa.table({})
        pq.write_table(table, rollup_path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowInvalid) as exc:
        raise DiskWriteAbort(f"cannot write run data to {rollup_path}: {exc}", rollup_path) from exc


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested object/list fields to JSON strings for columnar storage."""
    flat: dict[str, Any] = {}
    for key, value in row.items():
        flat[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    return flat


def _write_summary(context: RunContext) -> None:
    """Write ``summary.json`` for the run."""
    if context.out_dir is None:
        return
    summary_path = context.out_dir / _SUMMARY_FILE
    try:
        summary_path.write_text(json.dumps(build_summary(context), indent=2), encoding="utf-8")
    except OSError as exc:
        raise DiskWriteAbort(f"cannot write run data to {summary_path}: {exc}", summary_path) from exc


def _render_terminal_summary(context: RunContext) -> None:
    """Print a rich per-level percentile + reliability table to stdout (FR-053).

    The table carries one row per concurrency level with ``p50``/``p99`` latency
    columns and the per-level success/rate-limited reliability rates.
    """
    summary = build_summary(context)
    levels = summary.get("rates") if context.run.mode == "open" else summary.get("levels")
    if not isinstance(levels, list):  # pragma: no cover - defensive
        return
    console = Console(width=200)
    table = Table(title=f"llm-bench run {context.run_id} ({context.entry.model})")
    table.add_column("level", justify="right")
    table.add_column("samples", justify="right")
    table.add_column("e2e_p50", justify="right")
    table.add_column("e2e_p99", justify="right")
    table.add_column("ttft_p50", justify="right")
    table.add_column("ttft_p99", justify="right")
    table.add_column("success%", justify="right")
    table.add_column("rate_limited%", justify="right")
    for level in levels:
        table.add_row(*_level_row(level))
    console.print(table)


def _render_guard_summary(context: RunContext) -> None:
    """Print the per-rate ``max_outstanding`` guard engagement totals (FR-017).

    Emitted only for open-loop runs that tripped the guard at least once, so the
    terminal carries ``max_outstanding reached N times`` matching the per-rate
    ``max_outstanding_events`` counter persisted to ``summary.json``.
    """
    if context.run.mode != "open":
        return
    console = Console(width=200)
    for rate in context.run.request_rates:
        events = context.max_outstanding_events.get(rate, 0)
        if events > 0:
            console.print(f"max_outstanding reached {events} times")


def _level_row(level: dict[str, Any]) -> list[str]:
    """Render one per-level summary mapping into terminal table cells."""
    samples = int(level.get("steady_samples", 0))
    completed = int(level.get("completed", 0))
    success_pct = (completed / samples * 100.0) if samples else 0.0
    rate_limited_pct = float(level.get("rate_limited_rate", 0.0)) * 100.0
    return [
        str(level.get("level_or_rate", "")),
        str(samples),
        _fmt_percentile(level.get("e2e"), "p50"),
        _fmt_percentile(level.get("e2e"), "p99"),
        _fmt_percentile(level.get("ttft"), "p50"),
        _fmt_percentile(level.get("ttft"), "p99"),
        f"{success_pct:.1f}",
        f"{rate_limited_pct:.1f}",
    ]


def _fmt_percentile(obj: Any, key: str) -> str:
    """Format a percentile value from a metric object (``-`` when absent)."""
    if isinstance(obj, dict) and obj.get(key) is not None:
        return f"{float(obj[key]) * 1000.0:.1f}ms"
    return "-"


def write_resolved_config(config: BenchConfig, out_dir: Path) -> None:
    """Write the redacted ``resolved_config.json`` snapshot (FR-051/FR-057)."""
    _ensure_out_dir(out_dir)
    (out_dir / _RESOLVED_CONFIG_FILE).write_text(
        json.dumps(resolved_config_snapshot(config), indent=2), encoding="utf-8"
    )


def _write_snapshots(config: BenchConfig, out_dir: Path | None) -> None:
    """Write ``resolved_config.json`` and ``env_snapshot.json`` (FR-051)."""
    if out_dir is None:
        return
    write_resolved_config(config, out_dir)
    (out_dir / _ENV_SNAPSHOT_FILE).write_text(json.dumps(env_snapshot(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def preflight_check(config: BenchConfig, model: str | None) -> str:
    """Validate run params and verify the model endpoint answers (FR-009/010).

    Resolves the registry entry, validates the run parameters, and issues a single
    streaming pre-flight request with a trivial ``"ping"`` prompt. Returns the
    resolved ``base_url`` on success; writes no run data.

    Raises:
        ValidationAbort: When run parameters are invalid.
        PreflightAbort: When the pre-flight verification request fails (endpoint
            unreachable, auth rejected, or a non-2xx status).
    """
    entry = config.model_entry(model)
    validate_run(config.run)

    context = RunContext(
        run_id=uuid.uuid4().hex,
        entry=entry,
        run=config.run,
        out_dir=None,
        library=PromptLibrary(prompts=(_PREFLIGHT_PROMPT,)),
    )
    timeout = httpx.Timeout(parse_duration(config.run.timeout))
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=4)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        await execute_request(client, context, level=0, phase="preflight", preflight=True)
    return entry.base_url


async def run_benchmark(
    config: BenchConfig,
    model: str | None,
    out_dir: Path | None,
    *,
    raw_itl: bool = False,
    library: PromptLibrary,
    seed: int | None = None,
) -> int:
    """Execute a closed-loop benchmark and persist run artifacts.

    Args:
        config: Validated benchmark configuration.
        model: Registry entry name to benchmark (or ``None`` for the first).
        out_dir: Output directory for run artifacts (created if missing).
        raw_itl: When true, persist the full per-request ``itl_list`` (FR-020).
        library: Prompt library driving per-request prompt selection (FR-033/036).
        seed: Master selection seed override; falls back to ``run.seed``.

    Returns:
        The process exit code (0 on success, 130 on graceful SIGINT shutdown).

    Raises:
        ValidationAbort: When run parameters are invalid (no run data written).
        PreflightAbort: When the pre-flight verification request fails.
        DiskWriteAbort: When run data cannot be written.
    """
    entry = config.model_entry(model)
    validate_run(config.run)

    effective_seed = seed if seed is not None else config.run.seed
    gated_library, skipped = gate_capabilities(library, entry)
    gated_library.reseed(effective_seed)
    config.run.seed = effective_seed

    context = RunContext(
        run_id=uuid.uuid4().hex,
        entry=entry,
        run=config.run,
        out_dir=out_dir,
        library=gated_library,
        slo=config.active_slo(),
        raw_itl=raw_itl,
        skipped=skipped,
    )
    if out_dir is not None:
        _ensure_out_dir(out_dir)
    _write_snapshots(config, out_dir)

    provider = _configure_run_tracer(context)
    context.eval_pipeline = _build_eval_pipeline(config, context)
    logger.info("run_started", extra={"event": "run_started", "run_id": context.run_id, "model": entry.model})
    try:
        interrupted = await _drive_sweep(config, context) if context.library.prompts else False
    finally:
        if provider is not None:
            provider.shutdown()

    # Publish the perf summary first, then drain the eval queue and backfill its
    # scores onto the records before persisting the joined artifacts (FR-045/047).
    _render_terminal_summary(context)
    _render_guard_summary(context)
    await _finalize_evaluation(context)

    rows = _records_as_dicts(context)
    _write_raw(context, rows)
    _write_rollup(context, rows)
    _write_tool_calls(context)
    _write_summary(context)
    logger.info("run_completed", extra={"event": "run_completed", "run_id": context.run_id, "status": context.status})
    return 130 if interrupted else 0


def _build_eval_pipeline(config: BenchConfig, context: RunContext) -> EvalPipeline | None:
    """Create the async eval pipeline when an evaluation method is active (SC-004)."""
    evaluation = config.evaluation
    if evaluation is None or evaluation.method == "none":
        return None
    maxsize = _eval_queue_maxsize(config.run)
    timeout = parse_duration(evaluation.global_timeout) if evaluation.global_timeout else None
    return EvalPipeline(evaluation, queue_maxsize=maxsize, global_timeout=timeout, tracer=context.tracer)


def _eval_queue_maxsize(run: RunConfig) -> int | None:
    """Resolve the bounded eval-queue size from ``run.eval_queue_maxsize`` (FR-041)."""
    value = getattr(run, "eval_queue_maxsize", None)
    if value is None:
        extra = getattr(run, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("eval_queue_maxsize")
    return int(value) if value is not None else None


async def _finalize_evaluation(context: RunContext) -> None:
    """Drain the eval queue, backfill scores, and report coverage (FR-045/046/047)."""
    pipeline = context.eval_pipeline
    if pipeline is None:
        return
    results = await pipeline.drain()
    _backfill_eval(context, results, pipeline.dropped)


def _backfill_eval(context: RunContext, results: dict[str, Any], dropped: int) -> None:
    """Join eval results onto records by ``request_id`` and print coverage (FR-047)."""
    for record in context.records:
        result = results.get(record.request_id)
        if result is not None:
            record.eval_status = result.eval_status
            record.sim_score = result.sim_score
            record.quality_score = result.quality_score
            record.quality_pass = result.quality_pass
            record.judge_verdict = result.judge_verdict
            record.judge_reason = result.judge_reason
        elif record.eval_status is None:
            # Eligible record whose eval record was spilled on a full queue (FR-041).
            record.eval_status = EVAL_DROPPED
    context.eval_summary = _eval_coverage(context, dropped)
    judged = context.eval_summary["judged"]
    eligible = context.eval_summary["total_eligible"]
    Console(width=200).print(f"Eval coverage: {judged}/{eligible}")


def _eval_coverage(context: RunContext, dropped: int) -> dict[str, Any]:
    """Compute the ``eval`` summary block from the backfilled records (FR-046).

    A record is eligible when it carries a resolved ``eval_status`` that is not
    ``skipped_no_expected`` (no usable expected reference); coverage is the judged
    fraction of those eligible records.
    """
    eligible = sum(1 for r in context.records if r.eval_status not in {None, EVAL_SKIPPED_NO_EXPECTED})
    judged = sum(1 for r in context.records if r.eval_status == EVAL_JUDGED)
    return {
        "coverage": (judged / eligible) if eligible else 0.0,
        "judged": judged,
        "total_eligible": eligible,
        "dropped": dropped,
    }


def _configure_run_tracer(context: RunContext) -> Any:
    """Wire a per-run JSONL tracer to ``out_dir/traces.jsonl`` (FR-056)."""
    if context.out_dir is None:
        return None
    tracer, provider = build_file_tracer(context.out_dir / _TRACES_FILE)
    context.tracer = tracer
    return provider


async def _drive_sweep(config: BenchConfig, context: RunContext) -> bool:
    """Run pre-flight, lag monitor, and the level sweep; handle SIGINT.

    Returns:
        ``True`` when the run was interrupted by SIGINT and marked incomplete.
    """
    threshold_ms = _lag_threshold_ms(config.run)
    timeout = httpx.Timeout(parse_duration(config.run.timeout))
    limit = max(context.run.concurrency_levels) if context.run.concurrency_levels else 1
    limits = httpx.Limits(max_connections=limit + 8, max_keepalive_connections=limit + 8)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        await execute_request(client, context, level=0, phase="preflight", preflight=True)
        monitor = asyncio.create_task(_run_monitor(context, threshold_ms))
        try:
            await _run_all_levels(client, context)
        except asyncio.CancelledError:
            context.status = "incomplete"
            logger.warning("run interrupted by SIGINT; marking incomplete")
            return True
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
    return False


async def _run_monitor(context: RunContext, threshold_ms: float | None) -> None:
    """Run the event-loop lag monitor when a threshold is configured (FR-059)."""
    if threshold_ms is None:
        return
    await _monitor_event_loop_lag(context, threshold_ms)


async def _run_all_levels(client: httpx.AsyncClient, context: RunContext) -> None:
    """Run every configured concurrency level or arrival rate sequentially (FR-005/016)."""
    if context.run.mode == "open":
        for rate in context.run.request_rates:
            await _run_rate(client, context, rate)
        return
    for level in context.run.concurrency_levels:
        await _run_level(client, context, level)
