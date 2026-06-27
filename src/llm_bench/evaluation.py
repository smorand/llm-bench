"""Asynchronous, decoupled output-quality evaluation pipeline (SC-004).

This module implements the FR-040..047 evaluation pipeline: a worker pool that
consumes a bounded queue of evaluation records produced by the load generator,
scores each one (embedding cosine similarity by default, optional LLM-as-judge),
and joins the scores back onto the performance records by ``request_id``.

Design (Section 7, eval architecture):

* The load generator, on each completed request carrying an ``expected_output``,
  enqueues a lightweight :class:`EvalRecord` non-blocking (FR-040). The queue is
  bounded (``run.eval_queue_maxsize``); when it is full the item is dropped and a
  single aggregate counter is bumped, never blocking the load generator (FR-041).
* A separate worker pool (its own :class:`asyncio.Semaphore` / rate limiter on the
  embedding or judge provider, distinct from the SUT concurrency) drains the queue
  (FR-042). Embedding scoring computes cosine similarity and applies the inclusive
  threshold (FR-043); judge scoring grades with a binary or three-level rubric and
  never produces a numeric 1-10 score (FR-044).
* When the load test ends the caller publishes perf metrics first, then drains the
  queue and backfills scores (FR-045/047). A global timeout bounds draining; any
  still-unscored record is marked ``eval_skipped`` (FR-046). An unreachable
  endpoint marks the affected records ``eval_skipped`` while the perf data stays
  valid.

Only the judge / embedding LLM calls are traced, and only with model / token /
duration attributes; prompts, responses, and secrets are never logged or traced
(FR-056/FR-057).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from llm_bench import local_embed, metrics
from llm_bench.tracing import trace_span

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from llm_bench.config import EvaluationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Eval status vocabulary persisted onto each raw record.
EVAL_JUDGED = "judged"
EVAL_SKIPPED = "eval_skipped"
EVAL_SKIPPED_NO_EXPECTED = "skipped_no_expected"
EVAL_DROPPED = "eval_dropped"

# Default bound for the eval queue when the config omits eval_queue_maxsize; large
# enough that an ordinary run never spills (spilling is opt-in via a small bound).
_DEFAULT_QUEUE_MAXSIZE = 10000

# Default number of eval workers; the provider rate limiter is the real throttle.
_DEFAULT_WORKERS = 4

_HTTP_OK = 200

# Binary / three-level rubric verdict vocabularies (FR-044); never numeric.
_BINARY_VERDICTS: frozenset[str] = frozenset({"pass", "fail"})
_THREE_LEVEL_VERDICTS: frozenset[str] = frozenset({"correct", "partial", "incorrect"})


# ---------------------------------------------------------------------------
# Records and results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalRecord:
    """A lightweight evaluation request enqueued by the load generator (FR-040).

    Args:
        request_id: Join key back onto the performance record (FR-047).
        expected: The prompt's reference ``expected_output`` text.
        actual: The raw model output text (never cache-busting-prefixed).
    """

    request_id: str
    expected: str
    actual: str


@dataclass(slots=True)
class EvalResult:
    """A scored evaluation outcome to backfill onto a performance record (FR-047)."""

    eval_status: str
    sim_score: float | None = None
    quality_pass: bool | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None
    # Unified 0..1 quality score (embedding cosine, judge 'score' rubric, or a
    # mapping of the categorical verdict) for use as a dashboard metric.
    quality_score: float | None = None


# ---------------------------------------------------------------------------
# Provider rate limiter (FR-042)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """A minimal monotonic-clock pacing gate capping calls per second (FR-042).

    Serialises the "next allowed instant" under a lock so the eval pool issues at
    most ``rate`` calls per second to the embedding / judge provider, independent
    of the SUT concurrency. A non-positive rate disables pacing.
    """

    __slots__ = ("_interval", "_lock", "_next_allowed")

    def __init__(self, rate: float | None) -> None:
        self._interval = (1.0 / rate) if rate and rate > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        """Block until the next call is permitted by the configured rate."""
        if self._interval <= 0.0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            start = self._next_allowed if wait > 0 else now
            self._next_allowed = start + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Eval pipeline
# ---------------------------------------------------------------------------


class EvalPipeline:
    """Decoupled eval worker pool consuming a bounded queue (FR-040..047).

    The pipeline is created once per run. The load generator calls
    :meth:`enqueue` (non-blocking) for each completed request with an expected
    output; :meth:`drain` is awaited after the perf summary is published to score
    the backlog under a global timeout and return the per-``request_id`` results.
    """

    def __init__(
        self,
        evaluation: EvaluationConfig,
        *,
        queue_maxsize: int | None,
        global_timeout: float | None,
        tracer: Tracer | None = None,
    ) -> None:
        self._evaluation = evaluation
        self._method = evaluation.method
        self._global_timeout = global_timeout
        self._tracer = tracer
        maxsize = queue_maxsize if queue_maxsize and queue_maxsize > 0 else _DEFAULT_QUEUE_MAXSIZE
        self._queue: asyncio.Queue[EvalRecord] = asyncio.Queue(maxsize=maxsize)
        self._results: dict[str, EvalResult] = {}
        self._pending: set[str] = set()
        self._dropped = 0
        self._rate_limiter = _RateLimiter(_provider_rate_limit(evaluation))
        self._endpoint_down = False
        self._down_warned = False
        self._client: httpx.AsyncClient | None = None
        self._stopping = False
        self._workers: list[asyncio.Task[None]] = []
        self._enqueued = 0

    @property
    def dropped(self) -> int:
        """The aggregate count of records spilled on a full queue (FR-041)."""
        return self._dropped

    def enqueue(self, record: EvalRecord) -> None:
        """Enqueue an eval record without blocking the load generator (FR-040/041).

        On a full queue the record is dropped and the aggregate ``dropped``
        counter is bumped (FR-041); the load generator is never blocked.
        """
        if self._stopping:
            self._dropped += 1
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._dropped += 1
            return
        self._pending.add(record.request_id)
        self._enqueued += 1

    def progress(self) -> tuple[int, int]:
        """Return ``(scored, enqueued)`` for the live quality-eval progress bar.

        ``scored`` is the count of records the worker pool has finished evaluating;
        ``enqueued`` is the count accepted into the queue so far (drops excluded).
        Their ratio is how caught-up the concurrent eval is with the load.
        """
        return len(self._results), self._enqueued

    async def start(self) -> None:
        """Spawn the worker pool so the queue drains concurrently with the load (FR-045).

        Workers consume eval records as the load generator enqueues them, so the
        scoring runs in parallel with the benchmark instead of as a trailing phase.
        Idempotent: a second call is a no-op.
        """
        if self._workers:
            return
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._workers = [asyncio.create_task(self._worker()) for _ in range(_DEFAULT_WORKERS)]

    async def finish(self) -> dict[str, EvalResult]:
        """Stop intake, await the remaining backlog, join results (FR-045/046).

        The global timeout bounds only the post-load tail (records still queued
        when the sweep ends); anything scored during the load already counts.
        Still-pending records are marked ``eval_skipped`` (FR-046).
        """
        self._stopping = True
        timed_out = await self._await_queue()
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers = []
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if timed_out:
            remaining = len(self._pending)
            logger.warning("eval global timeout (%s) reached, %d items skipped", self._timeout_label(), remaining)
        self._mark_remaining_skipped()
        return dict(self._results)

    async def drain(self) -> dict[str, EvalResult]:
        """Score the whole backlog start-to-finish (when no concurrent pool was started)."""
        await self.start()
        return await self.finish()

    async def _await_queue(self) -> bool:
        """Wait for the queue to fully drain, honoring the global timeout.

        Returns:
            ``True`` when the global timeout fired before the backlog cleared.
        """
        join = asyncio.ensure_future(self._queue.join())
        try:
            await asyncio.wait_for(asyncio.shield(join), timeout=self._global_timeout)
        except TimeoutError:
            join.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await join
            return True
        return False

    async def _worker(self) -> None:
        """Consume eval records, scoring each and recording its result."""
        while True:
            record = await self._queue.get()
            try:
                await self._score(record)
            finally:
                self._queue.task_done()

    async def _score(self, record: EvalRecord) -> None:
        """Score one record by the active method, storing its result (FR-043/044)."""
        if self._endpoint_down:
            self._store(record.request_id, EvalResult(eval_status=EVAL_SKIPPED))
            return
        try:
            result = await self._evaluate(record)
        except httpx.HTTPError:
            self._mark_endpoint_down()
            self._store(record.request_id, EvalResult(eval_status=EVAL_SKIPPED))
            return
        self._store(record.request_id, result)

    async def _evaluate(self, record: EvalRecord) -> EvalResult:
        """Dispatch to embedding or judge scoring for one record."""
        if self._method == "judge":
            return await self._judge(record)
        return await self._embed(record)

    def _store(self, request_id: str, result: EvalResult) -> None:
        """Record a scored result and clear the request from the pending set."""
        self._results[request_id] = result
        self._pending.discard(request_id)

    # -- embedding scoring ----------------------------------------------------

    async def _embed(self, record: EvalRecord) -> EvalResult:
        """Embed both texts, compute cosine, apply the inclusive threshold (FR-043)."""
        embedding = self._evaluation.embedding
        if embedding is None or embedding.threshold is None:  # pragma: no cover - guarded by FR-003
            return EvalResult(eval_status=EVAL_SKIPPED)
        await self._rate_limiter.acquire()
        inputs = [record.expected, record.actual]
        if embedding.local:
            vectors = await self._embed_local(embedding.local, inputs)
        else:
            vectors = await self._embed_call(embedding, inputs)
        sim = metrics.cosine_similarity(vectors[0], vectors[1])
        return EvalResult(
            eval_status=EVAL_JUDGED,
            sim_score=sim,
            quality_score=sim,
            quality_pass=sim >= embedding.threshold,
        )

    async def _embed_call(self, embedding: Any, inputs: list[str]) -> list[list[float]]:
        """POST both texts to the embeddings endpoint and return their vectors."""
        client = self._require_client()
        url = f"{str(embedding.url).rstrip('/')}/embeddings"
        payload = {"model": embedding.model, "input": inputs}
        headers = _bearer(embedding.api_key)
        with trace_span(
            "eval.embedding",
            attributes={"llm.model": embedding.model, "llm.inputs": len(inputs)},
            tracer=self._tracer,
        ):
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", [])
        return [list(item.get("embedding", [])) for item in data]

    async def _embed_local(self, preset: str, inputs: list[str]) -> list[list[float]]:
        """Embed in-process with the built-in fastembed model (off the event loop)."""
        with trace_span(
            "eval.embedding",
            attributes={"llm.model": f"local:{preset}", "llm.inputs": len(inputs)},
            tracer=self._tracer,
        ):
            return await asyncio.to_thread(local_embed.embed_texts, inputs, preset)

    # -- judge scoring --------------------------------------------------------

    async def _judge(self, record: EvalRecord) -> EvalResult:
        """Grade the output with the judge model using the configured rubric (FR-044).

        ``score`` rubric: the model returns a 0..1 compliance score, stored as
        ``quality_score``. ``binary`` / ``three_level``: a categorical verdict,
        also mapped to a 0..1 ``quality_score`` so dashboards have one metric.
        """
        judge = self._evaluation.judge
        if judge is None:  # pragma: no cover - guarded by config selection
            return EvalResult(eval_status=EVAL_SKIPPED)
        await self._rate_limiter.acquire()
        body = await self._judge_call(judge, record)
        if judge.rubric == "score":
            score, reason = _parse_judge_score(body)
            return EvalResult(eval_status=EVAL_JUDGED, quality_score=score, judge_reason=reason)
        verdict, reason = _parse_judge_reply(body)
        normalized = _normalize_verdict(verdict, judge.rubric)
        return EvalResult(
            eval_status=EVAL_JUDGED,
            judge_verdict=normalized,
            judge_reason=reason,
            quality_score=_verdict_to_score(normalized, judge.rubric),
        )

    async def _judge_call(self, judge: Any, record: EvalRecord) -> dict[str, Any]:
        """POST a grading request and return the parsed chat-completion reply body."""
        client = self._require_client()
        model = judge.model
        url = f"{str(model.url).rstrip('/')}/chat/completions"
        payload = _judge_payload(model.model, model.prompt, record, judge.rubric)
        headers = _bearer(model.api_key)
        with trace_span(
            "eval.judge",
            attributes={"llm.model": model.model, "llm.rubric": judge.rubric},
            tracer=self._tracer,
        ):
            response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    # -- endpoint / timeout bookkeeping --------------------------------------

    def _mark_endpoint_down(self) -> None:
        """Flag the provider unreachable and warn once with the method-specific text."""
        self._endpoint_down = True
        if self._down_warned:
            return
        self._down_warned = True
        if self._method == "judge":
            logger.warning("judge endpoint unreachable, marking eval_skipped")
        else:
            logger.warning("embedding endpoint unreachable, marking eval_skipped")

    def _mark_remaining_skipped(self) -> None:
        """Mark every still-pending record ``eval_skipped`` after draining (FR-046)."""
        for request_id in self._pending:
            self._results.setdefault(request_id, EvalResult(eval_status=EVAL_SKIPPED))
        self._pending.clear()

    def _require_client(self) -> httpx.AsyncClient:
        """Return the active HTTP client, or raise if drain was not entered."""
        if self._client is None:  # pragma: no cover - drain always sets the client
            raise RuntimeError("eval HTTP client is not initialised")
        return self._client

    def _timeout_label(self) -> str:
        """Render the global timeout for the skip warning (``1s``)."""
        if self._global_timeout is None:  # pragma: no cover - timeout path implies a value
            return "none"
        text = f"{self._global_timeout:g}"
        return f"{text}s"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider_rate_limit(evaluation: EvaluationConfig) -> float | None:
    """Return the configured eval-pool rate limit (req/s), or ``None`` (FR-042)."""
    if evaluation.method == "embedding" and evaluation.embedding is not None:
        return evaluation.embedding.rate_limit
    return None


def _bearer(api_key: str | None) -> dict[str, str]:
    """Build request headers, adding a Bearer token only when a key is present."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _judge_payload(model: str, prompt: str | None, record: EvalRecord, rubric: str) -> dict[str, Any]:
    """Build the judge chat-completion request body for one record (FR-044)."""
    instruction = prompt or "Grade the answer against the expected output."
    if rubric == "score":
        system = (
            f'{instruction} Respond with a JSON object {{"score", "reason"}} where score is a '
            "single compliance number between 0 and 1 (0 = wrong, 1 = fully correct)."
        )
    else:
        vocabulary = "correct, partial, or incorrect" if rubric == "three_level" else "pass or fail"
        system = (
            f'{instruction} Respond with a JSON object {{"verdict", "reason"}} where verdict is '
            f"one of {vocabulary}. Do not use any numeric score."
        )
    user = json.dumps({"expected": record.expected, "actual": record.actual})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "stream": False,
    }


# Categorical verdicts mapped to a 0..1 quality score for the unified metric.
_VERDICT_SCORE: dict[str, float] = {
    "pass": 1.0,  # nosec B105  ('pass' is a judge verdict, not a password)
    "fail": 0.0,
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
}


def _verdict_to_score(verdict: str, _rubric: str) -> float:
    """Map a normalized categorical verdict to a 0..1 quality score."""
    return _VERDICT_SCORE.get(verdict, 0.0)


def _judge_content(body: dict[str, Any]) -> str:
    """Return the assistant message content from a chat-completion reply."""
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return ""


def _parse_judge_score(body: dict[str, Any]) -> tuple[float, str]:
    """Extract a clamped 0..1 ``(score, reason)`` from a judge 'score' reply."""
    content = _judge_content(body)
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    raw_score: Any = parsed.get("score") if isinstance(parsed, dict) else content
    reason = str(parsed.get("reason", "")).strip() if isinstance(parsed, dict) else content.strip()
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score)), reason


def _parse_judge_reply(body: dict[str, Any]) -> tuple[str, str]:
    """Extract the ``(verdict, reason)`` from a judge chat-completion reply."""
    choices = body.get("choices")
    content = ""
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = str(message.get("content", ""))
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content.strip(), content.strip()
    if not isinstance(parsed, dict):
        return str(parsed), str(parsed)
    verdict = str(parsed.get("verdict", "")).strip()
    reason = str(parsed.get("reason", verdict)).strip()
    return verdict, reason


def _normalize_verdict(verdict: str, rubric: str) -> str:
    """Map a raw judge verdict into the rubric vocabulary (FR-044).

    Numeric verdicts are never accepted; an out-of-vocabulary verdict maps to the
    negative pole of the rubric so the result stays categorical, never a score.
    """
    token = verdict.strip().lower()
    if rubric == "three_level":
        return token if token in _THREE_LEVEL_VERDICTS else "incorrect"
    if token in _BINARY_VERDICTS:
        return token
    return "pass" if token in {"yes", "true", "correct"} else "fail"
