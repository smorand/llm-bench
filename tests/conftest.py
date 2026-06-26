"""Shared test fixtures: the offline FakeSUT / FakeEval harness.

This module implements the "Common harness" described in Section 12 of the
specification. Every endpoint exercised by the end-to-end suite is replaced by a
local, deterministic, in-process ``aiohttp`` server so the whole test suite runs
fully offline.

Three building blocks are provided:

* :class:`SUTController` + the ``fake_sut`` fixture: a scriptable
  OpenAI-compatible streaming chat-completions server (the system under test).
* :class:`EvalController` + the ``fake_eval`` fixture: fake ``/v1/embeddings``
  and judge ``/v1/chat/completions`` endpoints with deterministic outputs.
* The ``cfg_base`` fixture: writes a ``CFG_BASE``-style ``config.yaml`` into a
  temporary directory wired to the running ``FakeSUT`` and sets ``SUT_API_KEY``.

Later test files (T-001..T-007) import these fixtures; keep them generic.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bind_loopback_socket() -> socket.socket:
    """Bind and listen on a free loopback TCP port, returning the bound socket.

    Binding the listening socket up front (rather than just reserving a port
    number) removes the classic bind-after-close race: the socket is handed to
    aiohttp's :class:`~aiohttp.web.SockSite` already bound, so no other process
    can steal the port between reservation and serving.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    # Large backlog so the 1000-virtual-user saturation test (E2E-010) can establish
    # ~1000 concurrent connections without the accept queue throttling the herd.
    sock.listen(1024)
    sock.setblocking(False)
    return sock


def _sse(payload: dict[str, Any]) -> bytes:
    """Encode a dict as a single Server-Sent Event ``data:`` line."""
    return f"data: {json.dumps(payload)}\n\n".encode()


_SSE_DONE: bytes = b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Background-thread server runtime
# ---------------------------------------------------------------------------


class _ThreadedServer:
    """Run an aiohttp app on a dedicated thread with its own event loop.

    The whole reason this class exists: the production CLI runs the benchmark
    under its own ``asyncio.run(...)`` event loop. If the fake servers shared the
    test's loop, that loop would be blocked inside ``asyncio.run`` for the whole
    run and the servers' sockets would never be serviced, so every real request
    would hit a read timeout. By giving each fake server its own thread and loop,
    its listening socket is serviced independently of whichever loop the test or
    the CLI is currently driving.

    The listening socket is bound *before* the thread starts (on the caller's
    thread) so the chosen port is known race-free; the bound socket is then
    handed to :class:`~aiohttp.web.SockSite` inside the server loop.
    """

    def __init__(self, app: web.Application, sock: socket.socket) -> None:
        self._app = app
        self._sock = sock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None

    def start(self) -> None:
        """Start the background thread and block until the server accepts."""
        self._thread = threading.Thread(target=self._run, name="fake-server", daemon=True)
        self._thread.start()
        # Wait until the server loop reports it is up (or failed to start).
        if not self._ready.wait(timeout=10.0):  # pragma: no cover - defensive
            raise RuntimeError("fake server did not start within 10s")
        if self._start_error is not None:  # pragma: no cover - defensive
            raise self._start_error

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._setup())
        except BaseException as exc:  # pragma: no cover - defensive
            self._start_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self._teardown())
            loop.close()

    async def _setup(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.SockSite(self._runner, self._sock)
        await site.start()

    async def _teardown(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def stop(self) -> None:
        """Stop the server loop and join the thread cleanly."""
        loop = self._loop
        thread = self._thread
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10.0)
        self._loop = None
        self._thread = None


# ---------------------------------------------------------------------------
# FakeSUT behavior model
# ---------------------------------------------------------------------------


@dataclass
class Delta:
    """One streamed content delta and the delay that precedes it.

    Args:
        text: Content fragment emitted in this chunk.
        sleep_ms: Milliseconds to sleep *before* emitting this chunk, used to
            shape inter-token latency (ITL) deterministically.
    """

    text: str
    sleep_ms: float = 0.0


@dataclass
class Usage:
    """Final ``usage`` object returned on the terminal SSE chunk."""

    prompt_tokens: int = 10
    completion_tokens: int = 8
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Render the OpenAI-compatible usage payload with details."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
        }


@dataclass
class Behavior:
    """Scriptable per-request behavior for :class:`FakeSUT`.

    Every knob maps to an item in the spec's harness checklist. A single
    ``Behavior`` instance fully determines the server's response to one request.

    Args:
        status: HTTP status code to return (200, 429, 500, ...).
        role_first_chunk: Emit an initial role-only delta (no content) first.
        deltas: Ordered content deltas, each with its own pre-delay.
        usage: Final usage object; omitted entirely when ``omit_usage`` is set.
        omit_usage: Drop the final usage chunk (still send ``[DONE]``).
        force_timeout_ms: If set, sleep this long before the first byte to force
            a client-side timeout.
        mid_stream_close: Abort the connection mid-stream without sending
            ``[DONE]`` (simulates a TCP reset / truncated stream).
        malformed_line: Inject one unparseable SSE ``data:`` line into the
            stream to exercise malformed-stream handling.
        error_body: Optional JSON body to return for non-200 responses.
        auto_tool_call: When true (the default), the server inspects the request
            body and, if it declares a non-empty ``tools`` array yet carries no
            prior ``tool``-role message, streams a single ``tool_calls`` delta
            naming the declared tool with model-provided ``arguments`` (an
            ``outline`` plus a ``query``) and a ``finish_reason:"tool_calls"``
            terminal chunk. The benchmark client then runs the local mock handler
            and POSTs the tool result back; that follow-up request carries a
            ``tool``-role message, so the server streams a normal synthesis turn.
            This is the harness knob that drives the SC-005 tool round-trip
            (E2E-036/037). It is inert for every request that does not declare
            ``tools``, so the existing suite is unaffected.
    """

    status: int = 200
    role_first_chunk: bool = True
    deltas: list[Delta] = field(default_factory=lambda: [Delta("x") for _ in range(8)])
    usage: Usage = field(default_factory=Usage)
    omit_usage: bool = False
    force_timeout_ms: float | None = None
    mid_stream_close: bool = False
    malformed_line: bool = False
    error_body: dict[str, Any] | None = None
    auto_tool_call: bool = True


def _requested_tool_name(body: dict[str, Any]) -> str | None:
    """Return the first declared function tool name in a request body, if any."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            return name
    return None


def _carries_tool_result(body: dict[str, Any]) -> bool:
    """Return true once the request body carries a ``tool``-role result message."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    return any(isinstance(message, dict) and message.get("role") == "tool" for message in messages)


def _should_emit_tool_call(behavior: Behavior, body: dict[str, Any]) -> str | None:
    """Return the tool name to call for a first-turn tool request, else ``None``.

    The server emits a ``tool_calls`` turn only when auto tool calling is enabled,
    the request declares a function tool, and no tool result has been posted back
    yet (so the synthesis turn streams normally on the follow-up request).
    """
    if not behavior.auto_tool_call or _carries_tool_result(body):
        return None
    return _requested_tool_name(body)


def default_behavior(n_deltas: int = 8, sleep_ms: float = 5.0) -> Behavior:
    """Return a vanilla 200 behavior with ``n_deltas`` content chunks.

    Args:
        n_deltas: Number of single-character content deltas to stream.
        sleep_ms: Per-delta pre-delay in milliseconds.
    """
    return Behavior(
        deltas=[Delta("x", sleep_ms=sleep_ms) for _ in range(n_deltas)],
        usage=Usage(prompt_tokens=10, completion_tokens=n_deltas),
    )


# ---------------------------------------------------------------------------
# FakeSUT controller + server
# ---------------------------------------------------------------------------


@dataclass
class RecordedRequest:
    """A request body captured by :class:`FakeSUT` for later assertions."""

    body: dict[str, Any]
    headers: dict[str, str]


class SUTController:
    """Drives :class:`FakeSUT`: scripts behaviors and records what arrived.

    The controller is the single object a test manipulates. It exposes the bound
    ``base_url`` (ending in ``/v1``), lets the test set a default behavior, queue
    per-request behaviors, or install an arbitrary callable, and records every
    received request body plus the maximum observed in-flight concurrency.
    """

    def __init__(self) -> None:
        self._default: Behavior = default_behavior()
        self._scripts: list[Behavior] = []
        self._fn: Callable[[int, dict[str, Any]], Behavior] | None = None
        self.requests: list[RecordedRequest] = []
        self._in_flight: int = 0
        self.max_in_flight: int = 0
        self.request_count: int = 0
        self.base_url: str = ""

    # -- scripting -------------------------------------------------------
    def set_default(self, behavior: Behavior) -> None:
        """Set the behavior used when no per-request script applies."""
        self._default = behavior

    def queue(self, behaviors: Iterable[Behavior]) -> None:
        """Queue an ordered list of behaviors, one consumed per request."""
        self._scripts.extend(behaviors)

    def set_function(self, fn: Callable[[int, dict[str, Any]], Behavior]) -> None:
        """Install a callable ``(index, body) -> Behavior`` deciding each response.

        Takes precedence over queued scripts and the default behavior.
        """
        self._fn = fn

    def nth_request_status(self, n: int, status: int) -> None:
        """Convenience: make the (0-based) ``n``-th request return ``status``.

        All other requests use the current default behavior.
        """
        default = self._default

        def _fn(index: int, _body: dict[str, Any]) -> Behavior:
            if index == n:
                return Behavior(status=status, role_first_chunk=False, deltas=[], error_body={"error": "scripted"})
            return default

        self._fn = _fn

    # -- internal --------------------------------------------------------
    def _next_behavior(self, index: int, body: dict[str, Any]) -> Behavior:
        if self._fn is not None:
            return self._fn(index, body)
        if index < len(self._scripts):
            return self._scripts[index]
        return self._default


class FakeSUT:
    """In-process OpenAI-compatible streaming chat-completions server.

    Exposes ``POST /v1/chat/completions`` bound to ``127.0.0.1`` on a random free
    port. Responses are driven by a :class:`SUTController`. SSE events are
    formatted as ``data: {json}\\n\\n`` and terminated by ``data: [DONE]\\n\\n``.

    The server runs on a dedicated :class:`_ThreadedServer` so its socket is
    serviced on its own event loop, independently of whatever loop the test (or
    the production CLI's ``asyncio.run``) happens to be driving.
    """

    def __init__(self, controller: SUTController) -> None:
        self._controller = controller

    @property
    def base_url(self) -> str:
        """Base URL including the ``/v1`` suffix."""
        return self._controller.base_url

    def build_app(self) -> web.Application:
        """Build the aiohttp application served on the background loop."""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle_chat)
        return app

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        ctrl = self._controller
        index = ctrl.request_count
        ctrl.request_count += 1

        body: dict[str, Any] = await request.json()
        ctrl.requests.append(RecordedRequest(body=body, headers=dict(request.headers)))

        ctrl._in_flight += 1
        ctrl.max_in_flight = max(ctrl.max_in_flight, ctrl._in_flight)
        try:
            behavior = ctrl._next_behavior(index, body)

            if behavior.force_timeout_ms is not None:
                await asyncio.sleep(behavior.force_timeout_ms / 1000.0)

            if behavior.status != 200:
                payload = behavior.error_body or {"error": {"message": "error", "code": behavior.status}}
                return web.json_response(payload, status=behavior.status)

            tool_name = _should_emit_tool_call(behavior, body)
            if tool_name is not None:
                return await self._stream_tool_call(request, behavior, tool_name)
            return await self._stream(request, behavior)
        finally:
            ctrl._in_flight -= 1

    async def _stream_tool_call(self, request: web.Request, behavior: Behavior, tool_name: str) -> web.StreamResponse:
        """Stream a single ``tool_calls`` turn naming ``tool_name`` (SC-005).

        The arguments are deterministic and carry both a generic ``query`` and an
        ``outline`` so the pptx round-trip (E2E-037) can assert the model-provided
        ``outline`` survived into the recorded handler invocation.
        """
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)
        if behavior.role_first_chunk:
            await response.write(_sse(_chunk(delta={"role": "assistant"})))
        arguments = json.dumps({"query": "deterministic mock query", "outline": "1. Intro\n2. Body\n3. Close"})
        tool_call_delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_fake_0",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ]
        }
        await response.write(_sse(_chunk(delta=tool_call_delta)))
        await response.write(_sse(_chunk(delta={}, finish_reason="tool_calls", usage=behavior.usage.to_dict())))
        await response.write(_SSE_DONE)
        await response.write_eof()
        return response

    async def _stream(self, request: web.Request, behavior: Behavior) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await response.prepare(request)

        if behavior.role_first_chunk:
            await response.write(_sse(_chunk(delta={"role": "assistant"})))

        for i, delta in enumerate(behavior.deltas):
            if delta.sleep_ms:
                await asyncio.sleep(delta.sleep_ms / 1000.0)
            if behavior.mid_stream_close and i == len(behavior.deltas) // 2:
                # Truncate the stream: drop the connection without [DONE].
                await response.write_eof()
                return response
            await response.write(_sse(_chunk(delta={"content": delta.text})))

        if behavior.malformed_line:
            await response.write(b"data: {not-valid-json\n\n")

        if not behavior.omit_usage:
            await response.write(_sse(_chunk(delta={}, finish_reason="stop", usage=behavior.usage.to_dict())))

        await response.write(_SSE_DONE)
        await response.write_eof()
        return response


def _chunk(
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI streaming chat-completion chunk."""
    payload: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "model": "fake/model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


# ---------------------------------------------------------------------------
# FakeEval controller + server
# ---------------------------------------------------------------------------


class EvalController:
    """Drives :class:`FakeEval`: deterministic embeddings and judge verdicts.

    The embedding endpoint returns a fixed vector per input text so cosine
    similarity is hand-computable in tests. Unmapped inputs fall back to
    ``default_vector``. The judge endpoint returns ``judge_verdict`` verbatim.
    "down" mode means the server is never started (connection refused); "slow"
    mode delays every response by ``delay_ms``.
    """

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.default_vector: list[float] = [1.0, 0.0, 0.0]
        self.judge_verdict: dict[str, Any] = {"verdict": "pass", "reason": "ok"}
        self.delay_ms: float = 0.0
        self.down: bool = False
        self.embedding_requests: list[dict[str, Any]] = []
        self.judge_requests: list[dict[str, Any]] = []
        self.base_url: str = ""

    def map_text(self, text: str, vector: list[float]) -> None:
        """Map an exact input ``text`` to the embedding ``vector`` to return."""
        self.vectors[text] = vector

    def vector_for(self, text: str) -> list[float]:
        """Return the configured vector for ``text`` (or the default)."""
        return self.vectors.get(text, self.default_vector)


class FakeEval:
    """In-process fake embeddings + judge server.

    Exposes ``POST /v1/embeddings`` (deterministic vectors) and a judge
    ``POST /v1/chat/completions`` (fixed JSON verdict), bound to ``127.0.0.1`` on
    a random free port. When the controller is in "down" mode the server is not
    started, so connections are refused.
    """

    def __init__(self, controller: EvalController) -> None:
        self._controller = controller

    @property
    def base_url(self) -> str:
        """Base URL including the ``/v1`` suffix (valid even when down)."""
        return self._controller.base_url

    def build_app(self) -> web.Application:
        """Build the aiohttp application served on the background loop."""
        app = web.Application()
        app.router.add_post("/v1/embeddings", self._handle_embeddings)
        app.router.add_post("/v1/chat/completions", self._handle_judge)
        return app

    async def _maybe_delay(self) -> None:
        if self._controller.delay_ms:
            await asyncio.sleep(self._controller.delay_ms / 1000.0)

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        await self._maybe_delay()
        body: dict[str, Any] = await request.json()
        self._controller.embedding_requests.append(body)
        raw_input = body.get("input", "")
        inputs = raw_input if isinstance(raw_input, list) else [raw_input]
        data = [
            {"object": "embedding", "index": i, "embedding": self._controller.vector_for(str(text))}
            for i, text in enumerate(inputs)
        ]
        return web.json_response({"object": "list", "data": data, "model": body.get("model", "fake/embed")})

    async def _handle_judge(self, request: web.Request) -> web.Response:
        await self._maybe_delay()
        body: dict[str, Any] = await request.json()
        self._controller.judge_requests.append(body)
        content = json.dumps(self._controller.judge_verdict)
        return web.json_response(
            {
                "id": "chatcmpl-judge",
                "object": "chat.completion",
                "model": body.get("model", "fake/judge"),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_default_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the default config and runs locations to tmp dirs for every test.

    A bare ``run`` defaults to ``~/.local/share/llm-bench/runs/`` and ``init``
    scaffolds ``~/.config/llm-bench/``; this guard guarantees no test ever writes
    into the real home directory.
    """
    monkeypatch.setattr("llm_bench.config.DEFAULT_RUNS_DIR", tmp_path / "default-runs")
    monkeypatch.setattr("llm_bench.config.DEFAULT_CONFIG_DIR", tmp_path / "default-config")


@pytest.fixture
def fake_sut() -> Iterator[tuple[str, SUTController]]:
    """Start a :class:`FakeSUT` and yield ``(base_url, controller)``.

    The default behavior is a 200 streamed completion with a role-only chunk then
    eight ``"x"`` content deltas. Tests reshape it via the controller.

    The server runs on a dedicated background thread (its own event loop), so its
    socket is serviced even while a test drives the production CLI's
    ``asyncio.run`` on a separate loop. The fixture is synchronous: it binds the
    port, starts the thread, waits until the listener is accepting, yields, then
    stops the loop and joins the thread on teardown.
    """
    controller = SUTController()
    server = FakeSUT(controller)
    sock = _bind_loopback_socket()
    port = int(sock.getsockname()[1])
    controller.base_url = f"http://127.0.0.1:{port}/v1"
    runtime = _ThreadedServer(server.build_app(), sock)
    runtime.start()
    try:
        yield controller.base_url, controller
    finally:
        runtime.stop()


@pytest.fixture
def fake_eval() -> Iterator[tuple[str, EvalController]]:
    """Start a :class:`FakeEval` and yield ``(base_url, controller)``.

    The server runs on a dedicated background thread with its own event loop, for
    the same reason as :func:`fake_sut`. The chosen port is reserved before the
    yield so ``base_url`` is valid even in "down" mode.

    "down" mode (connection refused) is honored by reserving the port but never
    starting a listener: set ``controller.down = True`` *before* the server
    starts by parametrizing/wiring the controller, and no background thread is
    launched, so connections are refused. The common case starts the server with
    deterministic vectors.
    """
    controller = EvalController()
    server = FakeEval(controller)
    sock = _bind_loopback_socket()
    port = int(sock.getsockname()[1])
    controller.base_url = f"http://127.0.0.1:{port}/v1"

    if controller.down:
        # No listener: close the reserved socket so connections are refused.
        sock.close()
        yield controller.base_url, controller
        return

    runtime = _ThreadedServer(server.build_app(), sock)
    runtime.start()
    try:
        yield controller.base_url, controller
    finally:
        runtime.stop()


@pytest.fixture
def cfg_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Return a factory writing a ``CFG_BASE``-style ``config.yaml``.

    The returned callable accepts the FakeSUT ``port`` (and optional overrides of
    the ``run`` block) and writes ``config.yaml`` into ``tmp_path``, sets the
    ``SUT_API_KEY`` and ``SUT_PORT`` environment variables, and returns the path
    to the written file. The config mirrors the spec's ``CFG_BASE`` defaults,
    including the ``base_url: http://127.0.0.1:${SUT_PORT}/v1`` and
    ``api_key: $ENV:SUT_API_KEY`` references.
    """

    def _write(port: int, *, run_overrides: dict[str, Any] | None = None, api_key: str = "sk-test") -> Path:
        monkeypatch.setenv("SUT_API_KEY", api_key)
        monkeypatch.setenv("SUT_PORT", str(port))

        run_block: dict[str, Any] = {
            "mode": "closed",
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "min_samples": 30,
            "concurrency_levels": [1, 2],
            "max_tokens": 8,
            "ignore_eos": True,
            "temperature": 0.0,
            "cache_busting": True,
            "retries": 0,
            "timeout": "5s",
            "seed": 42,
            "slo_profile": "interactive",
        }
        if run_overrides:
            run_block.update(run_overrides)

        # Written by hand (not via yaml.dump) to preserve the literal ${SUT_PORT}
        # and $ENV: tokens the loader is expected to resolve.
        levels = ", ".join(str(v) for v in run_block["concurrency_levels"])
        lines = [
            "models:",
            "  - name: sut",
            "    base_url: http://127.0.0.1:${SUT_PORT}/v1",
            "    model: fake/model",
            "    api_key: $ENV:SUT_API_KEY",
            "    supports_vision: false",
            "    supports_tools: false",
            "run:",
            f"  mode: {run_block['mode']}",
            f"  duration: {run_block['duration']}",
            f"  warmup: {run_block['warmup']}",
            f"  cooldown: {run_block['cooldown']}",
            f"  min_samples: {run_block['min_samples']}",
            f"  concurrency_levels: [{levels}]",
            f"  max_tokens: {run_block['max_tokens']}",
            f"  ignore_eos: {str(run_block['ignore_eos']).lower()}",
            f"  temperature: {run_block['temperature']}",
            f"  cache_busting: {str(run_block['cache_busting']).lower()}",
            f"  retries: {run_block['retries']}",
            f"  timeout: {run_block['timeout']}",
            f"  seed: {run_block['seed']}",
            f"  slo_profile: {run_block['slo_profile']}",
        ]
        # Forward any additional override keys (e.g. event_loop_lag_threshold_ms,
        # request_rates, burstiness, max_outstanding) so overrides are never silently
        # dropped. Bools render lower-case, lists as YAML flow sequences.
        emitted = {
            "mode",
            "duration",
            "warmup",
            "cooldown",
            "min_samples",
            "concurrency_levels",
            "max_tokens",
            "ignore_eos",
            "temperature",
            "cache_busting",
            "retries",
            "timeout",
            "seed",
            "slo_profile",
        }
        for key, value in run_block.items():
            if key in emitted:
                continue
            if isinstance(value, bool):
                rendered = str(value).lower()
            elif isinstance(value, list):
                rendered = "[" + ", ".join(str(item) for item in value) + "]"
            else:
                rendered = str(value)
            lines.append(f"  {key}: {rendered}")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return config_path

    return _write
