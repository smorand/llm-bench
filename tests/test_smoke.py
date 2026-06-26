"""Smoke tests: CLI version, harness sanity, and settings loading.

These keep the scaffolding honest: the CLI is wired, the FakeSUT harness serves
a real streamed completion end-to-end, and the process settings load.
"""

from __future__ import annotations

import asyncio
import json

import httpx
from typer.testing import CliRunner

from llm_bench.config import Settings
from llm_bench.llm_bench import app
from llm_bench.version import __version__

runner = CliRunner()


def test_version_flag() -> None:
    """``--version`` prints the version and exits 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
    assert "llm-bench" in result.stdout


def test_no_args_shows_help() -> None:
    """Invoking with no arguments shows help (no crash)."""
    result = runner.invoke(app, [])
    assert "run" in result.stdout
    assert "serve" in result.stdout
    assert "models" in result.stdout


def test_run_stub_exits_nonzero() -> None:
    """The ``run`` stub reports not-implemented and exits non-zero."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1


def test_settings_load() -> None:
    """Process settings load with sane defaults."""
    settings = Settings()
    assert settings.app_name == "llm-bench"
    assert settings.debug is False


async def test_fake_sut_streams_completion(fake_sut: tuple[str, object]) -> None:
    """FakeSUT serves a basic streamed completion end-to-end.

    Sanity-checks the harness: role-only first chunk, eight content deltas, a
    final usage chunk, and the ``[DONE]`` terminator; the request body is
    recorded for assertions.
    """
    base_url, controller = fake_sut

    payload = {
        "model": "fake/model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    content = ""
    usage: dict[str, object] | None = None
    saw_done = False

    async with (
        httpx.AsyncClient(timeout=5.0) as client,
        client.stream("POST", f"{base_url}/chat/completions", json=payload) as response,
    ):
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            content += delta.get("content", "")
            if chunk.get("usage"):
                usage = chunk["usage"]

    assert saw_done
    assert content == "x" * 8
    assert usage is not None
    assert usage["completion_tokens"] == 8

    # Harness recorded exactly one request carrying the streaming flags.
    assert controller.request_count == 1  # type: ignore[attr-defined]
    recorded = controller.requests[0]  # type: ignore[attr-defined]
    assert recorded.body["stream"] is True
    assert recorded.body["stream_options"]["include_usage"] is True


async def test_fake_sut_max_concurrency_tracked(fake_sut: tuple[str, object]) -> None:
    """FakeSUT tracks the maximum number of concurrent in-flight requests."""
    base_url, controller = fake_sut
    payload = {"model": "fake/model", "messages": [], "stream": True}

    async def _one() -> None:
        async with (
            httpx.AsyncClient(timeout=5.0) as client,
            client.stream("POST", f"{base_url}/chat/completions", json=payload) as response,
        ):
            async for _ in response.aiter_lines():
                pass

    await asyncio.gather(_one(), _one(), _one())
    assert controller.max_in_flight >= 1  # type: ignore[attr-defined]
    assert controller.request_count == 3  # type: ignore[attr-defined]
