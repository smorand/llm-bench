"""Harness self-test: a real CLI run reaches FakeSUT through ``asyncio.run``.

This guards the foundational property of the offline test harness: ``FakeSUT``
runs on a dedicated background thread (its own event loop), so its socket is
serviced even while the production CLI drives ``asyncio.run(run_benchmark(...))``
on a *separate* event loop. Before the threaded refactor the fake server shared
the test's loop, which ``asyncio.run`` blocked for the whole run, so every real
request hit a read timeout. This test fails loudly if that regression returns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from typer.testing import CliRunner

from llm_bench.llm_bench import app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import SUTController

runner = CliRunner()


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a FakeSUT ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def test_real_cli_request_reaches_fake_sut(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A non-dry ``llm-bench run`` actually streams requests to FakeSUT.

    Asserts the run exits 0, FakeSUT recorded real requests, at least one stream
    completed with outcome ``success``, and *no* request timed out (the symptom
    of the loop-starvation bug this harness fixes).
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(
        port,
        run_overrides={
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "concurrency_levels": [1],
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "runs" / "selftest"
    result = runner.invoke(app, ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)])

    assert result.exit_code == 0, result.stderr
    # A real request actually reached FakeSUT (not refused, not a timeout).
    assert len(controller.requests) >= 1, "no request reached FakeSUT"

    # At least one measured request streamed successfully; none timed out.
    raw_lines = (out_dir / "raw.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert raw_lines, "no raw.jsonl records written"
    outcomes = [json.loads(line)["outcome"] for line in raw_lines]
    assert "success" in outcomes, f"no successful stream, outcomes={outcomes}"
    assert "timeout" not in outcomes, f"unexpected timeouts: {outcomes}"

    # The streamed request carried the streaming flags the runner sends.
    assert controller.requests[0].body["stream"] is True
