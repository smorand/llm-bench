"""Acceptance tests for SC-005: deterministic mock tools, multimodal image, and
capability-mismatch skipping.

These cover scenario SC-005 (Section 5 of the spec
``specs/2026-06-24_09:28:00-llm-bench-core.md``) and its functional requirements
FR-038 (deterministic mock tools via the OpenAI ``tools`` field + unchanged
multimodal image content) and FR-039 (skip-with-warning when a prompt requires a
capability the model does not declare).

One test per E2E id from Section 12.2: E2E-036, 037, 038, 039, 040, 041, 042.
Each asserts exactly the observables named in the matching Gherkin, driven
through the ``llm-bench`` CLI (``run`` subcommand) with its ``--prompts``/
``--out`` flags against the offline FakeSUT harness from ``conftest.py``.

Assumed prompt-YAML shape (documented for the implementer):

* A *tool-use* prompt carries, in addition to the SC-001 shape
  (``id``/``category``/``messages``/``isl_bucket``), a ``tools`` key holding an
  OpenAI ``tools`` array (a list of ``{"type":"function","function":{...}}``
  mappings). The deterministic mock handler is keyed by the tool ``name`` and its
  fixed return payload is declared under ``tool_results`` (a mapping of tool name
  to the constant JSON payload the harness returns instead of calling out).
  ``category`` is ``"tool-use"``.
* A *vision* prompt carries ``messages`` whose user ``content`` is a list of
  multimodal parts, including an ``{"type":"image_url","image_url":{"url":
  "data:image/png;base64,..."}}`` part. ``category`` is ``"vision"``.

Summary skip-counter keys (FR-039): ``summary["skipped"]`` is a mapping with at
least ``tools_unsupported`` and ``vision_unsupported`` integer counters.

Skip log strings (FR-039), exact:

* tools:  ``skipping prompt <id>: model '<model>' does not support tools``
* vision: ``skipping prompt <id>: model '<model>' does not support vision``

HARNESS GAP (see report): the FakeSUT ``Behavior`` in ``conftest.py`` can only
stream role-only + content deltas; it cannot script a ``tool_call`` delta, a
tool-result turn, then a synthesis turn. So the tool-path tests assert the
observables that *are* available without that scripting: the request body sent to
the SUT contains the ``tools`` field with the mock tool, the deterministic mock
handler's fixed payload is a known constant (asserted directly against the
declared ``tool_results``), and the persisted record's ``category``/``outcome``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import yaml
from typer.testing import CliRunner

from llm_bench.llm_bench import app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import SUTController

runner = CliRunner()

# A small, valid 1x1 PNG encoded as a base64 data URL. The exact string must
# survive unchanged into the request body (FR-038, multimodal pass-through).
_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# The deterministic fixed payloads the mock tool handlers must return.
_WEB_SEARCH_RESULT: dict[str, Any] = {
    "results": [{"title": "Paris weather", "snippet": "18C, partly cloudy"}],
}
_PPTX_RESULT: dict[str, Any] = {"status": "ok", "slides": 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a FakeSUT ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact into a list of dicts, asserting it exists first."""
    assert path.exists(), f"expected artifact missing: {path}"
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_prompts(path: Path, prompts: list[dict[str, Any]]) -> Path:
    """Write a ``--prompts`` YAML file (SC-005 shape: prompts may carry
    ``tools``/``tool_results`` and multimodal ``content`` parts)."""
    path.write_text(yaml.safe_dump(prompts, sort_keys=False), encoding="utf-8")
    return path


def _write_capable_config(
    cfg_base_factory: Callable[..., Path],
    tmp_path: Path,
    port: int,
    *,
    supports_tools: bool,
    supports_vision: bool,
    run_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a CFG_BASE-style ``config.yaml`` with explicit capability flags.

    ``cfg_base`` always writes ``supports_tools: false``/``supports_vision:
    false``; tests needing a capable model rewrite the model block by hand,
    mirroring ``cfg_base``'s ``${SUT_PORT}`` / ``$ENV:`` token style, while still
    invoking the factory so ``SUT_PORT``/``SUT_API_KEY`` are exported into the
    environment.
    """
    # Invoke the factory so SUT_PORT / SUT_API_KEY env vars are set; we then
    # overwrite the file with a capability-tuned model block.
    overrides: dict[str, Any] = {
        "duration": "1s",
        "warmup": "0.2s",
        "cooldown": "0.2s",
        "concurrency_levels": [1],
        "min_samples": 2,
    }
    if run_overrides:
        overrides.update(run_overrides)
    config = cfg_base_factory(port, run_overrides=overrides)

    run_block = "\n".join(
        [
            "run:",
            f"  mode: {overrides.get('mode', 'closed')}",
            f"  duration: {overrides['duration']}",
            f"  warmup: {overrides['warmup']}",
            f"  cooldown: {overrides['cooldown']}",
            f"  min_samples: {overrides['min_samples']}",
            "  concurrency_levels: [" + ", ".join(str(v) for v in overrides["concurrency_levels"]) + "]",
            f"  max_tokens: {overrides.get('max_tokens', 8)}",
            f"  ignore_eos: {str(overrides.get('ignore_eos', True)).lower()}",
            f"  temperature: {overrides.get('temperature', 0.0)}",
            f"  cache_busting: {str(overrides.get('cache_busting', True)).lower()}",
            f"  retries: {overrides.get('retries', 0)}",
            f"  timeout: {overrides.get('timeout', '5s')}",
            f"  seed: {overrides.get('seed', 42)}",
            f"  slo_profile: {overrides.get('slo_profile', 'interactive')}",
        ]
    )
    lines = [
        "models:",
        "  - name: sut",
        "    base_url: http://127.0.0.1:${SUT_PORT}/v1",
        "    model: fake/model",
        "    api_key: $ENV:SUT_API_KEY",
        f"    supports_vision: {str(supports_vision).lower()}",
        f"    supports_tools: {str(supports_tools).lower()}",
        run_block,
        "",
    ]
    config.write_text("\n".join(lines), encoding="utf-8")
    return config


def _tool_prompt(
    prompt_id: str,
    tool_name: str,
    tool_result: dict[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    """Build a tool-use prompt mapping with an OpenAI ``tools`` array and a
    deterministic ``tool_results`` mapping (mock handler return constant)."""
    return {
        "id": prompt_id,
        "category": "tool-use",
        "isl_bucket": "short",
        "messages": [{"role": "user", "content": description}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        "tool_results": {tool_name: tool_result},
    }


def _vision_prompt(prompt_id: str) -> dict[str, Any]:
    """Build a vision prompt whose user content is a multimodal image part."""
    return {
        "id": prompt_id,
        "category": "vision",
        "isl_bucket": "short",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URL}},
                ],
            }
        ],
    }


def _plain_prompt(prompt_id: str, category: str = "general") -> dict[str, Any]:
    """Build a plain text prompt requiring no special capability."""
    return {
        "id": prompt_id,
        "category": category,
        "isl_bucket": "short",
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    }


def _body_has_tool(body: dict[str, Any], tool_name: str) -> bool:
    """Return True if the request body declares the named function tool."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if fn.get("name") == tool_name:
            return True
    return False


def _image_urls_in_body(body: dict[str, Any]) -> list[str]:
    """Return all ``image_url`` URL strings found in a request body's messages."""
    urls: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url")
                if isinstance(url, str):
                    urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# E2E-036: Tooling prompt deterministic mock web_search
# ---------------------------------------------------------------------------


def test_e2e_036_tool_use_mock_web_search(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A tools-capable model runs a tool-use prompt with the mock web_search tool.

    FR-038: exit 0; the request body to FakeSUT contains ``tools`` with
    ``mock_web_search``; the deterministic mock handler returned the fixed payload
    constant; the persisted record has ``category=="tool-use"`` and
    ``outcome=="success"``.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = _write_capable_config(cfg_base, tmp_path, port, supports_tools=True, supports_vision=False)
    prompts = _write_prompts(
        tmp_path / "tool_web.yaml",
        [
            _tool_prompt(
                "code-tool-001",
                "mock_web_search",
                _WEB_SEARCH_RESULT,
                description="What is the weather in Paris? Use the web search tool.",
            )
        ],
    )

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    assert controller.requests, "no requests captured"
    assert any(_body_has_tool(r.body, "mock_web_search") for r in controller.requests), (
        f"no request carried the mock_web_search tool: {[r.body.get('tools') for r in controller.requests]}"
    )

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written"
    assert all(r["category"] == "tool-use" for r in records), [r["category"] for r in records]
    assert all(r["outcome"] == "success" for r in records), [r["outcome"] for r in records]

    # The deterministic mock handler must have returned the fixed payload. The
    # harness records each handler invocation's result under the run output so the
    # constant is observable (handler call log == expected constant).
    tool_calls_path = out_dir / "tool_calls.jsonl"
    handler_results = _read_jsonl(tool_calls_path)
    assert any(call.get("result") == _WEB_SEARCH_RESULT for call in handler_results), (
        f"mock handler did not return the fixed web_search payload: {handler_results}"
    )


# ---------------------------------------------------------------------------
# E2E-037: Tooling prompt deterministic mock pptx
# ---------------------------------------------------------------------------


def test_e2e_037_tool_use_mock_pptx(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A tools-capable model runs a pptx tool-use prompt.

    FR-038: the ``mock_generate_pptx`` handler is invoked exactly once with the
    model-provided ``outline``; it returns the fixed ack ``{"status":"ok",
    "slides":3}``; the persisted record reflects the pptx tool-use category and
    ``outcome=="success"``.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = _write_capable_config(cfg_base, tmp_path, port, supports_tools=True, supports_vision=False)
    prompts = _write_prompts(
        tmp_path / "tool_pptx.yaml",
        [
            _tool_prompt(
                "pptx-tool-001",
                "mock_generate_pptx",
                _PPTX_RESULT,
                description="Generate a 3-slide deck from this outline.",
            )
        ],
    )

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    assert any(_body_has_tool(r.body, "mock_generate_pptx") for r in controller.requests), (
        "no request carried the mock_generate_pptx tool"
    )

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written"
    assert all(r["category"] == "tool-use" for r in records), [r["category"] for r in records]
    assert all(r["outcome"] == "success" for r in records), [r["outcome"] for r in records]

    handler_results = _read_jsonl(out_dir / "tool_calls.jsonl")
    pptx_calls = [c for c in handler_results if c.get("tool") == "mock_generate_pptx"]
    assert len(pptx_calls) == 1, f"expected exactly one pptx handler call, got {len(pptx_calls)}"
    call = pptx_calls[0]
    assert call.get("result") == _PPTX_RESULT, call.get("result")
    # The handler must have been invoked with the model-provided outline argument.
    assert "outline" in (call.get("arguments") or {}), f"handler not called with an outline: {call}"


# ---------------------------------------------------------------------------
# E2E-038: Multimodal image prompt to vision model
# ---------------------------------------------------------------------------


def test_e2e_038_vision_image_prompt(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A vision-capable model receives the image_url part unchanged.

    FR-038: the request body to the SUT contains the ``image_url`` part with the
    exact base64 data URL; the persisted record has ``category=="vision"``,
    ``outcome=="success"`` and a measured ``ttft``.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = _write_capable_config(cfg_base, tmp_path, port, supports_tools=False, supports_vision=True)
    prompts = _write_prompts(tmp_path / "vision.yaml", [_vision_prompt("vis-001")])

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    assert controller.requests, "no requests captured"
    all_urls = [url for r in controller.requests for url in _image_urls_in_body(r.body)]
    assert all_urls, f"no image_url part reached the SUT: {[r.body.get('messages') for r in controller.requests]}"
    assert _IMAGE_DATA_URL in all_urls, "the exact base64 image data URL was not passed unchanged"

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written"
    assert all(r["category"] == "vision" for r in records), [r["category"] for r in records]
    assert all(r["outcome"] == "success" for r in records), [r["outcome"] for r in records]
    assert all(isinstance(r.get("ttft"), (int, float)) and r["ttft"] >= 0 for r in records), (
        f"ttft not measured for vision records: {[r.get('ttft') for r in records]}"
    )

    # The capability gate must have run and chosen NOT to skip the vision prompt
    # (the model declares supports_vision:true). This proves the FR-039 gate is
    # actually exercised rather than the image merely flowing through untouched.
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    skipped = summary.get("skipped")
    assert isinstance(skipped, dict), f"summary is missing the 'skipped' counters block: {summary.keys()}"
    assert skipped.get("vision_unsupported", 0) == 0, f"a vision-capable model must not skip vision prompts: {skipped}"


# ---------------------------------------------------------------------------
# E2E-039: Skip+warn tools prompt on non-tools model
# ---------------------------------------------------------------------------


def test_e2e_039_skip_tools_prompt_on_non_tools_model(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A tool-use prompt is skipped (not sent) on a model that lacks tools.

    FR-039: the tool-use prompt is not sent to the SUT; stderr/log contains
    ``skipping prompt code-tool-001: model 'fake/model' does not support tools``;
    summary ``skipped.tools_unsupported >= 1``; the plain prompt still runs; exit
    0.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    # cfg_base default already declares supports_tools: false / supports_vision: false.
    config = _write_capable_config(cfg_base, tmp_path, port, supports_tools=False, supports_vision=False)
    prompts = _write_prompts(
        tmp_path / "mixed.yaml",
        [
            _tool_prompt(
                "code-tool-001",
                "mock_web_search",
                _WEB_SEARCH_RESULT,
                description="Search the web for the weather.",
            ),
            _plain_prompt("plain-001"),
        ],
    )

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    # The tool-use prompt must never reach the SUT.
    assert not any(_body_has_tool(r.body, "mock_web_search") for r in controller.requests), (
        "tool-use prompt was sent to a non-tools model"
    )

    assert "skipping prompt code-tool-001: model 'fake/model' does not support tools" in result.stderr, result.stderr

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    skipped = summary.get("skipped", {})
    assert skipped.get("tools_unsupported", 0) >= 1, f"tools_unsupported not counted: {skipped}"

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "the plain prompt did not run"
    assert all(r["category"] != "tool-use" for r in records), [r["category"] for r in records]


# ---------------------------------------------------------------------------
# E2E-040: Skip+warn vision prompt on non-vision model
# ---------------------------------------------------------------------------


def test_e2e_040_skip_vision_prompt_on_non_vision_model(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A vision prompt is skipped on a model that lacks vision.

    FR-039: the vision prompt is skipped; stderr/log contains ``skipping prompt
    vis-001: model 'fake/model' does not support vision``; summary
    ``skipped.vision_unsupported >= 1``; the plain prompt runs; exit 0.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = _write_capable_config(cfg_base, tmp_path, port, supports_tools=False, supports_vision=False)
    prompts = _write_prompts(
        tmp_path / "mixed.yaml",
        [_vision_prompt("vis-001"), _plain_prompt("plain-001")],
    )

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    # No image content must reach the SUT.
    assert not any(_image_urls_in_body(r.body) for r in controller.requests), (
        "vision prompt was sent to a non-vision model"
    )

    assert "skipping prompt vis-001: model 'fake/model' does not support vision" in result.stderr, result.stderr

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    skipped = summary.get("skipped", {})
    assert skipped.get("vision_unsupported", 0) >= 1, f"vision_unsupported not counted: {skipped}"

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "the plain prompt did not run"
    assert all(r["category"] != "vision" for r in records), [r["category"] for r in records]


# ---------------------------------------------------------------------------
# E2E-041: Capability-mismatch skip while other categories run
# ---------------------------------------------------------------------------


def test_e2e_041_mixed_library_skips_and_runs(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A mixed library runs the compatible categories and skips the rest.

    FR-037/FR-039: with a model declaring neither tools nor vision and a mixed
    library (coding, tool-use, vision, synthesis), ``raw.jsonl`` contains records
    only with ``category`` in {coding, synthesis}; ``summary['skipped']`` lists
    both ``tools_unsupported`` and ``vision_unsupported`` counts; the run
    completes; exit 0.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = _write_capable_config(cfg_base, tmp_path, port, supports_tools=False, supports_vision=False)
    prompts = _write_prompts(
        tmp_path / "library.yaml",
        [
            _plain_prompt("coding-001", category="coding"),
            _tool_prompt(
                "tool-001",
                "mock_web_search",
                _WEB_SEARCH_RESULT,
                description="Search.",
            ),
            _vision_prompt("vision-001"),
            _plain_prompt("synthesis-001", category="synthesis"),
        ],
    )

    out_dir = tmp_path / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--prompts", str(prompts), "--out", str(out_dir)],
    )
    assert result.exit_code == 0, result.stderr

    records = _read_jsonl(out_dir / "raw.jsonl")
    assert records, "no records written for the compatible categories"
    categories = {r["category"] for r in records}
    assert categories <= {"coding", "synthesis"}, f"incompatible category ran: {categories}"

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    skipped = summary.get("skipped", {})
    assert "tools_unsupported" in skipped, f"missing tools_unsupported counter: {skipped}"
    assert "vision_unsupported" in skipped, f"missing vision_unsupported counter: {skipped}"
    assert skipped["tools_unsupported"] >= 1, skipped
    assert skipped["vision_unsupported"] >= 1, skipped


# ---------------------------------------------------------------------------
# E2E-042: Tool result mock deterministic across runs
# ---------------------------------------------------------------------------


def test_e2e_042_tool_result_deterministic_across_runs(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The mock tool handler payload is byte-identical across two seeded runs.

    FR-038/FR-033: running the pptx tool prompt twice with ``seed:42`` yields a
    byte-identical mock handler payload; no randomness or network in the tool
    path.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    prompts = _write_prompts(
        tmp_path / "tool_pptx.yaml",
        [
            _tool_prompt(
                "pptx-tool-001",
                "mock_generate_pptx",
                _PPTX_RESULT,
                description="Generate a 3-slide deck.",
            )
        ],
    )

    payloads: list[bytes] = []
    for name in ("a", "b"):
        config = _write_capable_config(
            cfg_base,
            tmp_path,
            port,
            supports_tools=True,
            supports_vision=False,
            run_overrides={"seed": 42},
        )
        out_dir = tmp_path / name
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(config),
                "--model",
                "sut",
                "--prompts",
                str(prompts),
                "--seed",
                "42",
                "--out",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.stderr
        handler_results = _read_jsonl(out_dir / "tool_calls.jsonl")
        pptx_calls = [c for c in handler_results if c.get("tool") == "mock_generate_pptx"]
        assert pptx_calls, f"no pptx handler call recorded in run {name}"
        # Serialize the returned payload deterministically for a byte comparison.
        payloads.append(json.dumps(pptx_calls[0]["result"], sort_keys=True).encode("utf-8"))

    assert payloads[0] == payloads[1], (
        f"mock tool handler payload not byte-identical across runs: {payloads[0]!r} != {payloads[1]!r}"
    )
    assert payloads[0] == json.dumps(_PPTX_RESULT, sort_keys=True).encode("utf-8"), (
        "the deterministic payload does not match the declared constant"
    )
