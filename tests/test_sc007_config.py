"""Acceptance tests for SC-007: configuration loading + validation.

These cover the Configuration bounded context: loading ``config.yaml``,
validating it against the typed (Pydantic) schema, resolving ``$ENV:VAR`` and
``${VAR}`` interpolation, persisting a resolved snapshot with secrets redacted,
and the config-stage error paths (missing env var, malformed YAML, embedding
eval without a threshold).

The behavior is driven through the ``llm-bench`` CLI (``run`` subcommand) using
Typer's :class:`CliRunner`, against the offline FakeSUT harness from
``conftest.py``. Scenario SC-007 is exercised via
``llm-bench run --config <cfg> --model sut --dry-run`` (validate only; ``--dry-run``
never contacts the SUT) and via a short normal run that writes the resolved
snapshot artifacts (``resolved_config.json`` and ``env_snapshot.json``).

One test per E2E id: E2E-049..E2E-055 and the E2E-099 security check. Each
asserts exactly what the matching Gherkin in Section 12.2 specifies (CLI exit
codes, exact stderr strings, and resolved-snapshot field values/types).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from typer.testing import CliRunner

from llm_bench.llm_bench import app
from llm_bench.prompts import builtin_library, load_prompts

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

    from tests.conftest import SUTController

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_of(base_url: str) -> int:
    """Extract the TCP port from a FakeSUT ``base_url`` (``.../v1``)."""
    port = urlparse(base_url).port
    assert port is not None, base_url
    return port


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact, asserting it exists first for a clear failure."""
    assert path.exists(), f"expected artifact missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _model_entry(cfg: dict[str, Any], name: str = "sut") -> dict[str, Any]:
    """Return the named model registry entry from a resolved config snapshot."""
    models = cfg.get("models", [])
    for entry in models:
        if entry.get("name") == name:
            return entry
    raise AssertionError(f"model entry {name!r} not found in {models!r}")


# ---------------------------------------------------------------------------
# E2E-049: Config load + $ENV resolution success
# ---------------------------------------------------------------------------


def test_e2e_049_env_resolution_success(
    fake_sut: tuple[str, object],
    cfg_base: Callable[..., Path],
) -> None:
    """``--dry-run`` resolves ``$ENV:`` / ``${...}`` tokens and redacts the key.

    FR-001/FR-002/FR-004: exit 0, the resolved ``base_url`` carries the real
    port, no ``$ENV:`` token survives anywhere in the output, and the API key is
    printed redacted as ``***`` (never the literal ``sk-test``).
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(port, api_key="sk-test")

    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--dry-run"],
    )

    assert result.exit_code == 0, result.stderr
    out = result.stdout
    # Resolved base_url shows the interpolated ${SUT_PORT}.
    assert f"http://127.0.0.1:{port}/v1" in out
    # No interpolation token survives resolution.
    assert "$ENV:" not in out
    assert "${" not in out
    # The api_key is resolved internally but shown redacted, never in clear.
    assert "***" in out
    assert "sk-test" not in out


# ---------------------------------------------------------------------------
# E2E-050: Missing $ENV var aborts naming the variable
# ---------------------------------------------------------------------------


def test_e2e_050_missing_env_var_aborts(
    fake_sut: tuple[str, object],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """An unset ``$ENV:MISSING_KEY`` aborts with exit 2 and an exact message.

    FR-002: stderr contains exactly ``environment variable not set: MISSING_KEY``
    and no run data is written.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    # Write a base config, then point api_key at an unset variable.
    config = cfg_base(port)
    monkeypatch.delenv("MISSING_KEY", raising=False)
    text = config.read_text(encoding="utf-8")
    text = text.replace("api_key: $ENV:SUT_API_KEY", "api_key: $ENV:MISSING_KEY")
    config.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "runs" / "r_missing"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )

    assert result.exit_code == 2, result.output
    assert "environment variable not set: MISSING_KEY" in result.stderr
    # No run data written on a config-stage abort.
    assert not out_dir.exists() or not any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# E2E-051: Invalid YAML aborts with parse error
# ---------------------------------------------------------------------------


def test_e2e_051_malformed_yaml_aborts(tmp_path: Path) -> None:
    """Malformed YAML aborts non-zero with ``invalid config YAML`` + position.

    FR-001: stderr names the parse problem and a line/column; no run data.
    """
    config = tmp_path / "config.yaml"
    config.write_text("models: [ - name:: ::oops\n", encoding="utf-8")
    out_dir = tmp_path / "runs" / "r_badyaml"

    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )

    assert result.exit_code != 0, result.output
    assert "invalid config YAML" in result.stderr
    # A line/column reference is part of the message.
    assert "line" in result.stderr or "column" in result.stderr
    assert not out_dir.exists() or not any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# E2E-052: Model registry fields parsed with correct types
# ---------------------------------------------------------------------------


def test_e2e_052_registry_fields_parsed(
    fake_sut: tuple[str, object],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A full registry entry round-trips into ``resolved_config.json`` typed.

    FR-004/FR-051: ``--dry-run`` exits 0 and the snapshot records
    ``supports_vision`` as a bool, ``price_input`` as a float, and ``tokenizer``
    as the configured string.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(port)

    # Enrich the sut model entry with the full optional field set.
    text = config.read_text(encoding="utf-8")
    text = text.replace(
        "    supports_vision: false\n    supports_tools: false\n",
        (
            "    supports_vision: true\n"
            "    supports_tools: true\n"
            "    tokenizer: openai/gpt-oss-120b\n"
            "    price_input: 0.5\n"
            "    price_output: 1.5\n"
        ),
    )
    config.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "runs" / "r_fields"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--dry-run", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.stderr
    resolved = _load_json(out_dir / "resolved_config.json")
    entry = _model_entry(resolved, "sut")
    assert entry["supports_vision"] is True
    assert entry["supports_tools"] is True
    assert entry["tokenizer"] == "openai/gpt-oss-120b"
    assert isinstance(entry["price_input"], float)
    assert entry["price_input"] == 0.5
    assert isinstance(entry["price_output"], float)
    assert entry["price_output"] == 1.5


# ---------------------------------------------------------------------------
# E2E-053: api_key omitted is allowed
# ---------------------------------------------------------------------------


def test_e2e_053_api_key_omitted_allowed(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A model entry without ``api_key`` runs and sends no Authorization header.

    FR-004/FR-032: a short bench completes (exit 0), FakeSUT records no
    ``Authorization`` header on any request, and the resolved config shows
    ``api_key: null``.
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    # Short run: one quick level so the bench finishes promptly.
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
    # Remove the api_key line entirely.
    text = config.read_text(encoding="utf-8")
    text = text.replace("    api_key: $ENV:SUT_API_KEY\n", "")
    config.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "runs" / "r_noauth"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.stderr
    # No Authorization header on any captured request.
    for recorded in controller.requests:
        assert "Authorization" not in recorded.headers
        assert "authorization" not in {k.lower() for k in recorded.headers}
    resolved = _load_json(out_dir / "resolved_config.json")
    entry = _model_entry(resolved, "sut")
    assert entry.get("api_key") is None


# ---------------------------------------------------------------------------
# E2E-054: Resolved config + env snapshot persisted
# ---------------------------------------------------------------------------


def test_e2e_054_snapshots_persisted(
    fake_sut: tuple[str, object],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A normal run writes ``resolved_config.json`` and ``env_snapshot.json``.

    FR-051/FR-057: the resolved config captures run params (``seed:42``,
    ``model``, ``duration``, ``concurrency_levels``); the env snapshot captures
    tool version, timestamp, and python version; neither file contains the
    literal token ``$ENV:``.
    """
    base_url, _controller = fake_sut
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

    out_dir = tmp_path / "runs" / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.stderr

    resolved_path = out_dir / "resolved_config.json"
    env_path = out_dir / "env_snapshot.json"
    resolved = _load_json(resolved_path)
    env_snapshot = _load_json(env_path)

    # Resolved run params captured.
    run_block = resolved.get("run", resolved)
    assert run_block.get("seed") == 42
    assert run_block.get("concurrency_levels") == [1]
    assert "duration" in run_block
    entry = _model_entry(resolved, "sut")
    assert entry.get("model") == "fake/model"

    # Env snapshot captures tool version, timestamp, python version.
    keys = {k.lower() for k in env_snapshot}
    assert any("version" in k for k in keys)
    assert any("python" in k for k in keys)
    assert any("time" in k or "timestamp" in k or "date" in k for k in keys)

    # No interpolation token persisted in either snapshot.
    assert "$ENV:" not in resolved_path.read_text(encoding="utf-8")
    assert "$ENV:" not in env_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# E2E-055: Embedding eval missing threshold (config-level)
# ---------------------------------------------------------------------------


def test_e2e_055_embedding_missing_threshold(
    fake_sut: tuple[str, SUTController],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Embedding eval without a threshold aborts at the config stage.

    FR-003: non-zero exit; stderr contains
    ``embedding evaluation requires evaluation.embedding.threshold``; the SUT is
    never contacted (validation fails before any load).
    """
    base_url, controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(port)

    # Append an evaluation.embedding block lacking `threshold`.
    text = config.read_text(encoding="utf-8")
    text += (
        f"evaluation:\n  method: embedding\n  embedding:\n    url: http://127.0.0.1:{port}/v1\n    model: fake-embed\n"
    )
    config.write_text(text, encoding="utf-8")

    out_dir = tmp_path / "runs" / "r_nothresh"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--dry-run", "--out", str(out_dir)],
    )

    assert result.exit_code != 0, result.output
    assert "embedding evaluation requires evaluation.embedding.threshold" in result.stderr
    # Validation fails before any SUT contact.
    assert controller.request_count == 0


# ---------------------------------------------------------------------------
# E2E-099: API key never in the resolved-config snapshot
# ---------------------------------------------------------------------------


def test_e2e_099_api_key_never_persisted(
    fake_sut: tuple[str, object],
    cfg_base: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The resolved snapshot never contains the secret API key value.

    FR-057/FR-051: ``resolved_config.json`` contains the model entry but its
    ``api_key`` is ``"***"``/null/absent; grepping for ``sk-test`` returns zero
    matches; non-secret fields (model, base_url) remain present.
    """
    base_url, _controller = fake_sut
    port = _port_of(base_url)
    config = cfg_base(
        port,
        api_key="sk-test",
        run_overrides={
            "duration": "0.3s",
            "warmup": "0.05s",
            "cooldown": "0.05s",
            "concurrency_levels": [1],
            "min_samples": 1,
        },
    )

    out_dir = tmp_path / "runs" / "r1"
    result = runner.invoke(
        app,
        ["run", "--config", str(config), "--model", "sut", "--out", str(out_dir)],
    )

    assert result.exit_code == 0, result.stderr
    resolved_path = out_dir / "resolved_config.json"
    raw = resolved_path.read_text(encoding="utf-8")
    # The secret never appears anywhere in the snapshot.
    assert raw.count("sk-test") == 0
    resolved = _load_json(resolved_path)
    entry = _model_entry(resolved, "sut")
    # api_key redacted, null, or absent.
    assert entry.get("api_key") in (None, "***")
    # Non-secret fields present.
    assert entry.get("model") == "fake/model"
    assert f"http://127.0.0.1:{port}/v1" in entry.get("base_url", "")


# ---------------------------------------------------------------------------
# models command: list the registry (no endpoint contacted)
# ---------------------------------------------------------------------------


def test_models_command_lists_registry_with_redacted_keys(
    cfg_base: Callable[..., Path],
) -> None:
    """`models` lists each registry entry, resolves $ENV, and never leaks the key."""
    # Port only feeds the rendered base_url; `models` contacts no endpoint.
    config = cfg_base(9999)
    result = runner.invoke(app, ["models", "--config", str(config)])

    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert "1 model(s) registered" in out
    assert "* sut" in out  # the default registry entry, marked
    assert "fake/model" in out  # the model id
    assert "http://127.0.0.1:9999/v1" in out  # resolved base_url
    assert "key:***" in out  # api_key redacted
    assert "sk-test" not in out  # the real key never appears in clear


# ---------------------------------------------------------------------------
# init command: scaffold ~/.config/llm-bench/ + the runs dir
# ---------------------------------------------------------------------------


def test_init_command_scaffolds_dirs_and_starter_config(tmp_path: Path) -> None:
    """`init` creates config.yaml and a prompts/ dir with short.yaml + long.yaml."""
    # The autouse guard redirects the defaults to tmp_path/{default-config,default-runs}.
    config_dir = tmp_path / "default-config"
    runs_dir = tmp_path / "default-runs"
    config_file = config_dir / "config.yaml"
    short_file = config_dir / "prompts" / "short.yaml"
    long_file = config_dir / "prompts" / "long.yaml"
    dashboard_file = config_dir / "dashboards" / "default.yaml"
    assert not config_file.exists()

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.stderr
    assert config_dir.is_dir()
    assert runs_dir.is_dir()
    assert config_file.is_file()
    assert "models:" in config_file.read_text(encoding="utf-8")
    assert f"created: {config_file}" in result.stdout

    # prompts/short.yaml mirrors the built-in library and re-loads cleanly.
    assert short_file.is_file()
    assert f"created: {short_file}" in result.stdout
    library = load_prompts(short_file)
    assert len(library.prompts) == len(builtin_library().prompts)

    # prompts/long.yaml re-loads and every entry is an instruction-heavy long prompt.
    assert long_file.is_file()
    long_library = load_prompts(long_file)
    assert long_library.prompts
    assert all(prompt.isl_bucket == "long" for prompt in long_library.prompts)

    # prompts/quality.yaml re-loads and every entry carries an expected_output so the
    # quality eval scores (almost) every request instead of skipping most.
    quality_file = config_dir / "prompts" / "quality.yaml"
    assert quality_file.is_file()
    assert f"created: {quality_file}" in result.stdout
    quality_library = load_prompts(quality_file)
    assert quality_library.prompts
    assert all(prompt.expected_output for prompt in quality_library.prompts)

    # prompts/code-quality.yaml is the code-focused variant: coding tasks, all with
    # a canonical expected_output.
    code_quality_file = config_dir / "prompts" / "code-quality.yaml"
    assert code_quality_file.is_file()
    code_quality_library = load_prompts(code_quality_file)
    assert code_quality_library.prompts
    assert all(prompt.expected_output for prompt in code_quality_library.prompts)
    assert all(prompt.category == "coding" for prompt in code_quality_library.prompts)

    # A starter dashboards/default.yaml is scaffolded and parses.
    assert dashboard_file.is_file()
    from llm_bench.dashboards import parse_dashboard  # noqa: PLC0415

    assert parse_dashboard(dashboard_file.read_text(encoding="utf-8"))

    # Idempotent: a second init keeps existing (user-edited) files untouched.
    config_file.write_text("models: [custom]\n", encoding="utf-8")
    short_file.write_text("- {id: mine, messages: [{role: user, content: hi}]}\n", encoding="utf-8")
    result2 = runner.invoke(app, ["init"])
    assert result2.exit_code == 0, result2.stderr
    assert config_file.read_text(encoding="utf-8") == "models: [custom]\n"
    assert "mine" in short_file.read_text(encoding="utf-8")
    assert "kept existing" in result2.stdout
