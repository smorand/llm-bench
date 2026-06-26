"""Application settings and benchmark configuration loading.

This module provides two distinct concerns:

* :class:`Settings`: process-level settings via ``pydantic-settings`` (the
  ``LLM_BENCH_`` environment prefix).
* The benchmark configuration schema (model registry, run parameters, SLO
  profiles, evaluation config) plus ``$ENV:VAR`` / ``${VAR}`` interpolation,
  validation, and the resolved/env snapshot writers.

The benchmark config is loaded from a ``config.yaml`` file. String interpolation
tokens (``$ENV:NAME`` and ``${NAME}``) are resolved against the process
environment at load time; a missing variable aborts the load. Secrets (API keys)
are redacted in every persisted snapshot and never written in clear.
"""

from __future__ import annotations

import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_bench.version import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDACTED: str = "***"
MISSING_ENV_EXIT_CODE: int = 2

# Default config locations (used when no path is given on the CLI).
DEFAULT_CONFIG_DIR: Path = Path.home() / ".config" / "llm-bench"
DEFAULT_CONFIG_FILE: Path = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_PROMPTS_FILE: Path = DEFAULT_CONFIG_DIR / "prompts.yaml"

# Default results location (XDG data home) used when --out is omitted.
DEFAULT_RUNS_DIR: Path = Path.home() / ".local" / "share" / "llm-bench" / "runs"


def slugify(text: str) -> str:
    """Reduce ``text`` to a filesystem-safe token (alnum, dot, dash, underscore)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return cleaned or "run"


def default_run_dir(label: str | None = None) -> Path:
    """Return a fresh timestamped run directory under the data home (FR-048..051).

    Used when ``--out`` is omitted so a bare ``run`` still persists its artifacts.
    When ``label`` is given (e.g. the benchmarked model name) it is appended to
    the timestamp so runs are identifiable at a glance: ``<stamp>_<label>``.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{stamp}_{slugify(label)}" if label else stamp
    return DEFAULT_RUNS_DIR / name


def default_runs_dir() -> Path:
    """Return the root runs directory, resolved at call time (test-patchable)."""
    return DEFAULT_RUNS_DIR


def default_config_file() -> Path:
    """Return the default ``config.yaml`` path, resolved at call time.

    Reading :data:`DEFAULT_CONFIG_DIR` here (rather than the frozen
    :data:`DEFAULT_CONFIG_FILE`) keeps the lookup honest under test redirection.
    """
    return DEFAULT_CONFIG_DIR / "config.yaml"


def default_prompts_dir() -> Path:
    """Return the prompts library directory (``~/.config/llm-bench/prompts``)."""
    return DEFAULT_CONFIG_DIR / "prompts"


def default_dashboards_dir() -> Path:
    """Return the dashboards directory (``~/.config/llm-bench/dashboards``)."""
    return DEFAULT_CONFIG_DIR / "dashboards"


def default_prompts_file() -> Path:
    """Return the legacy single ``prompts.yaml`` path (pre-prompts/ layout)."""
    return DEFAULT_CONFIG_DIR / "prompts.yaml"


def default_prompts_path() -> Path | None:
    """Return the prompts file a bare ``run`` should use, or ``None`` for built-in.

    Prefers the new ``prompts/short.yaml`` layout, then the legacy top-level
    ``prompts.yaml``; returns ``None`` when neither exists so the built-in library
    is used.
    """
    short = default_prompts_dir() / "short.yaml"
    if short.is_file():
        return short
    legacy = default_prompts_file()
    if legacy.is_file():
        return legacy
    return None


# Starter config written by ``llm-bench init`` when no config.yaml exists yet.
STARTER_CONFIG: str = """\
# llm-bench config. Secrets are referenced via $ENV: and never stored in clear.
# `--model <name>` selects an entry; the first one is the default.
models:
  - name: ibm-haiku
    base_url: $ENV:IBM_ICA_BASE_URL   # IBM ICA gateway, OpenAI-compatible
    model: claude-haiku-4-5           # exact id the ICA team is entitled to (no 'ibm/' prefix)
    api_key: $ENV:IBM_ICA_API_KEY
    supports_vision: true
    supports_tools: true
  - name: ei-mistral-medium35
    base_url: $ENV:EI_MODEL_MISTRAL_MEDIUM35_URL
    model: mistral-medium-3.5         # set to the exact model id your gateway expects
    api_key: $ENV:EI_MODEL_MISTRAL_MEDIUM35_API_KEY
    supports_vision: false
    supports_tools: true
  - name: ei-qwen36
    base_url: $ENV:EI_MODEL_QWEN36_URL
    model: qwen-3.6                    # set to the exact model id your gateway expects
    api_key: $ENV:EI_MODEL_QWEN36_API_KEY
    supports_vision: false
    supports_tools: true
  - name: local
    base_url: http://localhost:8080/v1   # llama.cpp server (no auth)
    model: local-model
    supports_vision: false
    supports_tools: false

run:
  mode: closed
  duration: 30s
  warmup: 5s
  cooldown: 5s
  min_samples: 30
  concurrency_levels: [1, 2, 4]
  max_tokens: 128
  ignore_eos: false
  temperature: 0.0
  cache_busting: true
  timeout: 60s
  seed: 0
  slo_profile: interactive

slo_profiles:
  interactive: { ttft_ms: 500, tpot_ms: 50, e2e_ms: 5000 }
  relaxed:     { ttft_ms: 2000, tpot_ms: 200, e2e_ms: 30000 }

evaluation:
  method: none                        # none | embedding | judge (or pick via --eval-method)
  global_timeout: 60s
  embedding:
    url: http://localhost:8001/v1     # omit to use a local embedding model
    model: text-embedding-3-small
    threshold: 0.80                   # mandatory for the embedding method
    rate_limit: 20
  judge:
    rubric: three_level
    model:
      url: $ENV:IBM_ICA_BASE_URL
      api_key: $ENV:IBM_ICA_API_KEY
      model: claude-haiku-4-5           # exact id the ICA team is entitled to (no 'ibm/' prefix)
"""


# Instruction-heavy, long-input prompt set written by ``llm-bench init`` to
# ``prompts_long.yaml``. Select it with
# ``--prompts ~/.config/llm-bench/prompts/long.yaml`` to stress time-to-first-token
# and prefill behaviour under large prompts. Edit, trim, or extend freely.
STARTER_PROMPTS_LONG: str = """\
# Long, instruction-dense prompt library written by 'llm-bench init'.
# Use it to benchmark large-prefill / heavy-instruction behaviour:
#   llm-bench run -m <model> --prompts ~/.config/llm-bench/prompts/long.yaml
# Each entry follows the same schema as prompts.yaml (id, category, isl_bucket,
# messages; optional expected_output, tools, tool_results). All are isl_bucket: long.
- id: long-coding-rate-limiter
  category: coding
  isl_bucket: long
  messages:
    - role: system
      content: |
        You are a meticulous staff-level Python engineer. You write production-grade,
        fully typed, asyncio-native code. You never use blocking calls inside coroutines,
        never use mutable default arguments, never swallow exceptions with a bare except,
        and you document every public function with a Google-style docstring. You favour
        small, composable, dependency-injected units and you justify every non-obvious
        decision in a short comment.
    - role: user
      content: |
        Implement an asynchronous token-bucket rate limiter as a class `AsyncTokenBucket`.

        Requirements:
        1. Constructor takes `rate` (tokens added per second, float > 0) and `capacity`
           (max tokens, int > 0). Validate both and raise `ValueError` on bad input.
        2. Expose `async def acquire(self, tokens: int = 1) -> None` that waits until
           enough tokens are available, then consumes them. It must be fair under
           concurrency (no starvation) and must not busy-wait.
        3. Expose `def try_acquire(self, tokens: int = 1) -> bool` that consumes
           immediately if possible and returns whether it succeeded, never blocking.
        4. Refill must be computed lazily from a monotonic clock (`time.monotonic`),
           never from a background task, so the limiter holds no timers.
        5. Be safe for use from many coroutines on one event loop; protect shared state
           with an `asyncio.Lock`.
        6. Provide a `@property tokens` returning the current (lazily refilled) count.

        Deliver: the full implementation, then three short usage examples (single
        consumer, burst, and many concurrent consumers), then a bullet list of the
        edge cases you handled. Keep explanations terse; let the code carry the weight.
  expected_output: |
        A correct, fully typed AsyncTokenBucket with lazy monotonic refill, an async
        acquire that awaits without busy-waiting, a non-blocking try_acquire, lock-guarded
        state, a tokens property, usage examples, and an edge-case list.
- id: long-synthesis-exec-brief
  category: synthesis
  isl_bucket: long
  messages:
    - role: system
      content: |
        You are an executive analyst. You compress noisy input into decision-ready
        briefs. You are precise, neutral, and you never invent facts not present in the
        source. When the source is ambiguous, you say so explicitly.
    - role: user
      content: |
        Read the background below and produce an executive brief.

        BACKGROUND:
        Over the last two quarters our inference platform served a growing fleet of
        OpenAI-compatible endpoints. Latency was acceptable at low concurrency but the
        p99 time-to-first-token degraded sharply past ~200 concurrent streams, while
        median throughput kept rising, which masked the tail in dashboards. Cost per
        million output tokens fell 18% after a model swap, but goodput (requests meeting
        the 300ms TTFT / 50ms per-token SLO) actually dropped because the cheaper model
        had higher variance. Two regional gateways showed event-loop lag spikes
        correlated with large multimodal prompts. The team disagrees on whether to cap
        concurrency, switch to an open-loop arrival model for testing, or invest in
        speculative decoding first.

        Produce, in this exact structure and nothing else:
        - TL;DR: one sentence.
        - Key findings: exactly 4 bullets, each <= 20 words.
        - Risks: exactly 3 bullets.
        - Recommendation: one short paragraph naming a single first action and why.
        - Open questions: 2 bullets.

        Do not add headers, preamble, or closing remarks beyond that structure.
  expected_output: |
        A structured brief with a one-sentence TL;DR, four <=20-word findings, three
        risks, a single-action recommendation with rationale, and two open questions,
        grounded only in the supplied background.
- id: long-instruction-following
  category: general
  isl_bucket: long
  messages:
    - role: user
      content: |
        Follow every rule below exactly. Rule compliance matters more than content.

        Persona: you are a terse railway dispatcher giving track-change notices.
        Task: write notices for 3 trains (IDs T101, T204, T377) changing platforms.

        Rules:
        1. Output exactly 3 notices, one per train, in ascending ID order.
        2. Each notice is exactly two lines.
        3. Line 1 format: "[<ID>] platform <old> -> <new>" with single spaces.
        4. Line 2 is a single imperative sentence of at most 12 words.
        5. Never use the words "please", "sorry", or any exclamation mark.
        6. Old/new platforms are integers 1-12 and old != new for each train.
        7. No two trains may share the same new platform.
        8. Do not output anything before the first notice or after the last.
        9. Separate the three notices with exactly one blank line.
        10. Do not use markdown, bullets, numbering, or quotation marks.

        Choose plausible platform numbers yourself that satisfy rules 6-7.
  expected_output: |
        Three two-line notices for T101, T204, T377 in ID order, each matching the
        line-1 format and a <=12-word imperative line 2, distinct new platforms, no
        forbidden words, one blank line between notices, nothing else.
- id: long-data-extraction-json
  category: general
  isl_bucket: long
  messages:
    - role: system
      content: |
        You are a strict information-extraction engine. You output only valid JSON that
        conforms to the requested schema. You never include prose, code fences, or
        trailing commentary. Missing values are represented as null, never guessed.
    - role: user
      content: |
        Extract structured data from this messy customer message into JSON.

        MESSAGE:
        "hey so this is Marie Dubois writing again about order number 88-23104, the one I
        placed maybe the 3rd of last month. Two of the three items showed up (the blue
        kettle and the cast-iron pan) but the third, some ceramic mugs, never arrived. I'd
        like a refund just for the mugs if possible, otherwise resend. You can reach me on
        +33 6 12 34 56 78 or marie.d@example.org, I prefer email. Not in a huge rush."

        Output JSON with exactly these keys:
        - customer_name (string)
        - order_id (string)
        - items: array of objects {name (string), received (boolean)}
        - issue (one of: "missing_item", "damaged", "wrong_item", "other")
        - requested_resolution (one of: "refund", "resend", "either", "unknown")
        - contact: object {email (string|null), phone (string|null), preferred (one of
          "email","phone","unknown")}
        - urgency (one of "low","medium","high")

        Output only the JSON object.
  expected_output: |
        A single valid JSON object with the requested keys, three items with correct
        received flags, issue "missing_item", resolution "either", email/phone filled,
        preferred "email", urgency "low".
- id: long-refactor-review
  category: coding
  isl_bucket: long
  messages:
    - role: system
      content: |
        You are a senior reviewer. You give actionable, prioritized feedback. You quote
        the exact line you mean, classify each issue by severity, and you propose a
        concrete fix, not a vague suggestion.
    - role: user
      content: |
        Review this Python function against the checklist and report findings.

        CODE:
        def load(cfg={}, items=[]):
            import json, os
            f = open(os.environ['CFG'])
            data = json.loads(f.read())
            for k in data:
                try:
                    items.append(data[k])
                except:
                    pass
            print('loaded', len(items))
            return items

        CHECKLIST (report each as PASS or FAIL with a one-line reason):
        - C1: no mutable default arguments
        - C2: files closed deterministically (context manager)
        - C3: no bare except
        - C4: no print in library code (use logging)
        - C5: imports at module top, not inside the function
        - C6: input/config injected, not read from os.environ directly
        - C7: function has a type signature and docstring

        Then output a corrected version of the function that passes every item.
        Order your findings by severity (highest first).
  expected_output: |
        Per-item PASS/FAIL verdicts (most fail: C1,C2,C3,C4,C5,C6,C7), severity-ordered,
        each with a concrete fix, followed by a typed, docstringed, context-managed,
        logging-based, dependency-injected rewrite.
"""


def scaffold_config() -> tuple[list[Path], list[Path]]:
    """Create the default config + runs directories and starter config/prompt files.

    Writes a starter ``config.yaml`` and a ``prompts/`` directory holding
    ``short.yaml`` (a mirror of the built-in library) and ``long.yaml``
    (instruction-heavy long-input prompts). Idempotent: existing directories and
    files are left untouched. Returns ``(created, skipped)`` paths to report.
    """
    # Local imports keep these modules out of config's import-time graph.
    from llm_bench.dashboards import STARTER_DASHBOARD  # noqa: PLC0415
    from llm_bench.prompts import export_builtin_prompts_yaml  # noqa: PLC0415

    # Derive paths from the (test-patchable) directory, not the frozen DEFAULT_*
    # module constants, so the conftest redirection holds.
    prompts_dir = DEFAULT_CONFIG_DIR / "prompts"
    dashboards_dir = DEFAULT_CONFIG_DIR / "dashboards"
    created: list[Path] = []
    skipped: list[Path] = []
    for directory in (DEFAULT_CONFIG_DIR, prompts_dir, dashboards_dir, DEFAULT_RUNS_DIR):
        if directory.exists():
            skipped.append(directory)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    for target, content in (
        (DEFAULT_CONFIG_DIR / "config.yaml", STARTER_CONFIG),
        (prompts_dir / "short.yaml", export_builtin_prompts_yaml()),
        (prompts_dir / "long.yaml", STARTER_PROMPTS_LONG),
        (dashboards_dir / "default.yaml", STARTER_DASHBOARD),
    ):
        if target.exists():
            skipped.append(target)
        else:
            target.write_text(content, encoding="utf-8")
            created.append(target)
    return created, skipped


_ENV_PREFIX = "$ENV:"
_ENV_PREFIX_RE = re.compile(r"^\$ENV:([A-Za-z_][A-Za-z0-9_]*)$")
_BRACED_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Fields whose value must never be persisted in clear in any snapshot.
_SECRET_FIELDS: frozenset[str] = frozenset({"api_key"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Base error for configuration loading and validation failures."""


class ConfigYamlError(ConfigError):
    """Raised when the config file is not valid YAML."""


class MissingEnvVarError(ConfigError):
    """Raised when a ``$ENV:`` / ``${...}`` reference names an unset variable.

    Args:
        variable: Name of the environment variable that is not set.
    """

    __slots__ = ("variable",)

    def __init__(self, variable: str) -> None:
        self.variable = variable
        super().__init__(f"environment variable not set: {variable}")


class ConfigValidationError(ConfigError):
    """Raised when the config is syntactically valid YAML but semantically wrong."""


# ---------------------------------------------------------------------------
# Process settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Process-level configuration loaded from the environment.

    Environment variables are prefixed with ``LLM_BENCH_`` (for example
    ``LLM_BENCH_DEBUG``). A ``.env`` file is loaded automatically when present.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_BENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "llm-bench"
    debug: bool = False


# ---------------------------------------------------------------------------
# Benchmark config schema
# ---------------------------------------------------------------------------


class ModelRegistryEntry(BaseModel):
    """One model registry entry (FR-004).

    Carries the endpoint coordinates and capability/pricing metadata for a single
    benchmarkable model. ``api_key`` and pricing fields are optional.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    model: str
    api_key: str | None = None
    tokenizer: str | None = None
    supports_vision: bool = False
    supports_tools: bool = False
    price_input: float | None = None
    price_output: float | None = None


class RunConfig(BaseModel):
    """Run parameters for a benchmark sweep."""

    model_config = ConfigDict(extra="allow")

    mode: Literal["closed", "open"] = "closed"
    duration: str = "2s"
    warmup: str = "0.5s"
    cooldown: str = "0.5s"
    min_samples: int = 30
    concurrency_levels: list[int] = Field(default_factory=lambda: [1])
    request_rates: list[float] = Field(default_factory=list)
    burstiness: float = 1.0
    max_outstanding: int = 1000
    max_tokens: int = 8
    ignore_eos: bool = True
    temperature: float = 0.0
    cache_busting: bool = True
    retries: int = 0
    timeout: str = "5s"
    seed: int = 0
    slo_profile: str = "interactive"


class JudgeModel(BaseModel):
    """Judge LLM coordinates for judge-based evaluation."""

    model_config = ConfigDict(extra="forbid")

    url: str
    model: str
    api_key: str | None = None
    prompt: str | None = None


class JudgeConfig(BaseModel):
    """Judge evaluation block.

    ``rubric`` selects the verdict vocabulary: ``binary`` yields ``pass`` /
    ``fail`` verdicts, ``three_level`` yields ``correct`` / ``partial`` /
    ``incorrect`` (FR-044). No numeric 1-10 scale is ever produced.
    """

    model_config = ConfigDict(extra="forbid")

    model: JudgeModel
    rubric: Literal["binary", "three_level"] = "binary"


class EmbeddingConfig(BaseModel):
    """Embedding evaluation block.

    ``threshold`` is optional at the schema level so that a clear, dedicated
    error message can be raised when the embedding method is active without one
    (FR-003).
    """

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    model: str
    api_key: str | None = None
    threshold: float | None = None
    rate_limit: float | None = None


class EvaluationConfig(BaseModel):
    """Evaluation configuration (judge and/or embedding)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["judge", "embedding", "none"] = "none"
    judge: JudgeConfig | None = None
    embedding: EmbeddingConfig | None = None
    global_timeout: str | None = None


class SloProfile(BaseModel):
    """A single SLO profile (latency thresholds in milliseconds).

    The three thresholds bound TTFT, per-output-token latency (TPOT), and the
    end-to-end latency of a request; a request counts toward goodput only when it
    meets all three (FR-029/FR-030). Extra keys are tolerated so future
    thresholds do not break loading.
    """

    model_config = ConfigDict(extra="allow")

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None


# Built-in SLO profiles (FR-030); used when the config omits ``slo_profiles``.
_DEFAULT_SLO_PROFILES: dict[str, dict[str, float]] = {
    "interactive": {"ttft_ms": 500.0, "tpot_ms": 50.0, "e2e_ms": 5000.0},
    "relaxed": {"ttft_ms": 2000.0, "tpot_ms": 200.0, "e2e_ms": 30000.0},
}


class BenchConfig(BaseModel):
    """Top-level benchmark configuration (FR-001/FR-004)."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelRegistryEntry]
    run: RunConfig = Field(default_factory=RunConfig)
    evaluation: EvaluationConfig | None = None
    slo_profiles: dict[str, SloProfile] = Field(default_factory=dict)

    def model_entry(self, name: str | None) -> ModelRegistryEntry:
        """Return the registry entry by ``name`` (or the first entry if ``None``).

        Args:
            name: Registry entry name to select, or ``None`` for the first entry.

        Raises:
            ConfigValidationError: When the registry is empty or ``name`` is unknown.
        """
        if not self.models:
            raise ConfigValidationError("config defines no models")
        if name is None:
            return self.models[0]
        for entry in self.models:
            if entry.name == name:
                return entry
        known = ", ".join(entry.name for entry in self.models)
        raise ConfigValidationError(f"model not found in registry: {name} (known: {known})")

    def active_slo(self) -> dict[str, float | None]:
        """Resolve the active SLO thresholds for the selected profile (FR-030).

        Falls back to the built-in ``interactive``/``relaxed`` profiles when the
        config does not define ``slo_profiles``. An unknown profile name resolves
        to an empty (all-``None``) threshold block so goodput simply counts every
        steady success.

        Returns:
            A mapping with ``ttft_ms`` / ``tpot_ms`` / ``e2e_ms`` thresholds in
            milliseconds (any of which may be ``None`` when not configured).
        """
        name = self.run.slo_profile
        profile = self.slo_profiles.get(name)
        block = profile.model_dump() if profile is not None else dict(_DEFAULT_SLO_PROFILES.get(name, {}))
        return {
            "ttft_ms": _as_optional_float(block.get("ttft_ms")),
            "tpot_ms": _as_optional_float(block.get("tpot_ms")),
            "e2e_ms": _as_optional_float(block.get("e2e_ms")),
        }


def _as_optional_float(value: Any) -> float | None:
    """Coerce a value to ``float`` when present, preserving ``None``."""
    return None if value is None else float(value)


def apply_slo_overrides(config: BenchConfig, overrides: Mapping[str, float]) -> None:
    """Override the active SLO profile's thresholds in place (``--slo`` flag, FR-030).

    The overrides are merged onto the run's active profile so the resolved config
    snapshot and goodput computation both observe the overridden thresholds. A
    profile entry is materialised when the active profile is a built-in default.

    Args:
        config: The benchmark configuration to mutate.
        overrides: Threshold key/value pairs (for example ``{"ttft_ms": 300}``).
    """
    if not overrides:
        return
    name = config.run.slo_profile
    profile = config.slo_profiles.get(name)
    if profile is None:
        base = dict(_DEFAULT_SLO_PROFILES.get(name, {}))
        profile = SloProfile.model_validate(base)
        config.slo_profiles[name] = profile
    for key, value in overrides.items():
        setattr(profile, key, float(value))


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _resolve_string(value: str, env: Mapping[str, str]) -> str:
    """Resolve ``$ENV:NAME`` and ``${NAME}`` tokens in a single string.

    Args:
        value: Raw string that may contain interpolation tokens.
        env: Environment mapping to resolve against.

    Raises:
        MissingEnvVarError: When a referenced variable is not present in ``env``.
    """
    whole = _ENV_PREFIX_RE.match(value)
    if whole is not None:
        name = whole.group(1)
        if name not in env:
            raise MissingEnvVarError(name)
        return env[name]

    if value.startswith(_ENV_PREFIX):
        # ``$ENV:`` prefix without a clean identifier match is malformed input.
        name = value[len(_ENV_PREFIX) :]
        raise MissingEnvVarError(name)

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise MissingEnvVarError(name)
        return env[name]

    return _BRACED_RE.sub(_replace, value)


def _resolve_value(value: Any, env: Mapping[str, str]) -> Any:
    """Recursively resolve interpolation tokens in nested config data."""
    if isinstance(value, str):
        return _resolve_string(value, env)
    if isinstance(value, dict):
        return {key: _resolve_value(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, env) for item in value]
    return value


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_config(path: Path, env: Mapping[str, str]) -> BenchConfig:
    """Load, interpolate, and validate a benchmark config (FR-001..FR-004).

    Args:
        path: Path to the ``config.yaml`` file.
        env: Environment mapping used to resolve ``$ENV:`` / ``${...}`` tokens.

    Returns:
        A validated :class:`BenchConfig`.

    Raises:
        ConfigYamlError: When the file is not valid YAML.
        MissingEnvVarError: When an interpolation token names an unset variable.
        ConfigValidationError: When the resolved data fails schema or semantic
            validation (including the embedding-threshold rule, FR-003).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"config file not readable: {path}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigYamlError(f"invalid config YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigValidationError("invalid config YAML: top-level document must be a mapping")

    resolved = _resolve_value(raw, env)

    try:
        config = BenchConfig.model_validate(resolved)
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc

    _validate_semantics(config)
    return config


def _validate_semantics(config: BenchConfig) -> None:
    """Apply cross-field semantic rules not expressible in the schema (FR-003)."""
    evaluation = config.evaluation
    if evaluation is not None and evaluation.method == "embedding":
        embedding = evaluation.embedding
        if embedding is None or embedding.threshold is None:
            raise ConfigValidationError("embedding evaluation requires evaluation.embedding.threshold")


def apply_eval_method(config: BenchConfig, method: str | None) -> None:
    """Select the active evaluation method from the ``--eval-method`` flag (FR-003).

    Sets ``evaluation.method`` to ``embedding`` or ``judge`` and re-applies the
    config-stage semantic rules so an embedding method without a threshold aborts
    before any endpoint is contacted (FR-003). A ``None`` method leaves the config
    untouched (evaluation stays disabled).

    Args:
        config: The loaded benchmark configuration to mutate.
        method: The CLI-selected method (``embedding``/``judge``), or ``None``.

    Raises:
        ConfigValidationError: When the config carries no ``evaluation`` block for
            the requested method, or the embedding method lacks a threshold.
    """
    if method is None:
        return
    if config.evaluation is None:
        raise ConfigValidationError(f"--eval-method {method} requires an evaluation block in the config")
    # method is constrained to the literal set by the CLI choice.
    config.evaluation.method = method  # type: ignore[assignment]  # CLI restricts to embedding|judge
    _validate_semantics(config)


# ---------------------------------------------------------------------------
# Snapshots (FR-051 / FR-057)
# ---------------------------------------------------------------------------


def _redact(data: Any) -> Any:
    """Return a deep copy of ``data`` with every secret field redacted.

    Any value whose key is in :data:`_SECRET_FIELDS` is replaced: a present
    secret becomes :data:`REDACTED`; a ``None`` value stays ``None``. This
    guarantees the literal secret never appears in a persisted snapshot.
    """
    if isinstance(data, dict):
        result: dict[str, Any] = {}
        for key, value in data.items():
            if key in _SECRET_FIELDS:
                result[key] = REDACTED if value is not None else None
            else:
                result[key] = _redact(value)
        return result
    if isinstance(data, list):
        return [_redact(item) for item in data]
    return data


def resolved_config_snapshot(config: BenchConfig) -> dict[str, Any]:
    """Build the redacted ``resolved_config.json`` payload (FR-051/FR-057).

    The payload also carries a resolved ``slo`` block holding the *active*
    profile's thresholds (``ttft_ms``/``tpot_ms``/``e2e_ms``) so an operator can
    read the goodput targets that were actually applied (FR-030).
    """
    redacted = _redact(config.model_dump(mode="json"))
    if not isinstance(redacted, dict):  # pragma: no cover - model_dump always yields a mapping
        raise ConfigError("resolved config did not serialize to a mapping")
    redacted["slo"] = config.active_slo()
    return redacted


def env_snapshot() -> dict[str, Any]:
    """Build the ``env_snapshot.json`` payload (FR-051).

    Captures the tool version, an ISO-8601 UTC timestamp, and the Python version.
    No secrets or environment values are recorded.
    """
    return {
        "tool_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": sys.platform,
    }
