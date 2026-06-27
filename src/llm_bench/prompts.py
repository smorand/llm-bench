"""Prompt library, external override loading, and anti-cache prefixing.

This module implements the SC-001 PROMPTS + ANTI-CACHE layer (FR-033..037):

* a packaged, varied built-in prompt library spanning the spec categories
  (``coding``, ``synthesis``, ``tool-use``, ``vision``, ``general``),
* an external ``--prompts`` YAML loader (a top-level list of prompt mappings)
  that fully replaces the built-in set,
* a seeded, per-request random selector so prompt selection is reproducible for
  a given ``seed`` (FR-033), and
* a unique-prefix cache-buster that prepends a fresh UUID-derived marker to a
  request's user message when cache busting is enabled (FR-034).

Prompts never reach traces or logs; only their ``prompt_id``/``category``/bucket
metadata is recorded (FR-057).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# Valid prompt categories (Section 8 Data Model).
CATEGORIES: frozenset[str] = frozenset({"coding", "synthesis", "tool-use", "vision", "general"})

# Default input-length bucket when a prompt declares none.
_DEFAULT_ISL_BUCKET = "short"

_ROLE_USER = "user"


class PromptError(Exception):
    """Base error for prompt loading failures."""


class EmptyPromptSetError(PromptError):
    """Raised when an external prompt file yields no prompts (FR-036).

    Args:
        source: The prompt file path as supplied by the user, for the message.
    """

    __slots__ = ("source",)

    def __init__(self, source: str) -> None:
        self.source = source
        super().__init__(f"no prompts loaded from {source}")


@dataclass(frozen=True, slots=True)
class Prompt:
    """One library prompt and its tagging metadata (FR-036/FR-037/FR-038).

    Args:
        id: Stable identifier persisted as ``prompt_id``.
        category: One of :data:`CATEGORIES`.
        messages: OpenAI-style chat messages (each ``{role, content}``); the
            ``content`` may be a multimodal list carrying ``image_url`` parts.
        isl_bucket: Declared input-length bucket (``short``/``medium``/``long``).
        expected_output: Optional reference output for asynchronous evaluation.
        tools: OpenAI ``tools`` array passed verbatim in the request body for
            tool-use prompts (FR-038); empty when the prompt needs no tools.
        tool_results: Deterministic mock handler return constants keyed by tool
            name; the local handler returns these instead of calling out (FR-038).
    """

    id: str
    category: str
    messages: tuple[dict[str, Any], ...]
    isl_bucket: str = _DEFAULT_ISL_BUCKET
    expected_output: str | None = None
    tools: tuple[dict[str, Any], ...] = ()
    tool_results: tuple[tuple[str, Any], ...] = ()

    @property
    def requires_tools(self) -> bool:
        """True when the prompt needs the ``supports_tools`` capability (FR-039)."""
        return bool(self.tools) or self.category == "tool-use"

    @property
    def requires_vision(self) -> bool:
        """True when the prompt needs the ``supports_vision`` capability (FR-039)."""
        return self.category == "vision" or _has_image_part(self.messages)

    def tool_result_for(self, tool_name: str) -> Any:
        """Return the deterministic mock payload for ``tool_name`` (FR-038)."""
        for name, payload in self.tool_results:
            if name == tool_name:
                return payload
        return None


def _has_image_part(messages: tuple[dict[str, Any], ...]) -> bool:
    """Return true when any message carries a multimodal ``image_url`` part."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(isinstance(part, dict) and part.get("type") == "image_url" for part in content):
            return True
    return False


@dataclass(slots=True)
class PromptLibrary:
    """An ordered set of prompts with seeded per-request selection (FR-033).

    The selector draws uniformly from the prompts using a dedicated
    :class:`random.Random` seeded once per run, so two runs with the same seed
    produce the same ordered ``prompt_id`` sequence.
    """

    prompts: Sequence[Prompt]
    _rng: random.Random = field(default_factory=random.Random, repr=False)

    def reseed(self, seed: int) -> None:
        """Reset the selection RNG so selection is reproducible for ``seed``."""
        # Reproducible prompt selection, not a security context (FR-033).
        self._rng = random.Random(seed)  # nosec B311

    def select(self) -> Prompt:
        """Return the next seeded random prompt (FR-033)."""
        return self._rng.choice(list(self.prompts))


def _builtin_prompts() -> tuple[Prompt, ...]:
    """Return the packaged, varied built-in prompt library (FR-036)."""
    return (
        Prompt(
            id="coding-fizzbuzz",
            category="coding",
            messages=({"role": _ROLE_USER, "content": "Write a Python fizzbuzz from 1 to 100."},),
            isl_bucket="short",
            expected_output="A fizzbuzz implementation in Python.",
        ),
        Prompt(
            id="coding-refactor",
            category="coding",
            messages=(
                {
                    "role": _ROLE_USER,
                    "content": "Refactor this loop into a comprehension: for x in xs: out.append(x*2)",
                },
            ),
            isl_bucket="medium",
            expected_output="out = [x * 2 for x in xs]",
        ),
        Prompt(
            id="synthesis-summary",
            category="synthesis",
            messages=({"role": _ROLE_USER, "content": "Summarize the benefits of asynchronous I/O in two sentences."},),
            isl_bucket="medium",
            expected_output="Async I/O improves throughput under concurrency.",
        ),
        Prompt(
            id="synthesis-outline",
            category="synthesis",
            messages=({"role": _ROLE_USER, "content": "Outline a short blog post about load testing LLM endpoints."},),
            isl_bucket="medium",
            expected_output=(
                "An outline for a blog post on load testing LLM endpoints, covering an introduction, "
                "key latency and throughput metrics, methodology, and a conclusion."
            ),
        ),
        Prompt(
            id="tool-use-weather",
            category="tool-use",
            messages=({"role": _ROLE_USER, "content": "What is the weather in Paris? Use the available tools."},),
            isl_bucket="short",
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "mock_web_search",
                        "description": "Search the web for a query and return fixed results.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                },
            ),
            tool_results=(
                ("mock_web_search", {"results": [{"title": "Paris weather", "snippet": "18C, partly cloudy"}]}),
            ),
        ),
        Prompt(
            id="tool-use-pptx",
            category="tool-use",
            messages=({"role": _ROLE_USER, "content": "Generate a 3-slide deck from this outline."},),
            isl_bucket="short",
            tools=(
                {
                    "type": "function",
                    "function": {
                        "name": "mock_generate_pptx",
                        "description": "Generate a slide deck from an outline and return a fixed ack.",
                        "parameters": {
                            "type": "object",
                            "properties": {"outline": {"type": "string"}},
                            "required": ["outline"],
                        },
                    },
                },
            ),
            tool_results=(("mock_generate_pptx", {"status": "ok", "slides": 3}),),
        ),
        Prompt(
            id="vision-describe",
            category="vision",
            messages=(
                {
                    "role": _ROLE_USER,
                    "content": [
                        {"type": "text", "text": "Describe the contents of the provided image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                # A valid 32x32 solid PNG. Degenerate 1x1 images are
                                # rejected by some real backends (e.g. Bedrock returns
                                # HTTP 500 "Could not process image").
                                "url": (
                                    "data:image/png;base64,"
                                    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR42mM4"
                                    "ISdHU8QwasGoBaMWjFowasGoBaMWjFowasGoBaMWDBULACXLED1gHZEpAAAA"
                                    "AElFTkSuQmCC"
                                )
                            },
                        },
                    ],
                },
            ),
            isl_bucket="short",
        ),
        Prompt(
            id="general-greeting",
            category="general",
            messages=({"role": _ROLE_USER, "content": "Say hello and introduce yourself in one sentence."},),
            isl_bucket="short",
            expected_output="A one-sentence friendly greeting that introduces the assistant.",
        ),
        Prompt(
            id="general-explain",
            category="general",
            messages=({"role": _ROLE_USER, "content": "Explain what a benchmark is to a five year old."},),
            isl_bucket="short",
            expected_output=(
                "A simple, child-friendly explanation that a benchmark is a test measuring how fast "
                "or how well something works, so you can compare options."
            ),
        ),
    )


def builtin_library(seed: int = 0) -> PromptLibrary:
    """Return the built-in library with its selector seeded (FR-033/FR-036)."""
    library = PromptLibrary(prompts=_builtin_prompts())
    library.reseed(seed)
    return library


def _coerce_prompt(raw: Any, index: int) -> Prompt:
    """Validate and coerce one external prompt mapping into a :class:`Prompt`."""
    if not isinstance(raw, dict):
        raise PromptError(f"prompt #{index} is not a mapping")
    prompt_id = raw.get("id")
    category = raw.get("category", "general")
    messages = raw.get("messages")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise PromptError(f"prompt #{index} is missing a string 'id'")
    if not isinstance(messages, list) or not messages:
        raise PromptError(f"prompt {prompt_id!r} is missing a non-empty 'messages' list")
    isl_bucket = raw.get("isl_bucket", _DEFAULT_ISL_BUCKET)
    expected = raw.get("expected_output")
    tools = _coerce_tools(raw.get("tools"), prompt_id)
    tool_results = _coerce_tool_results(raw.get("tool_results"), prompt_id)
    return Prompt(
        id=prompt_id,
        category=str(category),
        messages=tuple(dict(message) for message in messages),
        isl_bucket=str(isl_bucket),
        expected_output=str(expected) if expected is not None else None,
        tools=tools,
        tool_results=tool_results,
    )


def _coerce_tools(raw: Any, prompt_id: str) -> tuple[dict[str, Any], ...]:
    """Validate and freeze an external prompt's OpenAI ``tools`` array (FR-038)."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PromptError(f"prompt {prompt_id!r} 'tools' must be a list")
    return tuple(dict(tool) for tool in raw if isinstance(tool, dict))


def _coerce_tool_results(raw: Any, prompt_id: str) -> tuple[tuple[str, Any], ...]:
    """Validate and freeze the deterministic mock ``tool_results`` mapping (FR-038)."""
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise PromptError(f"prompt {prompt_id!r} 'tool_results' must be a mapping")
    return tuple((str(name), payload) for name, payload in raw.items())


def parse_prompts(text: str, source: str, seed: int = 0) -> PromptLibrary:
    """Parse a prompt-library YAML ``text`` into a seeded library (FR-036).

    Args:
        text: YAML holding a top-level list of prompt mappings.
        source: Human-readable origin (file name) used in error messages.
        seed: Master seed for reproducible selection (FR-033).

    Raises:
        EmptyPromptSetError: When the text yields no prompts.
        PromptError: When the YAML is malformed or a prompt entry is invalid.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PromptError(f"invalid prompt YAML: {exc}") from exc

    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise PromptError("prompt file must hold a top-level list of prompts")
    if not raw:
        raise EmptyPromptSetError(source)

    prompts = tuple(_coerce_prompt(item, index) for index, item in enumerate(raw))
    library = PromptLibrary(prompts=prompts)
    library.reseed(seed)
    return library


def load_prompts(path: Path, seed: int = 0) -> PromptLibrary:
    """Load an external ``--prompts`` YAML file into a seeded library (FR-036).

    Raises:
        EmptyPromptSetError: When the file yields no prompts.
        PromptError: When the file is unreadable, malformed, or has a bad entry.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"prompt file not readable: {path}") from exc
    return parse_prompts(text, path.name, seed)


def _prompt_to_mapping(prompt: Prompt) -> dict[str, Any]:
    """Serialise one :class:`Prompt` back into the external YAML schema."""
    mapping: dict[str, Any] = {
        "id": prompt.id,
        "category": prompt.category,
        "isl_bucket": prompt.isl_bucket,
        "messages": [dict(message) for message in prompt.messages],
    }
    if prompt.expected_output is not None:
        mapping["expected_output"] = prompt.expected_output
    if prompt.tools:
        mapping["tools"] = [dict(tool) for tool in prompt.tools]
    if prompt.tool_results:
        mapping["tool_results"] = dict(prompt.tool_results)
    return mapping


def export_builtin_prompts_yaml() -> str:
    """Serialise the built-in library to a ``--prompts``-compatible YAML document.

    The output round-trips through :func:`load_prompts`, so ``llm-bench init`` can
    write it as a starter ``prompts.yaml`` that reproduces the built-in coverage
    exactly while remaining editable (FR-036).
    """
    payload = [_prompt_to_mapping(prompt) for prompt in _builtin_prompts()]
    header = (
        "# Starter prompt library written by 'llm-bench init'.\n"
        "# Mirrors the built-in library; edit, add, or remove entries freely.\n"
        "# Each entry is a mapping: id, category, isl_bucket, messages (required),\n"
        "# plus optional expected_output, tools, tool_results. Used automatically\n"
        "# when present, or pass --prompts <path> to point elsewhere.\n"
    )
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    return header + body


def apply_cache_busting(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend a unique prefix to the last user message (FR-034).

    A fresh UUID-derived marker is prepended to the last user text message so two
    requests for the same prompt carry different outgoing content while the body
    after the prefix is unchanged. Non-text (multimodal) content is left intact.

    Args:
        messages: The selected prompt's chat messages.

    Returns:
        A new message list with the unique prefix applied.
    """
    out = [dict(message) for message in messages]
    marker = f"[cb-{uuid.uuid4().hex}] "
    for message in reversed(out):
        if message.get("role") == _ROLE_USER and isinstance(message.get("content"), str):
            message["content"] = marker + message["content"]
            break
    return out
