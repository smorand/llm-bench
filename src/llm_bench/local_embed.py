"""Built-in local text embeddings for quality evaluation (no external API).

When the embedding evaluator is configured ``local: cpu`` or ``local: gpu``,
llm-bench computes the vectors in-process with `fastembed <https://github.com/qdrant/fastembed>`_
(ONNX runtime) instead of POSTing to an embeddings endpoint. Two presets:

* ``cpu`` - ``BAAI/bge-small-en-v1.5`` (small, fast on CPU)
* ``gpu`` - ``BAAI/bge-large-en-v1.5`` (larger / better; runs on the best
  available accelerator - CUDA on NVIDIA, CoreML on Apple Silicon - and falls
  back to CPU, instead of hard-requiring a CUDA runtime)

The model is downloaded once on first use and cached by fastembed. Models are
loaded lazily and memoised, so importing this module is cheap.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Preset -> fastembed model id.
_PRESETS: dict[str, str] = {
    "cpu": "BAAI/bge-small-en-v1.5",
    "gpu": "BAAI/bge-large-en-v1.5",
}
LOCAL_PRESETS: tuple[str, ...] = tuple(_PRESETS)

_lock = threading.Lock()
_models: dict[str, Any] = {}


class LocalEmbedError(Exception):
    """Raised when a local embedding model cannot be loaded or run."""


def model_name(preset: str) -> str:
    """Return the fastembed model id backing a preset (raises on unknown preset)."""
    try:
        return _PRESETS[preset]
    except KeyError as exc:
        raise LocalEmbedError(f"unknown local embedding preset: {preset!r} (use {list(LOCAL_PRESETS)})") from exc


# Accelerator preference for the ``gpu`` preset, best first. Intersected with what
# onnxruntime actually has installed, so an unavailable one is never requested
# (onnxruntime raises if you list a provider it does not have).
_GPU_PROVIDER_PREFERENCE: tuple[str, ...] = (
    "CUDAExecutionProvider",  # NVIDIA
    "ROCMExecutionProvider",  # AMD
    "CoreMLExecutionProvider",  # Apple Silicon (GPU / Neural Engine)
    "CPUExecutionProvider",  # always-present fallback
)


def _gpu_providers() -> list[str]:
    """Pick the ONNX execution providers for the gpu preset, dynamically.

    Queries the runtime for installed providers and returns the available ones in
    preference order (CUDA -> ROCm -> Apple CoreML -> CPU). CPU is always kept last
    so onnxruntime can fall back per-op; the list is never empty.
    """
    try:
        import onnxruntime  # noqa: PLC0415  (optional heavy import, deferred)

        available = set(onnxruntime.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime ships with fastembed
        return ["CPUExecutionProvider"]
    providers = [p for p in _GPU_PROVIDER_PREFERENCE if p in available]
    return providers or ["CPUExecutionProvider"]


def _get_model(preset: str) -> Any:
    """Lazily construct and memoise the fastembed model for ``preset``."""
    name = model_name(preset)
    with _lock:
        cached = _models.get(preset)
        if cached is not None:
            return cached
        try:
            from fastembed import TextEmbedding  # noqa: PLC0415  (heavy import, deferred to first use)
        except ImportError as exc:  # pragma: no cover - fastembed is a base dependency
            raise LocalEmbedError("fastembed is not installed") from exc
        kwargs: dict[str, Any] = {}
        if preset == "gpu":
            kwargs["providers"] = _gpu_providers()
        try:
            model = TextEmbedding(model_name=name, **kwargs)
        except Exception as exc:
            raise LocalEmbedError(f"could not load local embedding model {name!r}: {exc}") from exc
        _models[preset] = model
        return model


def embed_texts(texts: Sequence[str], preset: str) -> list[list[float]]:
    """Embed ``texts`` locally with the model for ``preset`` (synchronous)."""
    model = _get_model(preset)
    try:
        return [[float(x) for x in vector] for vector in model.embed(list(texts))]
    except Exception as exc:
        raise LocalEmbedError(f"local embedding failed: {exc}") from exc
