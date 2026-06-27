"""Tests for the built-in local embedder (fastembed presets) without downloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnxruntime
import pytest

from llm_bench import local_embed
from llm_bench.local_embed import LOCAL_PRESETS, LocalEmbedError, embed_texts, model_name

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeModel:
    """Stand-in for fastembed.TextEmbedding returning fixed vectors."""

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


def test_model_name_presets() -> None:
    """The presets map to small (CPU) and large (GPU) bge models; bad preset raises."""
    assert LOCAL_PRESETS == ("cpu", "gpu")
    assert "small" in model_name("cpu")
    assert "large" in model_name("gpu")
    with pytest.raises(LocalEmbedError, match="unknown local embedding preset"):
        model_name("bogus")


def test_embed_texts_uses_cached_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """embed_texts returns plain float lists from the (here faked) model."""
    monkeypatch.setitem(local_embed._models, "cpu", _FakeModel())
    out = embed_texts(["a", "b"], "cpu")
    assert out == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_gpu_providers_pick_best_available_dynamically(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gpu preset dynamically picks CUDA > ROCm > Apple CoreML > CPU, never an absent one."""

    def fake_available(providers: list[str]) -> None:
        monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: providers)

    # Apple Silicon Mac (no CUDA): CoreML chosen, CPU kept last for per-op fallback.
    fake_available(["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"])
    assert local_embed._gpu_providers() == ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    # NVIDIA box: CUDA wins over CPU.
    fake_available(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert local_embed._gpu_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # CPU-only host: just CPU, never empty.
    fake_available(["CPUExecutionProvider"])
    assert local_embed._gpu_providers() == ["CPUExecutionProvider"]
