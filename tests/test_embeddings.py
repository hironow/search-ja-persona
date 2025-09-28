from __future__ import annotations

import math
from typing import Any

import pytest

from search_ja_persona.embeddings import (
    FastEmbedder,
    HashedNgramEmbedder,
    SentenceTransformerEmbedder,
)


def test_hashed_ngram_embedder_produces_unit_vector() -> None:
    embedder = HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3))
    vector = embedder.embed("介護の品質を高めるリーダー")

    assert len(vector) == 8
    magnitude = math.sqrt(sum(value * value for value in vector))
    assert pytest.approx(magnitude, rel=1e-6) == 1.0


def test_sentence_transformer_embedder_uses_underlying_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, normalize_embeddings: bool = True):
            captured["texts"] = texts
            captured["normalize"] = normalize_embeddings
            return [[1.0, 2.0, 3.0, 4.0]]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_sentence_transformer",
        lambda model_name, device=None: FakeModel(),
    )

    embedder = SentenceTransformerEmbedder(
        model_name="fake-model", normalize_embeddings=False
    )
    vector = embedder.embed("こんにちは")

    assert vector == [1.0, 2.0, 3.0, 4.0]
    assert embedder.dimension == 4
    assert captured["texts"] == ["こんにちは"]
    assert captured["normalize"] is False


def test_fast_embedder_invokes_fastembed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeFastModel:
        vector_size = 3

        def embed(self, texts, normalize: bool = True):
            captured["texts"] = texts
            captured["normalize"] = normalize
            yield [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_fastembed_model",
        lambda model_name, cache_dir=None: FakeFastModel(),
    )

    embedder = FastEmbedder(
        model_name="fast-model", cache_dir="/tmp/cache", normalize_embeddings=False
    )
    vector = embedder.embed("こんにちは")

    assert vector == [1.0, 0.0, 0.0]
    assert embedder.dimension == 3
    assert captured["texts"] == ["こんにちは"]
    assert captured["normalize"] is False
