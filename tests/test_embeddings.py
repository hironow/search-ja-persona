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


def test_hashed_ngram_embedder_embed_many_matches_embed() -> None:
    embedder = HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3))
    texts = ["介護の品質を高めるリーダー", "", "地域医療に貢献する看護師"]

    vectors = embedder.embed_many(texts)

    assert vectors == [embedder.embed(text) for text in texts]


def test_sentence_transformer_embed_many_encodes_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts, normalize_embeddings: bool = True):
            calls.append(list(texts))
            return [[1.0, float(index)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_sentence_transformer",
        lambda model_name, device=None: FakeModel(),
    )

    embedder = SentenceTransformerEmbedder(model_name="fake-model")
    vectors = embedder.embed_many(["こんにちは", "", "世界"])

    # One batched encode call covering the non-empty texts; the empty text
    # becomes a zero vector without hitting the model.
    assert calls == [["こんにちは", "世界"]]
    assert vectors == [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]


def test_fast_embedder_embed_many_embeds_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeFastModel:
        vector_size = 2

        def embed(self, texts, normalize: bool = True):
            calls.append(list(texts))
            for index, _ in enumerate(texts):
                yield [1.0, float(index)]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_fastembed_model",
        lambda model_name, cache_dir=None: FakeFastModel(),
    )

    embedder = FastEmbedder(model_name="fast-model")
    vectors = embedder.embed_many(["こんにちは", "", "世界"])

    assert calls == [["こんにちは", "世界"]]
    assert vectors == [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
