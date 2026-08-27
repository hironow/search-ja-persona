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


def test_sentence_transformer_applies_asymmetric_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts, normalize_embeddings: bool = True, **kwargs):
            calls.append(list(texts))
            return [[1.0, 1.0] for _ in texts]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_sentence_transformer",
        lambda model_name, device=None: FakeModel(),
    )

    embedder = SentenceTransformerEmbedder(
        model_name="fake-model",
        query_prefix="検索クエリ: ",
        document_prefix="検索文書: ",
    )
    embedder.embed_query("介護")
    vectors = embedder.embed_documents(["東京", "", "大阪"])

    assert calls[0] == ["検索クエリ: 介護"]
    # Documents get the document prefix; empty text stays a zero vector and
    # never reaches the model.
    assert calls[1] == ["検索文書: 東京", "検索文書: 大阪"]
    assert vectors[1] == [0.0, 0.0]


def test_sentence_transformer_encode_batch_size_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, texts, normalize_embeddings: bool = True, **kwargs):
            captured.update(kwargs)
            return [[1.0, 1.0] for _ in texts]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_sentence_transformer",
        lambda model_name, device=None: FakeModel(),
    )

    embedder = SentenceTransformerEmbedder(
        model_name="fake-model", encode_batch_size=16
    )
    embedder.embed_documents(["東京", "大阪"])

    assert captured["batch_size"] == 16


def test_fast_embedder_applies_asymmetric_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class FakeFastModel:
        vector_size = 2

        def embed(self, texts, normalize: bool = True):
            calls.append(list(texts))
            for _ in texts:
                yield [1.0, 1.0]

    monkeypatch.setattr(
        "search_ja_persona.embeddings._load_fastembed_model",
        lambda model_name, cache_dir=None: FakeFastModel(),
    )

    embedder = FastEmbedder(
        model_name="fast-model",
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    embedder.embed_query("介護")
    embedder.embed_documents(["東京", "", "大阪"])

    assert calls[0] == ["query: 介護"]
    assert calls[1] == ["passage: 東京", "passage: 大阪"]


def test_hashed_embedder_query_and_documents_delegate() -> None:
    embedder = HashedNgramEmbedder(dimension=8, ngram_sizes=(2, 3))

    assert embedder.embed_query("介護") == embedder.embed("介護")
    assert embedder.embed_documents(["東京", "大阪"]) == embedder.embed_many(
        ["東京", "大阪"]
    )


def test_retrieval_presets_declare_prefixes() -> None:
    from search_ja_persona.embeddings import EMBEDDER_PRESETS

    for name in ("e5-small", "e5-large", "fast-e5-small", "fast-e5-large"):
        preset = EMBEDDER_PRESETS[name]
        assert preset["query_prefix"] == "query: "
        assert preset["document_prefix"] == "passage: "

    for name in ("ruri-v3-310m", "ruri-v3-130m"):
        preset = EMBEDDER_PRESETS[name]
        assert preset["type"] == "sentence"
        assert preset["model"] == f"cl-nagoya/{name}"
        assert preset["query_prefix"] == "検索クエリ: "
        assert preset["document_prefix"] == "検索文書: "

    assert EMBEDDER_PRESETS["ruri-v3-310m"]["encode_batch_size"] == 16
    assert EMBEDDER_PRESETS["ruri-v3-130m"]["encode_batch_size"] == 32
