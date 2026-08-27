from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class Embedder(Protocol):
    """Retrieval-facing surface: queries and documents embed asymmetrically
    (models like e5 and Ruri expect distinct prefixes per side)."""

    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


def _to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        return list(map(float, vector.tolist()))
    return [float(value) for value in vector]


@dataclass(frozen=True)
class HashedNgramEmbedder:
    dimension: int
    ngram_sizes: tuple[int, ...]

    def embed(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * self.dimension

        buckets = [0.0] * self.dimension
        for n in self.ngram_sizes:
            if n <= 0:
                continue
            for ngram in self._generate_ngrams(cleaned, n):
                index = self._hash_to_index(ngram)
                buckets[index] += 1.0

        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0:
            return buckets
        return [value / norm for value in buckets]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(texts)

    def _generate_ngrams(self, text: str, n: int) -> Iterable[str]:
        if len(text) < n:
            yield text
            return
        for index in range(len(text) - n + 1):
            yield text[index : index + n]

    def _hash_to_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        return value % self.dimension


def _load_sentence_transformer(
    model_name: str, device: str | None = None
):  # pragma: no cover - wrapper
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "sentence-transformers is required for the SentenceTransformerEmbedder"
        ) from exc
    return SentenceTransformer(model_name, device=device)


def _load_fastembed_model(
    model_name: str, cache_dir: str | None = None
):  # pragma: no cover - wrapper
    try:
        from fastembed import TextEmbedding
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError("fastembed is required for the FastEmbedder") from exc

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return TextEmbedding(model_name=model_name, **kwargs)


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    device: str | None = None
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""
    encode_batch_size: int | None = None

    def __post_init__(self) -> None:
        self._model = _load_sentence_transformer(self.model_name, device=self.device)
        dimension_getter = getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if callable(dimension_getter):
            self._dimension = int(dimension_getter())
        else:  # pragma: no cover - fallback for unexpected models
            vector = self._model.encode(
                [""], normalize_embeddings=self.normalize_embeddings
            )[0]
            self._dimension = len(vector)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [(text or "").strip() for text in texts]
        non_empty = [text for text in cleaned if text]
        if non_empty:
            encode_kwargs: dict[str, Any] = {
                "normalize_embeddings": self.normalize_embeddings
            }
            if self.encode_batch_size is not None:
                encode_kwargs["batch_size"] = self.encode_batch_size
            vectors = self._model.encode(non_empty, **encode_kwargs)
            encoded = iter([_to_float_list(vector) for vector in vectors])
        else:
            encoded = iter([])
        return [next(encoded) if text else [0.0] * self._dimension for text in cleaned]

    def embed_query(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * self._dimension
        return self.embed(f"{self.query_prefix}{cleaned}")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(
            [
                f"{self.document_prefix}{cleaned}" if cleaned else ""
                for cleaned in ((text or "").strip() for text in texts)
            ]
        )


@dataclass
class FastEmbedder:
    model_name: str
    cache_dir: str | None = None
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        self._model = _load_fastembed_model(self.model_name, cache_dir=self.cache_dir)
        size = getattr(self._model, "vector_size", None) or getattr(
            self._model, "embedding_size", None
        )
        if size:
            self._dimension = int(size)
        else:  # pragma: no cover - fallback
            vector = next(self._model.embed([""], normalize=self.normalize_embeddings))
            self._dimension = len(vector)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [(text or "").strip() for text in texts]
        non_empty = [text for text in cleaned if text]
        if non_empty:
            try:
                iterator = self._model.embed(
                    non_empty, normalize=self.normalize_embeddings
                )
            except TypeError:  # pragma: no cover - older fastembed without kw
                iterator = self._model.embed(non_empty)
            encoded = iter([_to_float_list(vector) for vector in iterator])
        else:
            encoded = iter([])
        return [next(encoded) if text else [0.0] * self._dimension for text in cleaned]

    def embed_query(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * self._dimension
        return self.embed(f"{self.query_prefix}{cleaned}")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_many(
            [
                f"{self.document_prefix}{cleaned}" if cleaned else ""
                for cleaned in ((text or "").strip() for text in texts)
            ]
        )


# Backwards compatibility alias for existing imports
HashedNgramVectorizer = HashedNgramEmbedder

__all__ = [
    "EMBEDDER_PRESETS",
    "Embedder",
    "FastEmbedder",
    "HashedNgramEmbedder",
    "HashedNgramVectorizer",
    "SentenceTransformerEmbedder",
]

EMBEDDER_PRESETS = {
    "hashed": {
        "type": "hashed",
    },
    "mini-lm": {
        "type": "sentence",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "mpnet": {
        "type": "sentence",
        "model": "sentence-transformers/all-mpnet-base-v2",
    },
    "e5-small": {
        "type": "sentence",
        "model": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "e5-large": {
        "type": "sentence",
        "model": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "fast-e5-small": {
        "type": "fast",
        "model": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    "fast-e5-large": {
        "type": "fast",
        "model": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    },
    # Ruri v3 (cl-nagoya): Japanese retrieval models; encode_batch_size per
    # measured 4090 throughput sweet spots.
    "ruri-v3-310m": {
        "type": "sentence",
        "model": "cl-nagoya/ruri-v3-310m",
        "query_prefix": "検索クエリ: ",
        "document_prefix": "検索文書: ",
        "encode_batch_size": 16,
    },
    "ruri-v3-130m": {
        "type": "sentence",
        "model": "cl-nagoya/ruri-v3-130m",
        "query_prefix": "検索クエリ: ",
        "document_prefix": "検索文書: ",
        "encode_batch_size": 32,
    },
}
