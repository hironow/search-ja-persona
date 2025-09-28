from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Protocol


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


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
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * self._dimension

        vectors = self._model.encode(
            [cleaned], normalize_embeddings=self.normalize_embeddings
        )
        vector = vectors[0]
        if hasattr(vector, "tolist"):
            return list(map(float, vector.tolist()))
        return [float(value) for value in vector]


@dataclass
class FastEmbedder:
    model_name: str
    cache_dir: str | None = None
    normalize_embeddings: bool = True

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
        cleaned = (text or "").strip()
        if not cleaned:
            return [0.0] * self._dimension

        try:
            iterator = self._model.embed([cleaned], normalize=self.normalize_embeddings)
        except TypeError:  # pragma: no cover - older fastembed without normalize kw
            iterator = self._model.embed([cleaned])
        vectors = list(iterator)
        vector = vectors[0]
        if hasattr(vector, "tolist"):
            return list(map(float, vector.tolist()))
        return [float(value) for value in vector]


# Backwards compatibility alias for existing imports
HashedNgramVectorizer = HashedNgramEmbedder

__all__ = [
    "Embedder",
    "HashedNgramEmbedder",
    "FastEmbedder",
    "SentenceTransformerEmbedder",
    "HashedNgramVectorizer",
    "EMBEDDER_PRESETS",
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
    },
    "e5-large": {
        "type": "sentence",
        "model": "intfloat/multilingual-e5-large",
    },
    "fast-e5-small": {
        "type": "fast",
        "model": "intfloat/multilingual-e5-small",
    },
    "fast-e5-large": {
        "type": "fast",
        "model": "intfloat/multilingual-e5-large",
    },
}
