from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HashedNgramVectorizer:
    """Deterministic character n-gram vectorizer.

    Generates a dense vector by hashing character n-grams into a fixed number of
    buckets. Keeps implementation self-contained so we do not rely on external
    embedding models or network access.
    """

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
