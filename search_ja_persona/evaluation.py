"""Objective search-quality metrics for golden-query evaluation.

Two complementary measurements, both machine-checkable (no LLM judge):

- Golden queries: each query carries an expectation over the result's
  text/prefecture/region; precision@k counts how many of the top k satisfy
  every criterion.
- Self-retrieval: a persona's own summary text is issued as the query and
  recall@k checks whether that persona comes back in the top k.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID


def _normalize_uuid(value: object) -> str:
    try:
        return UUID(str(value)).hex
    except (ValueError, TypeError):
        return str(value)


def matches_expectation(result: dict[str, Any], expect: dict[str, Any]) -> bool:
    """True when the result satisfies every criterion in the expectation.

    Supported criteria: ``text_any`` (at least one keyword appears in the
    aggregated text), ``prefecture`` and ``region`` (exact match).
    """

    keywords = expect.get("text_any")
    if keywords is not None:
        text = result.get("text") or ""
        if not any(keyword in text for keyword in keywords):
            return False

    for field in ("prefecture", "region"):
        wanted = expect.get(field)
        if wanted is not None and result.get(field) != wanted:
            return False

    return True


def precision_at_k(
    results: Sequence[dict[str, Any]], expect: dict[str, Any], *, k: int
) -> float:
    """Fraction of the top ``k`` results that satisfy the expectation.

    An empty result list scores 0.0: returning nothing is a failure, not a
    vacuous success.
    """

    if k <= 0 or not results:
        return 0.0
    top = results[:k]
    matched = sum(1 for result in top if matches_expectation(result, expect))
    return matched / k


def recall_at_k(
    ranked_uuids: Sequence[object], expected_uuid: object, *, k: int
) -> bool:
    """True when the expected uuid appears in the top ``k`` ranked ids.

    Both sides are normalized so hyphenated and hyphen-less UUID forms
    compare equal.
    """

    expected = _normalize_uuid(expected_uuid)
    return any(_normalize_uuid(candidate) == expected for candidate in ranked_uuids[:k])
