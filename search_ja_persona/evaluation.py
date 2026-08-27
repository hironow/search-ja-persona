"""Objective search-quality metrics for golden-query evaluation.

Two complementary measurements, both machine-checkable (no LLM judge):

- Golden queries: each query carries an expectation over the result's
  text/prefecture/region; precision@k counts how many of the top k satisfy
  every criterion.
- Self-retrieval: a persona's own summary text is issued as the query and
  recall@k checks whether that persona comes back in the top k.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from .prefectures import validate_prefecture

_EXPECT_KEYS = frozenset({"text_any", "text_all", "prefecture", "region"})
_ENTRY_KEYS = frozenset({"query", "expect", "tier", "filters"})
_FILTER_KEYS = frozenset({"prefecture"})
_TIERS = frozenset({"basic", "hard"})


def _normalize_uuid(value: object) -> str:
    try:
        return UUID(str(value)).hex
    except (ValueError, TypeError):
        return str(value)


def _require_keywords(keywords: Sequence[object], label: str) -> None:
    for keyword in keywords:
        if not isinstance(keyword, str) or not keyword:
            raise ValueError(f"{label}: empty keyword")


def _validate_expect(expect: object, label: str) -> None:
    if not isinstance(expect, dict) or not expect:
        raise ValueError(f"{label}: empty expect")
    unknown = set(expect) - _EXPECT_KEYS
    if unknown:
        raise ValueError(f"{label}: unknown expect key {sorted(unknown)}")

    keywords = expect.get("text_any")
    if keywords is not None:
        if not isinstance(keywords, list) or not keywords:
            raise ValueError(f"{label}: empty text_any")
        _require_keywords(keywords, label)

    groups = expect.get("text_all")
    if groups is not None:
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"{label}: empty text_all")
        for group in groups:
            if isinstance(group, str):
                _require_keywords([group], label)
            elif not isinstance(group, list) or not group:
                raise ValueError(f"{label}: empty text_all group")
            else:
                _require_keywords(group, label)

    for field in ("prefecture", "region"):
        wanted = expect.get(field)
        if wanted is not None and (not isinstance(wanted, str) or not wanted):
            raise ValueError(f"{label}: empty {field}")


def _validate_filters(filters: object, label: str) -> None:
    if not isinstance(filters, dict) or not filters:
        raise ValueError(f"{label}: empty filters")
    unknown = set(filters) - _FILTER_KEYS
    if unknown:
        raise ValueError(f"{label}: unknown filters key {sorted(unknown)}")
    validate_prefecture(str(filters["prefecture"]))


def load_golden_queries(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the golden-query set, failing fast on bad data.

    ``matches_expectation`` silently ignores unknown criteria, so a typo in
    the JSON would otherwise grade every result as a match. This loader
    rejects such data up front (call it before touching live services).
    """

    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: golden queries must be a non-empty list")

    seen_queries: set[str] = set()
    seen_tiers: set[str] = set()
    for position, entry in enumerate(entries):
        label = f"{path}[{position}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{label}: entry must be an object")
        if set(entry) - _ENTRY_KEYS:
            raise ValueError(
                f"{label}: unknown entry key {sorted(set(entry) - _ENTRY_KEYS)}"
            )
        query = entry.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError(f"{label}: empty query")
        if query in seen_queries:
            raise ValueError(f"{label}: duplicate query {query!r}")
        seen_queries.add(query)
        if "tier" not in entry:
            raise ValueError(f"{label}: missing tier")
        tier = entry["tier"]
        if tier not in _TIERS:
            raise ValueError(f"{label}: unknown tier {tier!r}")
        seen_tiers.add(tier)
        _validate_expect(entry.get("expect"), label)
        if "filters" in entry:
            _validate_filters(entry["filters"], label)

    for tier in sorted(_TIERS):
        if tier not in seen_tiers:
            raise ValueError(f"{path}: no {tier} entries")
    return entries


def matches_expectation(result: dict[str, Any], expect: dict[str, Any]) -> bool:
    """True when the result satisfies every criterion in the expectation.

    Supported criteria: ``text_any`` (at least one keyword appears in the
    aggregated text), ``text_all`` (every group is satisfied, where a group
    is a required keyword or a synonym list of which one must appear), and
    ``prefecture`` / ``region`` (exact match).
    """

    keywords = expect.get("text_any")
    if keywords is not None:
        text = result.get("text") or ""
        if not any(keyword in text for keyword in keywords):
            return False

    groups = expect.get("text_all")
    if groups is not None:
        text = result.get("text") or ""
        for group in groups:
            terms = [group] if isinstance(group, str) else list(group)
            if not any(term in text for term in terms):
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


def summarize_golden(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query precision rows into tier-aware means.

    ``golden_mean_precision`` stays the basic-tier mean — the quantity the
    intent.md quality bar was ratified against — while the overall and
    per-tier means are reported alongside it under new keys.
    """

    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_tier: dict[str, list[float]] = {}
    for row in rows:
        by_tier.setdefault(str(row["tier"]), []).append(float(row["precision_at_k"]))
    return {
        "golden_mean_precision": _mean(by_tier.get("basic", [])),
        "golden_overall_mean_precision": _mean(
            [float(row["precision_at_k"]) for row in rows]
        ),
        "golden_mean_precision_by_tier": {
            tier: _mean(values) for tier, values in sorted(by_tier.items())
        },
    }


def build_report(
    *,
    per_query: Sequence[dict[str, Any]],
    self_retrieval: dict[str, Any],
    embedder: str,
    k: int,
    generated_at: str,
    elapsed_seconds: float,
    filtered: Sequence[dict[str, Any]] | None = None,
    fusion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON benchmark report (pure — no I/O, no clock).

    Schema version 3: ``golden_queries`` holds every row with its ``tier``;
    ``golden_mean_precision`` remains the basic-tier mean (the intent.md
    bar), with overall and per-tier means under separate keys. All tier
    means are computed from unfiltered runs only; ``filtered`` carries the
    paired filtered reruns of entries that declare ``filters``.
    """

    filtered_rows = list(filtered or [])
    filtered_mean = (
        round(
            sum(row["filtered_precision_at_k"] for row in filtered_rows)
            / len(filtered_rows),
            4,
        )
        if filtered_rows
        else None
    )
    summary = summarize_golden(per_query)
    return {
        "report_schema_version": 3,
        "fusion": fusion or {},
        "filtered": filtered_rows,
        "filtered_mean_precision": filtered_mean,
        "generated_at": generated_at,
        "embedder": embedder,
        "k": k,
        "golden_mean_precision": round(summary["golden_mean_precision"], 4),
        "golden_overall_mean_precision": round(
            summary["golden_overall_mean_precision"], 4
        ),
        "golden_mean_precision_by_tier": {
            tier: round(value, 4)
            for tier, value in summary["golden_mean_precision_by_tier"].items()
        },
        "golden_queries": list(per_query),
        "self_retrieval": self_retrieval,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }


GOLDEN_BASIC_BAR = 0.85
SELF_RETRIEVAL_RECALL_BAR = 0.99


def check_thresholds(report: dict[str, Any]) -> list[str]:
    """Return failure messages against the ratified intent.md bars.

    Existence is verified before comparison: a missing metric is a
    failure, never a silent pass. Report-only runs may skip sections;
    check mode must not.
    """

    failures: list[str] = []
    by_tier = report.get("golden_mean_precision_by_tier") or {}
    basic = by_tier.get("basic")
    if basic is None:
        failures.append("basic tier missing from the report")
    elif basic < GOLDEN_BASIC_BAR:
        failures.append(f"basic mean precision {basic} < {GOLDEN_BASIC_BAR}")
    if "hard" not in by_tier:
        failures.append("hard tier missing from the report")
    if not report.get("filtered"):
        failures.append("filtered section empty (geo filters did not run)")
    self_retrieval = report.get("self_retrieval") or {}
    recall_1 = self_retrieval.get("recall_at_1")
    if not self_retrieval.get("samples") or recall_1 is None:
        failures.append("self-retrieval did not run")
    elif recall_1 < SELF_RETRIEVAL_RECALL_BAR:
        failures.append(
            f"self-retrieval recall@1 {recall_1} < {SELF_RETRIEVAL_RECALL_BAR}"
        )
    return failures


def recall_at_k(
    ranked_uuids: Sequence[object], expected_uuid: object, *, k: int
) -> bool:
    """True when the expected uuid appears in the top ``k`` ranked ids.

    Both sides are normalized so hyphenated and hyphen-less UUID forms
    compare equal.
    """

    expected = _normalize_uuid(expected_uuid)
    return any(_normalize_uuid(candidate) == expected for candidate in ranked_uuids[:k])
