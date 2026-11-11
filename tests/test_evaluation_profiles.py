from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from search_ja_persona.dataset_profiles import FINANCE, JAPAN, PII
from search_ja_persona.evaluation import (
    check_thresholds,
    load_golden_queries,
    matches_expectation,
)


def _write(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def _pii_entries() -> list[dict[str, Any]]:
    return [
        {
            "query": "ローンの審査書類",
            "tier": "basic",
            "expect": {"document_type": "ローン審査書類"},
        },
        {
            "query": "口座番号が書かれた金融の書類",
            "tier": "hard",
            "expect": {"domain": "金融", "text_any": ["口座番号"]},
            "filters": {"domain": "金融"},
        },
    ]


def test_pii_golden_uses_pii_payload_fields_and_filters(tmp_path: Path) -> None:
    entries = load_golden_queries(_write(tmp_path, _pii_entries()), profile=PII)

    assert len(entries) == 2


def test_expect_keys_outside_the_profile_are_rejected(tmp_path: Path) -> None:
    entries = _pii_entries()
    entries[0]["expect"] = {"prefecture": "東京都"}

    with pytest.raises(ValueError, match="unknown expect key"):
        load_golden_queries(_write(tmp_path, entries), profile=PII)


def test_filter_keys_outside_the_profile_are_rejected(tmp_path: Path) -> None:
    entries = [
        {"query": "介護職", "tier": "basic", "expect": {"text_any": ["介護"]}},
        {
            "query": "金融の人",
            "tier": "hard",
            "expect": {"text_any": ["金融"]},
            "filters": {"domain": "金融"},
        },
    ]

    with pytest.raises(ValueError, match="unknown filters key"):
        load_golden_queries(_write(tmp_path, entries), profile=JAPAN)


def test_a_filtered_entry_needs_a_criterion_beyond_its_filter(tmp_path: Path) -> None:
    # With the filter applied, an expectation that only restates the filter
    # scores 1.0 whatever the ranking does.
    entries = _pii_entries()
    entries[1]["expect"] = {"domain": "金融"}

    with pytest.raises(ValueError, match="beyond its filters"):
        load_golden_queries(_write(tmp_path, entries), profile=PII)


def test_matches_expectation_compares_any_payload_field() -> None:
    hit = {"text": "口座番号は1234567です", "domain": "金融", "document_type": "申込書"}

    assert matches_expectation(hit, {"domain": "金融", "text_any": ["口座番号"]})
    assert not matches_expectation(hit, {"document_type": "ローン審査書類"})


def _report(**overrides: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "golden_mean_precision_by_tier": {"basic": 0.2, "hard": 0.1},
        "filtered": [{"filtered_precision_at_k": 0.1}],
        "filtered_mean_precision": 0.1,
        "self_retrieval": {"samples": 100, "recall_at_1": 0.1, "recall_at_10": 0.2},
        "context_coverage": 1.0,
        "keyword_contribution_rate": 1.0,
        "store_counts": {"qdrant": 5, "elasticsearch": 5, "neo4j": 5},
    }
    report.update(overrides)
    return report


def test_unratified_profiles_gate_only_on_the_canaries() -> None:
    assert check_thresholds(_report(), profile=FINANCE) == []
    assert check_thresholds(_report(self_retrieval={"samples": 0}), profile=PII) == []
    assert check_thresholds(_report(), profile=JAPAN)  # ratified bars fail

    dead_keyword_leg = _report(keyword_contribution_rate=0.0)
    assert check_thresholds(dead_keyword_leg, profile=FINANCE) == [
        "keyword leg contributed to no query (dead leg)"
    ]
    drifted = _report(store_counts={"qdrant": 5, "elasticsearch": 4, "neo4j": 5})
    assert check_thresholds(drifted, profile=PII)
