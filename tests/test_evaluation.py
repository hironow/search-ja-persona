from __future__ import annotations

import json
from pathlib import Path

import pytest

from search_ja_persona.evaluation import (
    build_report,
    check_thresholds,
    load_golden_queries,
    matches_expectation,
    precision_at_k,
    recall_at_k,
    summarize_golden,
)


def _result(text: str = "", prefecture: str = "", region: str = "") -> dict:
    return {"text": text, "prefecture": prefecture, "region": region}


def test_matches_expectation_text_any() -> None:
    expect = {"text_any": ["介護", "看護"]}

    assert matches_expectation(_result(text="長年の介護経験を持つ"), expect)
    assert matches_expectation(_result(text="病棟の看護リーダー"), expect)
    assert not matches_expectation(_result(text="機械設計のエンジニア"), expect)


def test_matches_expectation_prefecture_and_region() -> None:
    expect = {"prefecture": "大阪府"}
    assert matches_expectation(_result(prefecture="大阪府"), expect)
    assert not matches_expectation(_result(prefecture="東京都"), expect)

    expect_region = {"region": "近畿地方"}
    assert matches_expectation(_result(region="近畿地方"), expect_region)
    assert not matches_expectation(_result(region="関東地方"), expect_region)


def test_matches_expectation_text_all_requires_every_group() -> None:
    expect = {"text_all": [["看護"], ["子育て", "育児"]]}

    assert matches_expectation(_result(text="病棟で看護しながら子育て中"), expect)
    assert matches_expectation(_result(text="看護師として働き育児にも奮闘"), expect)
    assert not matches_expectation(_result(text="看護一筋のベテラン"), expect)
    assert not matches_expectation(_result(text="子育てに専念しています"), expect)


def test_matches_expectation_text_all_accepts_bare_string_group() -> None:
    expect = {"text_all": ["温泉", "旅行"]}

    assert matches_expectation(_result(text="温泉旅行が毎年の楽しみ"), expect)
    assert not matches_expectation(_result(text="温泉が好き"), expect)


def test_matches_expectation_requires_all_criteria() -> None:
    expect = {"text_any": ["たこ焼き", "粉もの"], "prefecture": "大阪府"}

    assert matches_expectation(
        _result(text="たこ焼きが得意", prefecture="大阪府"), expect
    )
    assert not matches_expectation(
        _result(text="たこ焼きが得意", prefecture="東京都"), expect
    )
    assert not matches_expectation(
        _result(text="お寿司が好き", prefecture="大阪府"), expect
    )


def test_precision_at_k() -> None:
    expect = {"text_any": ["介護"]}
    results = [
        _result(text="介護のベテラン"),
        _result(text="機械設計"),
        _result(text="介護施設の運営"),
        _result(text="旅行好き"),
    ]

    assert precision_at_k(results, expect, k=4) == 0.5
    assert precision_at_k(results, expect, k=1) == 1.0
    assert precision_at_k([], expect, k=5) == 0.0


def test_recall_at_k_normalizes_uuid_forms() -> None:
    ranked = [
        "63f4de5a-14e7-4acd-a918-16138ef70dfe",
        "9ab67434675a46f98cab22f779e8550e",
    ]

    assert recall_at_k(ranked, "63f4de5a14e74acda91816138ef70dfe", k=1)
    assert recall_at_k(ranked, "9ab67434-675a-46f9-8cab-22f779e8550e", k=2)
    assert not recall_at_k(ranked, "9ab67434675a46f98cab22f779e8550e", k=1)
    assert not recall_at_k(ranked, "ffffffffffffffffffffffffffffffff", k=2)


def _golden_entry(
    query: str = "q-basic",
    tier: str = "basic",
    expect: dict | None = None,
) -> dict:
    return {
        "query": query,
        "tier": tier,
        "expect": expect if expect is not None else {"text_any": ["介護"]},
    }


def _write_golden(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_golden_queries_returns_entries_with_tiers(tmp_path: Path) -> None:
    path = _write_golden(
        tmp_path,
        [
            _golden_entry(),
            _golden_entry(
                query="q-hard",
                tier="hard",
                expect={
                    "text_all": [["看護"], ["子育て", "育児"]],
                    "prefecture": "大阪府",
                },
            ),
        ],
    )

    entries = load_golden_queries(path)

    assert [entry["tier"] for entry in entries] == ["basic", "hard"]
    assert entries[0]["query"] == "q-basic"


@pytest.mark.parametrize(
    ("bad_entry", "message"),
    [
        (_golden_entry(expect={}), "empty expect"),
        (_golden_entry(expect={"text_any": []}), "empty text_any"),
        (_golden_entry(expect={"text_any": ["介護", ""]}), "empty keyword"),
        (_golden_entry(expect={"text_all": []}), "empty text_all"),
        (_golden_entry(expect={"text_all": [[]]}), "empty text_all group"),
        (_golden_entry(expect={"text_none": ["介護"]}), "unknown expect key"),
        (_golden_entry(expect={"prefecture": ""}), "empty prefecture"),
        (_golden_entry(tier="expert"), "unknown tier"),
        ({"query": "q-basic", "expect": {"text_any": ["介護"]}}, "missing tier"),
        (_golden_entry(query=""), "empty query"),
        (_golden_entry(query="q-hard"), "duplicate query"),
    ],
)
def test_load_golden_queries_rejects_invalid_entries(
    tmp_path: Path, bad_entry: dict, message: str
) -> None:
    path = _write_golden(
        tmp_path,
        [_golden_entry(query="q-hard", tier="hard"), bad_entry],
    )

    with pytest.raises(ValueError, match=message):
        load_golden_queries(path)


def test_load_golden_queries_requires_both_tiers(tmp_path: Path) -> None:
    path = _write_golden(tmp_path, [_golden_entry()])

    with pytest.raises(ValueError, match="no hard entries"):
        load_golden_queries(path)


def test_summarize_golden_keeps_basic_mean_and_adds_tier_means() -> None:
    rows = [
        {"query": "a", "tier": "basic", "precision_at_k": 1.0},
        {"query": "b", "tier": "basic", "precision_at_k": 0.5},
        {"query": "c", "tier": "hard", "precision_at_k": 0.25},
    ]

    summary = summarize_golden(rows)

    assert summary["golden_mean_precision"] == pytest.approx(0.75)
    assert summary["golden_overall_mean_precision"] == pytest.approx(1.75 / 3)
    assert summary["golden_mean_precision_by_tier"] == {
        "basic": pytest.approx(0.75),
        "hard": pytest.approx(0.25),
    }


def test_summarize_golden_handles_empty_rows() -> None:
    summary = summarize_golden([])

    assert summary["golden_mean_precision"] == 0.0
    assert summary["golden_overall_mean_precision"] == 0.0
    assert summary["golden_mean_precision_by_tier"] == {}


def test_build_report_preserves_bar_metric_and_tier_rows() -> None:
    rows = [
        {"query": "a", "tier": "basic", "precision_at_k": 1.0},
        {"query": "b", "tier": "basic", "precision_at_k": 0.5},
        {"query": "c", "tier": "hard", "precision_at_k": 0.25},
    ]

    report = build_report(
        per_query=rows,
        self_retrieval={"samples": 2, "recall_at_1": 1.0, "recall_at_10": 1.0},
        embedder="ruri-v3-310m",
        k=5,
        generated_at="2026-08-27T00:00:00+00:00",
        elapsed_seconds=1.26,
    )

    assert report["report_schema_version"] == 3
    assert report["golden_mean_precision"] == pytest.approx(0.75)
    assert report["golden_overall_mean_precision"] == pytest.approx(0.5833)
    assert report["golden_mean_precision_by_tier"] == {"basic": 0.75, "hard": 0.25}
    assert [row["tier"] for row in report["golden_queries"]] == [
        "basic",
        "basic",
        "hard",
    ]
    basic_scores = [
        row["precision_at_k"]
        for row in report["golden_queries"]
        if row["tier"] == "basic"
    ]
    assert report["golden_mean_precision"] == pytest.approx(
        sum(basic_scores) / len(basic_scores)
    )
    assert report["elapsed_seconds"] == 1.3
    assert report["embedder"] == "ruri-v3-310m"
    assert report["k"] == 5
    assert report["self_retrieval"]["samples"] == 2


def test_shipped_golden_queries_are_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "golden_queries.json"

    entries = load_golden_queries(path)

    tiers = {entry["tier"] for entry in entries}
    assert tiers == {"basic", "hard"}
    assert len(entries) >= 24


def test_load_golden_queries_accepts_prefecture_filter(tmp_path: Path) -> None:
    filtered_entry = _golden_entry(
        query="q-hard", tier="hard", expect={"prefecture": "北海道"}
    )
    filtered_entry["filters"] = {"prefecture": "北海道"}
    path = _write_golden(tmp_path, [_golden_entry(), filtered_entry])

    entries = load_golden_queries(path)

    assert entries[1]["filters"] == {"prefecture": "北海道"}


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({}, "empty filters"),
        ({"region": "関東地方"}, "unknown filters key"),
        ({"prefecture": "沖縄"}, "unknown prefecture"),
        ({"prefecture": ""}, "unknown prefecture"),
    ],
)
def test_load_golden_queries_rejects_invalid_filters(
    tmp_path: Path, filters: dict, message: str
) -> None:
    bad_entry = _golden_entry(query="q-hard", tier="hard")
    bad_entry["filters"] = filters
    path = _write_golden(tmp_path, [_golden_entry(), bad_entry])

    with pytest.raises(ValueError, match=message):
        load_golden_queries(path)


def test_build_report_v3_includes_filtered_section() -> None:
    rows = [
        {"query": "a", "tier": "basic", "precision_at_k": 1.0},
        {"query": "b", "tier": "hard", "precision_at_k": 0.4},
    ]
    filtered = [
        {
            "query": "b",
            "tier": "hard",
            "filters": {"prefecture": "沖縄県"},
            "unfiltered_precision_at_k": 0.4,
            "filtered_precision_at_k": 0.8,
            "delta": 0.4,
        }
    ]

    report = build_report(
        per_query=rows,
        filtered=filtered,
        self_retrieval={"samples": 1, "recall_at_1": 1.0, "recall_at_10": 1.0},
        embedder="ruri-v3-310m",
        k=5,
        generated_at="2026-08-27T00:00:00+00:00",
        elapsed_seconds=1.0,
    )

    assert report["report_schema_version"] == 3
    assert report["filtered"] == filtered
    assert report["filtered_mean_precision"] == 0.8
    assert report["golden_mean_precision_by_tier"]["hard"] == 0.4


def test_build_report_defaults_to_empty_filtered_section() -> None:
    rows = [{"query": "a", "tier": "basic", "precision_at_k": 1.0}]

    report = build_report(
        per_query=rows,
        self_retrieval={"samples": 1, "recall_at_1": 1.0, "recall_at_10": 1.0},
        embedder="ruri-v3-310m",
        k=5,
        generated_at="2026-08-27T00:00:00+00:00",
        elapsed_seconds=1.0,
    )

    assert report["filtered"] == []
    assert report["filtered_mean_precision"] is None


def _healthy_report(**overrides: object) -> dict:
    report: dict = {
        "golden_mean_precision": 0.9,
        "golden_mean_precision_by_tier": {"basic": 0.9, "hard": 0.43},
        "filtered": [{"query": "q"}],
        "filtered_mean_precision": 1.0,
        "self_retrieval": {"samples": 100, "recall_at_1": 1.0, "recall_at_10": 1.0},
    }
    report.update(overrides)
    return report


def test_check_thresholds_passes_on_healthy_report() -> None:
    assert check_thresholds(_healthy_report()) == []


def test_check_thresholds_accepts_the_amended_self_retrieval_bar() -> None:
    # Ratified 2026-08-27 with the name-exclusion adoption: recall@1 >= 0.90
    # and recall@10 >= 0.99 (anonymized vectors make same-vibe personas
    # equivalent, so rank-1 by one's own named summary is no longer owed).
    report = _healthy_report(
        self_retrieval={"samples": 100, "recall_at_1": 0.92, "recall_at_10": 1.0}
    )
    assert check_thresholds(report) == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"golden_mean_precision_by_tier": {"basic": 0.84, "hard": 0.43}}, "basic"),
        ({"golden_mean_precision_by_tier": {"hard": 0.43}}, "basic"),
        ({"golden_mean_precision_by_tier": {"basic": 0.9}}, "hard"),
        (
            {
                "self_retrieval": {
                    "samples": 0,
                    "recall_at_1": None,
                    "recall_at_10": None,
                }
            },
            "self-retrieval",
        ),
        (
            {
                "self_retrieval": {
                    "samples": 100,
                    "recall_at_1": 0.88,
                    "recall_at_10": 1.0,
                }
            },
            "recall@1",
        ),
        (
            {
                "self_retrieval": {
                    "samples": 100,
                    "recall_at_1": 0.95,
                    "recall_at_10": 0.98,
                }
            },
            "recall@10",
        ),
        ({"filtered": [], "filtered_mean_precision": None}, "filtered"),
    ],
)
def test_check_thresholds_reports_failures(overrides: dict, fragment: str) -> None:
    failures = check_thresholds(_healthy_report(**overrides))

    assert failures
    assert any(fragment in failure for failure in failures)


def test_build_report_records_fusion_config() -> None:
    report = build_report(
        per_query=[{"query": "a", "tier": "basic", "precision_at_k": 1.0}],
        self_retrieval={"samples": 1, "recall_at_1": 1.0, "recall_at_10": 1.0},
        embedder="ruri-v3-310m",
        k=5,
        generated_at="2026-08-27T00:00:00+00:00",
        elapsed_seconds=1.0,
        fusion={"rrf_weights": [2.0, 1.0]},
    )

    assert report["fusion"] == {"rrf_weights": [2.0, 1.0]}
