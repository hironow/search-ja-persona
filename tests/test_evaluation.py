from __future__ import annotations

from search_ja_persona.evaluation import (
    matches_expectation,
    precision_at_k,
    recall_at_k,
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
