from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_lookup_fixture import pick_unique_names, pick_unique_values
from scripts.evaluate_name_lookup import fixture_path
from search_ja_persona.dataset_profiles import FINANCE, PII


def _persona(uuid: str, name: str) -> dict:
    return {
        "uuid": uuid,
        "persona": f"{name}は、堅実な会社員だ。",
        "professional_persona": f"{name}は、経理を担当する。",
        "finance_persona": f"{name}は、毎月貯蓄する。",
    }


def test_names_are_corpus_unique_and_stratified_per_shard() -> None:
    shards = [
        [
            _persona("a1", "山辺 楓人"),
            _persona("a2", "田中 一郎"),
            _persona("a3", "佐藤 花子"),
        ],
        [
            _persona("b1", "田中 一郎"),
            _persona("b2", "鈴木 次郎"),
            _persona("b3", "高橋 三郎"),
        ],
    ]

    picked = pick_unique_names(shards, per_shard=2, seed=7)

    names = [entry["name"] for entry in picked]
    assert "田中 一郎" not in names  # appears in two shards
    assert len(picked) == 4
    assert {entry["uuid"][0] for entry in picked} == {"a", "b"}
    assert picked == pick_unique_names(shards, per_shard=2, seed=7)


def _document(uid: str, domain: str, text: str, spans: list[tuple[str, str]]) -> dict:
    return {
        "uuid": uid,
        "domain": domain,
        "document_text": text,
        "spans": [{"text": value, "label": label} for value, label in spans],
    }


def test_values_occur_in_one_document_only_and_cover_each_domain() -> None:
    documents = [
        _document(
            "m1",
            "医療",
            "電話 03-1111-2222 カルテ MRN-778899",
            [("03-1111-2222", "phone_number"), ("MRN-778899", "medical_record_number")],
        ),
        _document(
            "m2", "医療", "電話 03-1111-2222", [("03-1111-2222", "phone_number")]
        ),
        _document(
            "f1",
            "金融",
            "口座 7654321 連絡先 a@example.jp",
            [("7654321", "account_number"), ("a@example.jp", "email")],
        ),
        _document("f2", "金融", "氏名 中島", [("中島", "last_name")]),
    ]

    picked = pick_unique_values(documents, per_domain=1, seed=3)

    assert {entry["domain"] for entry in picked} == {"医療", "金融"}
    by_domain = {entry["domain"]: entry for entry in picked}
    assert by_domain["医療"] == {
        "value": "MRN-778899",
        "label": "medical_record_number",
        "domain": "医療",
        "uuid": "m1",
    }
    assert by_domain["金融"]["uuid"] == "f1"
    assert by_domain["金融"]["value"] in {"7654321", "a@example.jp"}


@pytest.mark.parametrize("profile", [FINANCE, PII])
def test_committed_fixtures_are_well_formed(profile) -> None:
    path: Path = fixture_path(profile)
    entries = json.loads(path.read_text(encoding="utf-8"))

    assert len(entries) == 40
    assert len({entry["uuid"] for entry in entries}) == 40
    key = "name" if profile is FINANCE else "value"
    assert len({entry[key] for entry in entries}) == 40
    if profile is PII:
        domains = [entry["domain"] for entry in entries]
        assert all(domains.count(domain) == 5 for domain in set(domains))
        assert len(set(domains)) == 8
