from __future__ import annotations

from pathlib import Path

from scripts.catalog_snapshot import SNAPSHOT_QUERIES, build_snapshot, snapshot_path
from search_ja_persona.dataset_profiles import FINANCE, JAPAN, PII, PROFILES

MARIMO = Path(__file__).resolve().parents[1] / "marimo"


def test_each_dataset_has_its_own_snapshot_file() -> None:
    assert snapshot_path(JAPAN) == MARIMO / "catalog_snapshot.json"
    assert snapshot_path(FINANCE) == MARIMO / "catalog_snapshot.finance.json"
    assert snapshot_path(PII) == MARIMO / "catalog_snapshot.pii.json"
    assert set(SNAPSHOT_QUERIES) == set(PROFILES)


def test_snapshot_rows_carry_the_profiles_payload_fields() -> None:
    hits = [
        {
            "score": 0.912345,
            "text": "審" * 300,
            "domain": "金融",
            "document_type": "ローン審査書類",
            "prefecture": "ignored",
        }
    ]

    snapshot = build_snapshot(
        PII,
        [("ローンの審査書類", 123.4, hits)],
        embedder="ruri-v3-310m",
        dimension=768,
        corpus_size=100_000,
        generated_at="2026-11-11T00:00:00+00:00",
    )

    assert snapshot["dataset"] == "pii"
    assert snapshot["embedding_text_policy"] == "raw-text-v1"
    assert snapshot["corpus_size"] == 100_000
    (entry,) = snapshot["queries"]
    assert entry["latency_ms"] == 123
    assert entry["results"] == [
        {
            "score": 0.9123,
            "domain": "金融",
            "document_type": "ローン審査書類",
            "excerpt": "審" * 200,
        }
    ]
