from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import pytest

from search_ja_persona import cli
from search_ja_persona.repository import PersonaRepository


def _write_sample_dataset(path: Path) -> None:
    PersonaRepository.write_sample(
        path,
        [
            {
                "uuid": "1",
                "persona": "東京で介護に従事するリーダー",
                "prefecture": "東京都",
                "region": "関東地方",
            }
        ],
    )


def test_cli_index_invokes_indexer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    _write_sample_dataset(parquet_path)

    recorded: dict[str, Any] = {}

    class FakeRepository:
        def __init__(self, paths):
            recorded["paths"] = list(paths)

        def iter_personas(self, limit: int | None = None):  # pragma: no cover - not used
            return iter([])

    class FakeIndexer:
        def __init__(self, **kwargs):
            recorded["indexer_init"] = kwargs

        def index(self, *, batch_size: int, limit: int | None) -> None:
            recorded["index_call"] = {"batch_size": batch_size, "limit": limit}

    class FakeService:
        def __init__(self, **kwargs):
            recorded.setdefault("services", []).append(kwargs)

    monkeypatch.setattr(cli, "PersonaRepository", FakeRepository)
    monkeypatch.setattr(cli, "PersonaIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "QdrantService", FakeService)
    monkeypatch.setattr(cli, "ElasticsearchService", FakeService)
    monkeypatch.setattr(cli, "Neo4jService", FakeService)

    cli.main(
        [
            "index",
            "--dataset",
            str(parquet_path),
            "--batch-size",
            "2",
            "--limit",
            "1",
            "--vector-dimension",
            "8",
        ]
    )

    assert recorded["paths"] == [parquet_path]
    assert recorded["index_call"] == {"batch_size": 2, "limit": 1}
    # ensure services are initialised (three entries: qdrant, elastic, neo4j)
    assert len(recorded["services"]) == 3
    assert recorded["indexer_init"]["vectorizer"].dimension == 8


def test_cli_search_outputs_results(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeSearchService:
        def __init__(self, **_: Any) -> None:
            pass

        def search(self, query: str, limit: int) -> list[dict[str, Any]]:
            return [
                {
                    "uuid": "1",
                    "score": 0.9,
                    "text": "東京で介護に従事するリーダー",
                    "prefecture": "東京都",
                    "region": "関東地方",
                    "context": {"relationships": []},
                }
            ]

    monkeypatch.setattr(cli, "PersonaSearchService", FakeSearchService)

    cli.main(["search", "--query", "介護", "--limit", "1"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload[0]["uuid"] == "1"
    assert payload[0]["prefecture"] == "東京都"


def test_cli_download_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = {}

    def fake_cache(config):
        recorded["config"] = config

    monkeypatch.setattr(cli.datasets, "ensure_dataset_cached", fake_cache)

    cache_dir = tmp_path / "cache"

    cli.main(
        [
            "download-dataset",
            "--cache-dir",
            str(cache_dir),
            "--force",
        ]
    )

    config = recorded["config"]
    assert config.cache_dir == cache_dir
    assert config.force_download is True
