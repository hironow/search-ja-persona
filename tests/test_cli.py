from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import json
from rich.console import Console

import search_ja_persona.cli as cli
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
    monkeypatch.setattr(cli, "METADATA_PATH", tmp_path / "metadata.json")
    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)

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
    assert recorded["indexer_init"]["embedder"].dimension == 8
    assert recorded["indexer_init"]["persona_fields"] == ("persona",)


def test_cli_search_outputs_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeSearchService:
        def __init__(self, **_: Any) -> None:
            pass

        def search(self, query: str, limit: int, *, return_stats: bool = False):
            results = [
                {
                    "uuid": "1",
                    "score": 0.9,
                    "text": "東京で介護に従事するリーダー",
                    "prefecture": "東京都",
                    "region": "関東地方",
                    "context": {"relationships": []},
                    "persona_fields": {"persona": "東京で介護に従事するリーダー"},
                }
            ]
            if return_stats:
                return results, {"vector_hits": 1, "keyword_hits": 1, "context_calls": 1, "results": 1}
            return results

    monkeypatch.setattr(cli, "PersonaSearchService", FakeSearchService)
    monkeypatch.setattr(cli, "METADATA_PATH", tmp_path / "metadata.json")

    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)

    cli.main(["search", "--query", "介護", "--limit", "1", "--format", "json"])

    payload = json.loads(test_console.export_text().strip())
    assert payload[0]["uuid"] == "1"
    assert payload[0]["prefecture"] == "東京都"
    assert payload[0]["persona_fields"]["persona"] == "東京で介護に従事するリーダー"


def test_cli_search_verbose_outputs_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeSearchService:
        def __init__(self, **_: Any) -> None:
            pass

        def search(self, query: str, limit: int, *, return_stats: bool = False):
            results = [
                {
                    "uuid": "1",
                    "score": 0.1,
                    "text": "sample",
                    "prefecture": "東京都",
                    "region": "関東地方",
                    "persona_fields": {"persona": "sample"},
                }
            ]
            stats = {"vector_hits": 2, "keyword_hits": 3, "context_calls": 1, "results": 1}
            return (results, stats) if return_stats else results

    monkeypatch.setattr(cli, "PersonaSearchService", FakeSearchService)
    monkeypatch.setattr(cli, "METADATA_PATH", tmp_path / "metadata.json")

    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)

    cli.main(["search", "--query", "テスト", "--limit", "1", "--verbose"])

    output = test_console.export_text()
    assert "Qdrant candidates: 2" in output
    assert "Returned 1 combined result" in output
    assert "Search Results" in output


def test_cli_download_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorded = {}

    def fake_cache(config):
        recorded["config"] = config

    monkeypatch.setattr(cli.datasets, "ensure_dataset_cached", fake_cache)

    cache_dir = tmp_path / "cache"

    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)
    monkeypatch.setattr(cli, "METADATA_PATH", tmp_path / "metadata.json")

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

def test_cli_index_sentence_embedder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(cli, "METADATA_PATH", tmp_path / "metadata.json")

    class FakeSentenceEmbedder:
        def __init__(self, model_name: str, device=None, normalize_embeddings: bool = True) -> None:
            recorded["sentence_config"] = {
                "model_name": model_name,
                "device": device,
                "normalize": normalize_embeddings,
            }
            self.dimension = 384

        def embed(self, text: str) -> list[float]:  # pragma: no cover - not used
            return [0.0] * self.dimension

    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", FakeSentenceEmbedder)
    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)

    cli.main(
        [
            "index",
            "--dataset",
            str(parquet_path),
            "--embedder",
            "mini-lm",
            "--embedder-model",
            "fake/small",
            "--embedder-device",
            "cpu",
            "--no-embedder-normalize",
        ]
    )

    assert recorded["sentence_config"] == {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "normalize": False,
    }
    assert recorded["indexer_init"]["embedder"].dimension == 384
    assert recorded["indexer_init"]["persona_fields"] == ("persona",)

def test_cli_index_fast_embedder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(cli, "METADATA_PATH", tmp_path / "metadata.json")

    class FakeFastEmbedder:
        def __init__(self, model_name: str, cache_dir=None, normalize_embeddings: bool = True) -> None:
            recorded["fast_config"] = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "normalize": normalize_embeddings,
            }
            self.dimension = 512

        def embed(self, text: str) -> list[float]:  # pragma: no cover - not used
            return [0.0] * self.dimension

    monkeypatch.setattr(cli, "FastEmbedder", FakeFastEmbedder)
    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)

    cli.main(
        [
            "index",
            "--dataset",
            str(parquet_path),
            "--embedder",
            "fast-e5-small",
            "--fastembed-cache-dir",
            "/tmp/cache",
            "--no-embedder-normalize",
        ]
    )

    assert recorded["fast_config"] == {
        "model_name": "intfloat/multilingual-e5-small",
        "cache_dir": "/tmp/cache",
        "normalize": False,
    }
    assert recorded["indexer_init"]["embedder"].dimension == 512
    assert recorded["indexer_init"]["persona_fields"] == ("persona",)

def test_cli_index_prompt_reset_decline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    _write_sample_dataset(parquet_path)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "embedder": {
                    "preset": "hashed",
                    "type": "hashed",
                    "vector_dimension": 256,
                    "ngram_sizes": [2, 3],
                    "normalize": True,
                    "persona_fields": ["persona"],
                }
            }
        )
    )

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
    monkeypatch.setattr(cli, "METADATA_PATH", metadata_path)

    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)
    test_console.input = lambda prompt="": "n"

    cli.main([
        "index",
        "--dataset",
        str(parquet_path),
        "--embedder",
        "mini-lm",
    ])

    output = test_console.export_text()
    assert "Indexing aborted" in output
    assert "index_call" not in recorded


def test_cli_index_prompt_reset_accept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    _write_sample_dataset(parquet_path)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "embedder": {
                    "preset": "hashed",
                    "type": "hashed",
                    "vector_dimension": 256,
                    "ngram_sizes": [2, 3],
                    "normalize": True,
                    "persona_fields": ["persona"],
                }
            }
        )
    )

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

    def fake_reset(q, e, n, a):
        recorded["reset_called"] = True

    monkeypatch.setattr(cli, "PersonaRepository", FakeRepository)
    monkeypatch.setattr(cli, "PersonaIndexer", FakeIndexer)
    monkeypatch.setattr(cli, "QdrantService", FakeService)
    monkeypatch.setattr(cli, "ElasticsearchService", FakeService)
    monkeypatch.setattr(cli, "Neo4jService", FakeService)
    monkeypatch.setattr(cli, "_reset_indexes", fake_reset)
    monkeypatch.setattr(cli, "METADATA_PATH", metadata_path)

    test_console = Console(record=True)
    monkeypatch.setattr(cli, "console", test_console)
    test_console.input = lambda prompt="": "y"

    cli.main([
        "index",
        "--dataset",
        str(parquet_path),
        "--embedder",
        "mini-lm",
    ])

    assert recorded.get("reset_called") is True
    assert "index_call" in recorded
    assert recorded["indexer_init"]["persona_fields"] == ("persona",)
