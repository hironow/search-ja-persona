from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from search_ja_persona.repository import PersonaRepository
from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS


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


def test_persona_application_orchestrates_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parquet_path = dataset_dir / "sample.parquet"
    _write_sample_dataset(parquet_path)

    recorded: dict[str, Any] = {}

    class FakeRepository:
        def __init__(self, paths):
            recorded["repository_paths"] = list(paths)

        def iter_personas(
            self, limit: int | None = None
        ):  # pragma: no cover - not used
            return iter([])

    class FakeIndexer:
        def __init__(self, **kwargs):
            recorded["indexer_init"] = kwargs

        def index(self, *, batch_size: int, limit: int | None) -> None:
            recorded["index_call"] = {"batch_size": batch_size, "limit": limit}

    class FakeServiceFactory:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, **kwargs: Any) -> Any:
            recorded.setdefault(f"{self.name}_init", []).append(kwargs)
            return object()

    class FakeSearchService:
        def __init__(self, **kwargs: Any) -> None:
            recorded["search_service_init"] = kwargs

        def search(self, query: str, limit: int, *, return_stats: bool = False):
            recorded["search_call"] = {"query": query, "limit": limit}
            results = [{"uuid": "1"}]
            stats = {
                "vector_hits": 1,
                "keyword_hits": 0,
                "context_calls": 0,
                "results": 1,
            }
            return (results, stats) if return_stats else results

    import search_ja_persona.application as app_module

    monkeypatch.setattr(app_module, "PersonaRepository", FakeRepository)
    monkeypatch.setattr(app_module, "PersonaIndexer", FakeIndexer)
    monkeypatch.setattr(app_module, "QdrantService", FakeServiceFactory("qdrant"))
    monkeypatch.setattr(
        app_module, "ElasticsearchService", FakeServiceFactory("elasticsearch")
    )
    monkeypatch.setattr(app_module, "Neo4jService", FakeServiceFactory("neo4j"))
    monkeypatch.setattr(app_module, "PersonaSearchService", FakeSearchService)

    from search_ja_persona.application import ApplicationConfig, PersonaApplication

    config = ApplicationConfig(vector_dimension=8, ngram_sizes=(2, 3))
    app = PersonaApplication.build(config)

    app.index([dataset_dir], batch_size=2, limit=1)
    results = app.search("介護", limit=1)

    assert Path(recorded["repository_paths"][0]) == parquet_path
    assert recorded["index_call"] == {"batch_size": 2, "limit": 1}
    assert recorded["indexer_init"]["embedder"].dimension == 8
    assert recorded["search_call"] == {"query": "介護", "limit": 1}
    assert results == [{"uuid": "1"}]
    assert recorded["indexer_init"]["persona_fields"] == PERSONA_TEXT_FIELDS


def test_persona_application_uses_sentence_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parquet_path = dataset_dir / "sample.parquet"
    _write_sample_dataset(parquet_path)

    recorded: dict[str, Any] = {}

    class FakeRepository:
        def __init__(self, paths):
            recorded["repository_paths"] = list(paths)

        def iter_personas(self, limit: int | None = None):  # pragma: no cover
            return iter([])

    class FakeIndexer:
        def __init__(self, **kwargs):
            recorded["indexer_init"] = kwargs

        def index(self, *, batch_size: int, limit: int | None) -> None:
            recorded["index_call"] = {"batch_size": batch_size, "limit": limit}

    class FakeSentenceEmbedder:
        def __init__(
            self, model_name: str, device=None, normalize_embeddings: bool = True
        ) -> None:
            recorded["embedder_args"] = {
                "model_name": model_name,
                "device": device,
                "normalize": normalize_embeddings,
            }
            self.dimension = 768

        def embed(self, text: str) -> list[float]:  # pragma: no cover
            return [0.0] * self.dimension

    import search_ja_persona.application as app_module

    monkeypatch.setattr(app_module, "PersonaRepository", FakeRepository)
    monkeypatch.setattr(app_module, "PersonaIndexer", FakeIndexer)
    monkeypatch.setattr(app_module, "SentenceTransformerEmbedder", FakeSentenceEmbedder)

    from search_ja_persona.application import ApplicationConfig, PersonaApplication

    config = ApplicationConfig(embedder="mpnet")
    app = PersonaApplication.build(config)

    app.index([dataset_dir], batch_size=2, limit=1)

    assert recorded["embedder_args"] == {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "device": None,
        "normalize": True,
    }
    assert recorded["indexer_init"]["embedder"].dimension == 768
    assert recorded["indexer_init"]["persona_fields"] == PERSONA_TEXT_FIELDS


def test_persona_application_uses_fast_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    parquet_path = dataset_dir / "sample.parquet"
    _write_sample_dataset(parquet_path)

    recorded: dict[str, Any] = {}

    class FakeRepository:
        def __init__(self, paths):
            recorded["repository_paths"] = list(paths)

        def iter_personas(self, limit: int | None = None):  # pragma: no cover
            return iter([])

    class FakeIndexer:
        def __init__(self, **kwargs):
            recorded["indexer_init"] = kwargs

        def index(self, *, batch_size: int, limit: int | None) -> None:
            recorded["index_call"] = {"batch_size": batch_size, "limit": limit}

    class FakeFastEmbedder:
        def __init__(
            self, model_name: str, cache_dir=None, normalize_embeddings: bool = True
        ) -> None:
            recorded["embedder_args"] = {
                "model_name": model_name,
                "cache_dir": cache_dir,
                "normalize": normalize_embeddings,
            }
            self.dimension = 256

        def embed(self, text: str) -> list[float]:  # pragma: no cover
            return [0.0] * self.dimension

    import search_ja_persona.application as app_module

    monkeypatch.setattr(app_module, "PersonaRepository", FakeRepository)
    monkeypatch.setattr(app_module, "PersonaIndexer", FakeIndexer)
    monkeypatch.setattr(app_module, "FastEmbedder", FakeFastEmbedder)

    from search_ja_persona.application import ApplicationConfig, PersonaApplication

    config = ApplicationConfig(
        embedder="fast-e5-small", fastembed_cache_dir="/tmp/cache"
    )
    app = PersonaApplication.build(config)

    app.index([dataset_dir], batch_size=2, limit=1)

    assert recorded["embedder_args"] == {
        "model_name": "intfloat/multilingual-e5-small",
        "cache_dir": "/tmp/cache",
        "normalize": True,
    }
    assert recorded["indexer_init"]["embedder"].dimension == 256
    assert recorded["indexer_init"]["persona_fields"] == PERSONA_TEXT_FIELDS
