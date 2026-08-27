"""CLI behavior when the index metadata file is missing or unusable.

The metadata file is the only record of which embedder built the live
index; these tests pin the fail-closed behavior that replaces the old
silent fallback to the hashed default (which produced an opaque Qdrant
dimension error against a sentence-embedder index).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from search_ja_persona import cli


class _BlockedService:
    """Backend service that must never be constructed."""

    def __init__(self, **_: Any) -> None:
        raise AssertionError("no backend service may be constructed")


def _isolate_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    metadata_path = tmp_path / "index_metadata.json"
    monkeypatch.setattr(cli, "METADATA_PATH", metadata_path)
    return metadata_path


def _record_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    console = Console(record=True)
    monkeypatch.setattr(cli, "console", console)
    return console


def _block_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "QdrantService", _BlockedService)
    monkeypatch.setattr(cli, "ElasticsearchService", _BlockedService)
    monkeypatch.setattr(cli, "Neo4jService", _BlockedService)


def test_search_fails_fast_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_metadata(monkeypatch, tmp_path)
    _block_backends(monkeypatch)
    console = _record_console(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["search", "--query", "介護"])

    assert excinfo.value.code == 2
    assert "repair-metadata" in console.export_text()


class _AcceptingService:
    def __init__(self, **_: Any) -> None:
        pass


class _EmptySearchService:
    def __init__(self, **_: Any) -> None:
        pass

    def search(
        self,
        query: str,
        limit: int,
        *,
        return_stats: bool = False,
        prefecture: str | None = None,
    ) -> list[dict[str, Any]]:
        return []


def _allow_search_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "QdrantService", _AcceptingService)
    monkeypatch.setattr(cli, "ElasticsearchService", _AcceptingService)
    monkeypatch.setattr(cli, "Neo4jService", _AcceptingService)
    monkeypatch.setattr(cli, "PersonaSearchService", _EmptySearchService)


def test_search_fails_fast_when_metadata_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    metadata_path.write_text(
        json.dumps({"schema_version": "1999-01-01", "embedder": {"preset": "hashed"}}),
        encoding="utf-8",
    )
    _block_backends(monkeypatch)
    console = _record_console(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["search", "--query", "介護"])

    assert excinfo.value.code == 2
    assert "unusable" in console.export_text()


def test_search_with_explicit_embedder_proceeds_without_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_metadata(monkeypatch, tmp_path)
    _allow_search_backends(monkeypatch)
    _record_console(monkeypatch)

    cli.main(["search", "--query", "介護", "--embedder", "hashed", "--format", "json"])


def test_search_reuses_usable_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": cli.INDEX_METADATA_SCHEMA_VERSION,
                "embedder": {
                    "preset": "hashed",
                    "type": "hashed",
                    "vector_dimension": 8,
                    "ngram_sizes": [2, 3],
                    "persona_fields": ["persona"],
                },
            }
        ),
        encoding="utf-8",
    )
    _allow_search_backends(monkeypatch)
    console = _record_console(monkeypatch)

    cli.main(["search", "--query", "介護", "--format", "json"])

    assert "recorded in" in console.export_text()


class _StatsTransport:
    """Answers the index-stats probes; None points = collection absent."""

    def __init__(self, points: int | None) -> None:
        self.points = points
        self.requests: list[Any] = []

    def request(self, descriptor: Any) -> dict[str, Any]:
        self.requests.append(descriptor)
        path = descriptor.path
        if path.startswith("/collections/"):
            if self.points is None:
                raise RuntimeError("collection missing")
            return {
                "result": {
                    "points_count": self.points,
                    "config": {"params": {"vectors": {"size": 8}}},
                }
            }
        if path.endswith("/_count"):
            return {"count": self.points or 0}
        return {"results": [{"data": [{"row": [self.points or 0]}]}]}


class _BlockedIndexer:
    def __init__(self, **_: Any) -> None:
        raise AssertionError("indexing must not start")


def _stats_backends(
    monkeypatch: pytest.MonkeyPatch, transport: _StatsTransport
) -> None:
    from types import SimpleNamespace

    def _service(**kwargs: Any) -> Any:
        return SimpleNamespace(
            transport=transport,
            collection=kwargs.get("collection", "personas"),
            index=kwargs.get("index", "personas"),
        )

    monkeypatch.setattr(cli, "QdrantService", _service)
    monkeypatch.setattr(cli, "ElasticsearchService", _service)
    monkeypatch.setattr(cli, "Neo4jService", _service)


def _write_parquet(tmp_path: Path) -> Path:
    from search_ja_persona.repository import PersonaRepository

    parquet = tmp_path / "sample.parquet"
    PersonaRepository.write_sample(
        parquet,
        [
            {
                "uuid": "1",
                "persona": "テスト",
                "prefecture": "東京都",
                "region": "関東地方",
            }
        ],
    )
    return parquet


def test_index_fails_closed_when_metadata_missing_but_store_populated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_metadata(monkeypatch, tmp_path)
    console = _record_console(monkeypatch)
    transport = _StatsTransport(points=5)
    _stats_backends(monkeypatch, transport)
    monkeypatch.setattr(cli, "PersonaIndexer", _BlockedIndexer)
    parquet = _write_parquet(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["index", "--dataset", str(parquet)])

    assert excinfo.value.code == 2
    assert "repair-metadata" in console.export_text()
    assert not [d for d in transport.requests if d.method == "DELETE"]


def test_index_proceeds_on_fresh_store_without_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    _record_console(monkeypatch)
    transport = _StatsTransport(points=None)
    _stats_backends(monkeypatch, transport)
    recorded: dict[str, Any] = {}

    class _RecordingIndexer:
        def __init__(self, **kwargs: Any) -> None:
            recorded["init"] = kwargs

        def index(self, *, batch_size: int, limit: int | None) -> None:
            recorded["run"] = {"batch_size": batch_size, "limit": limit}

    monkeypatch.setattr(cli, "PersonaIndexer", _RecordingIndexer)
    parquet = _write_parquet(tmp_path)

    cli.main(["index", "--dataset", str(parquet), "--limit", "1"])

    assert recorded["run"]["limit"] == 1
    assert metadata_path.exists()


class _RepairTransport:
    """Read-only Qdrant fake: any mutating request is a test failure."""

    def __init__(self, size: int, payload_fields: list[str]) -> None:
        self.size = size
        self.payload_fields = payload_fields
        self.requests: list[Any] = []

    def request(self, descriptor: Any) -> dict[str, Any]:
        self.requests.append(descriptor)
        if descriptor.method == "GET":
            return {
                "result": {
                    "points_count": 5,
                    "config": {"params": {"vectors": {"size": self.size}}},
                }
            }
        if descriptor.path.endswith("/points/scroll"):
            return {
                "result": {
                    "points": [
                        {
                            "payload": {
                                "persona_fields": dict.fromkeys(
                                    self.payload_fields, "x"
                                )
                            }
                        }
                    ]
                }
            }
        raise AssertionError(
            f"unexpected request {descriptor.method} {descriptor.path}"
        )


def _repair_backend(
    monkeypatch: pytest.MonkeyPatch, transport: _RepairTransport
) -> None:
    from types import SimpleNamespace

    def _qdrant(**kwargs: Any) -> Any:
        return SimpleNamespace(
            transport=transport, collection=kwargs.get("collection", "personas")
        )

    monkeypatch.setattr(cli, "QdrantService", _qdrant)


def test_repair_metadata_records_verified_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    _record_console(monkeypatch)
    _repair_backend(monkeypatch, _RepairTransport(size=8, payload_fields=["persona"]))

    cli.main(
        [
            "repair-metadata",
            "--embedder",
            "hashed",
            "--vector-dimension",
            "8",
            "--persona-fields",
            "persona",
        ]
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == cli.INDEX_METADATA_SCHEMA_VERSION
    assert metadata["embedder"]["preset"] == "hashed"
    assert metadata["embedder"]["persona_fields"] == ["persona"]
    assert metadata["qdrant"]["collection"] == "personas"


def test_repair_metadata_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    console = _record_console(monkeypatch)
    _repair_backend(monkeypatch, _RepairTransport(size=768, payload_fields=["persona"]))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "repair-metadata",
                "--embedder",
                "hashed",
                "--vector-dimension",
                "8",
                "--persona-fields",
                "persona",
            ]
        )

    assert excinfo.value.code == 2
    assert not metadata_path.exists()
    assert "768" in console.export_text()


def test_repair_metadata_rejects_persona_fields_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    _record_console(monkeypatch)
    _repair_backend(
        monkeypatch,
        _RepairTransport(size=8, payload_fields=["persona", "sports_persona"]),
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "repair-metadata",
                "--embedder",
                "hashed",
                "--vector-dimension",
                "8",
                "--persona-fields",
                "persona",
            ]
        )

    assert excinfo.value.code == 2
    assert not metadata_path.exists()


def test_index_metadata_records_embedding_text_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_path = _isolate_metadata(monkeypatch, tmp_path)
    _record_console(monkeypatch)
    transport = _StatsTransport(points=None)
    _stats_backends(monkeypatch, transport)

    class _NoopIndexer:
        def __init__(self, **_: Any) -> None:
            pass

        def index(self, *, batch_size: int, limit: int | None) -> None:
            pass

    monkeypatch.setattr(cli, "PersonaIndexer", _NoopIndexer)
    parquet = _write_parquet(tmp_path)

    cli.main(["index", "--dataset", str(parquet), "--limit", "1"])

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["embedder"]["embedding_text_policy"] == "strip-person-names-v1"
