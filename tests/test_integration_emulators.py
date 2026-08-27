from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from uuid import uuid4

import pytest

from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.repository import PersonaRepository
from search_ja_persona.services import RequestDescriptor, SimpleHttpTransport

pytest.importorskip("datasets")
from datasets import DownloadConfig, load_dataset
from datasets import config as datasets_config

pytestmark = pytest.mark.integration

_DATASET_COLUMNS = (
    "uuid",
    "persona",
    "prefecture",
    "region",
    "occupation",
    "age",
    "sex",
)
_DEFAULT_CACHE_DIR = Path(".hf-cache")


def _ensure_emulators_available(config: ApplicationConfig) -> None:
    qdrant_transport = SimpleHttpTransport(
        config.qdrant_host, config.qdrant_port, timeout=2.0
    )
    try:
        qdrant_transport.request(RequestDescriptor("GET", "/collections"))
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Qdrant emulator unavailable: {exc}")

    elastic_transport = SimpleHttpTransport(config.es_host, config.es_port, timeout=2.0)
    try:
        elastic_transport.request(RequestDescriptor("GET", "/"))
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Elasticsearch emulator unavailable: {exc}")

    neo4j_transport = SimpleHttpTransport(
        config.neo4j_host,
        config.neo4j_port,
        timeout=2.0,
        auth=(config.neo4j_user, config.neo4j_password),
    )
    try:
        neo4j_transport.request(
            RequestDescriptor(
                method="POST",
                path="/db/neo4j/tx/commit",
                body={"statements": [{"statement": "RETURN 1"}]},
            )
        )
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Neo4j emulator unavailable: {exc}")


def _load_sample_rows(limit: int = 10) -> list[dict]:
    sample_file_env = os.getenv("PERSONA_DATASET_SAMPLE_PATH")
    if sample_file_env:
        sample_file = Path(sample_file_env)
        if not sample_file.exists():
            pytest.skip(f"Dataset sample path not found: {sample_file}")
        repository = PersonaRepository([sample_file])
        return list(repository.iter_personas(limit=limit))

    sample_dir_env = os.getenv("PERSONA_DATASET_DIR")
    if sample_dir_env:
        sample_dir = Path(sample_dir_env)
        if not sample_dir.exists():
            pytest.skip(f"Dataset directory not found: {sample_dir}")
        parquet_files = sorted(sample_dir.glob("*.parquet"))
        if not parquet_files:
            pytest.skip("No parquet files found in provided dataset directory")
        repository = PersonaRepository(parquet_files)
        return list(repository.iter_personas(limit=limit))

    cache_roots: list[Path] = []
    cache_dir_env = os.getenv("PERSONA_DATASET_CACHE_DIR")
    if cache_dir_env:
        cache_roots.append(Path(cache_dir_env).expanduser())
    if _DEFAULT_CACHE_DIR.exists():
        cache_roots.append(_DEFAULT_CACHE_DIR.resolve())
    cache_roots.append(Path(datasets_config.HF_DATASETS_CACHE).expanduser())

    cache_root: Path | None = None
    for candidate_root in cache_roots:
        if not candidate_root.exists():
            continue
        if list(candidate_root.glob("nvidia___nemotron-personas-japan/*")):
            cache_root = candidate_root
            break

    if cache_root is None:
        pytest.skip(
            "Persona dataset cache not found. Provide PERSONA_DATASET_SAMPLE_PATH, "
            "PERSONA_DATASET_DIR, or run the download-dataset CLI."
        )

    try:
        dataset = load_dataset(
            "nvidia/Nemotron-Personas-Japan",
            split=f"train[:{limit}]",
            streaming=False,
            download_mode="reuse_cache_if_exists",
            download_config=DownloadConfig(
                cache_dir=str(cache_root),
                local_files_only=True,
            ),
        )
    except Exception as exc:  # noqa: BLE001 # pragma: no cover - any cache problem -> skip
        pytest.skip(f"Dataset cache unavailable: {exc}")

    rows: list[dict] = []
    for item in dataset:
        row = {column: item.get(column) for column in _DATASET_COLUMNS}
        rows.append(row)
    if not rows:
        pytest.skip("Dataset returned no rows")
    return rows


def _retry_search(
    app: PersonaApplication, query: str, *, attempts: int = 5, delay: float = 0.5
) -> list[dict]:
    for _ in range(attempts):
        results = app.search(query, limit=1)
        if results:
            return results
        time.sleep(delay)
    return []


def _cleanup_resources(app: PersonaApplication, uuids: list[str]) -> None:
    with contextlib.suppress(Exception):  # cleanup is best effort
        app.qdrant.transport.request(
            RequestDescriptor("DELETE", f"/collections/{app.qdrant.collection}")
        )

    with contextlib.suppress(Exception):  # cleanup is best effort
        app.elasticsearch.transport.request(
            RequestDescriptor("DELETE", f"/{app.elasticsearch.index}")
        )

    if not uuids:
        return
    with contextlib.suppress(Exception):  # cleanup is best effort
        app.neo4j.transport.request(
            RequestDescriptor(
                method="POST",
                path="/db/neo4j/tx/commit",
                body={
                    "statements": [
                        {
                            "statement": "MATCH (p:Persona) WHERE p.uuid IN $uuids DETACH DELETE p",
                            "parameters": {"uuids": uuids},
                        }
                    ]
                },
            )
        )


def test_index_and_search_with_emulators(tmp_path: Path) -> None:
    sample_rows = _load_sample_rows(limit=5)
    # Neo4j has no per-test database: with the dataset's real uuids the
    # teardown's DETACH DELETE would remove live personas from the shared
    # graph. Re-key the rows so cleanup only ever touches test data.
    for row in sample_rows:
        row["uuid"] = uuid4().hex
    parquet_path = tmp_path / "sample.parquet"
    PersonaRepository.write_sample(parquet_path, sample_rows)

    unique_suffix = uuid4().hex[:12]
    config = ApplicationConfig(
        qdrant_collection=f"test_personas_{unique_suffix}",
        es_index=f"test-personas-{unique_suffix}",
    )
    _ensure_emulators_available(config)

    app = PersonaApplication.build(config)
    uuids = [row["uuid"] for row in sample_rows if row.get("uuid")]

    try:
        app.index([parquet_path], batch_size=len(sample_rows))

        query_text = sample_rows[0]["persona"] or ""
        assert query_text

        results = _retry_search(app, query_text)
        assert results, "No results returned from combined search"
        assert results[0]["uuid"] == sample_rows[0]["uuid"]
    finally:
        _cleanup_resources(app, uuids)


def test_prefecture_filtered_search_with_emulators(tmp_path: Path) -> None:
    rows = [
        {
            "uuid": uuid4().hex,
            "persona": "スキーとスノーボードを楽しむ会社員",
            "prefecture": "北海道",
            "region": "北海道地方",
        },
        {
            "uuid": uuid4().hex,
            "persona": "スキーとスノーボードを楽しむ会社員",
            "prefecture": "東京都",
            "region": "関東地方",
        },
    ]
    parquet_path = tmp_path / "sample.parquet"
    PersonaRepository.write_sample(parquet_path, rows)

    unique_suffix = uuid4().hex[:12]
    config = ApplicationConfig(
        qdrant_collection=f"test_personas_{unique_suffix}",
        es_index=f"test-personas-{unique_suffix}",
    )
    _ensure_emulators_available(config)
    app = PersonaApplication.build(config)
    uuids = [row["uuid"] for row in rows]

    try:
        app.index([parquet_path], batch_size=len(rows))

        # The payload index is created at index time; the migration path
        # (a second ensure) must be an idempotent noop.
        info = app.qdrant.transport.request(
            RequestDescriptor("GET", f"/collections/{app.qdrant.collection}")
        )
        schema = info.get("result", {}).get("payload_schema", {})
        assert schema.get("prefecture", {}).get("data_type") == "keyword"
        assert app.qdrant.ensure_payload_index() == {"status": "exists"}

        # Elasticsearch answers from refreshed segments only.
        app.elasticsearch.transport.request(
            RequestDescriptor("POST", f"/{app.elasticsearch.index}/_refresh")
        )

        query = "スキーが好きな人"
        vector = app.embedder.embed_query(query)
        vector_hits = app.qdrant.search(vector, limit=5, prefecture="北海道")
        assert vector_hits
        assert all(
            hit.get("payload", {}).get("prefecture") == "北海道" for hit in vector_hits
        )

        keyword_hits = (
            app.elasticsearch.search(query, limit=5, prefecture="北海道")
            .get("hits", {})
            .get("hits", [])
        )
        assert keyword_hits
        assert all(
            hit.get("_source", {}).get("prefecture") == "北海道" for hit in keyword_hits
        )

        fused = app.search(query, limit=5, prefecture="北海道")
        assert fused
        assert all(hit.get("prefecture") == "北海道" for hit in fused)
    finally:
        _cleanup_resources(app, uuids)
