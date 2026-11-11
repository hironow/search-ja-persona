"""Each dataset profile reads and writes only its own stores and metadata.

Every HTTP call is captured at ``SimpleHttpTransport.request`` so the
assertions see exactly what would reach Qdrant, Elasticsearch, and Neo4j.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from search_ja_persona import cli
from search_ja_persona.repository import PersonaRepository

FINANCE_TEXT_FIELDS = [
    "finance_persona",
    "financial_behavior_persona",
    "financial_concerns_persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
]

FINANCE_ROWS: list[dict[str, Any]] = [
    {
        "uuid": "3b38da9499de5ace8a229ad5dcee8a45",
        **{field: f"山辺 楓人は、{field} の記述だ。" for field in FINANCE_TEXT_FIELDS},
        "prefecture": "大阪府",
        "region": "近畿地方",
        "occupation": "販売店員",
        "age": 54,
        "sex": "男",
    }
]


def _finance_metadata_path() -> Path:
    return cli.METADATA_PATH.with_name("index_metadata.finance.json")


def _index_finance(tmp_path: Path, *extra: str) -> None:
    dataset = tmp_path / "finance.parquet"
    PersonaRepository.write_sample(dataset, FINANCE_ROWS)
    cli.main(
        [
            "index",
            "--dataset-profile",
            "finance",
            "--dataset",
            str(dataset),
            "--embedder",
            "hashed",
            "--vector-dimension",
            "8",
            "--persona-fields",
            "all",
            *extra,
        ]
    )


def _cypher(requests: list[dict[str, Any]]) -> list[str]:
    return [
        statement["statement"]
        for request in requests
        if request["path"] == "/db/neo4j/tx/commit"
        for statement in request["body"]["statements"]
    ]


def _store_paths(requests: list[dict[str, Any]]) -> list[str]:
    return [r["path"] for r in requests if r["path"] != "/db/neo4j/tx/commit"]


def test_finance_index_touches_only_finance_stores(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_finance(tmp_path)

    for path in _store_paths(store_requests):
        assert path.startswith(("/collections/finance_personas", "/finance_personas"))
    statements = _cypher(store_requests)
    assert statements
    for statement in statements:
        # \bPersona\b matches the japan label (":Persona") but not
        # "FinancePersona".
        assert not re.search(r"\bPersona\b", statement), statement
    assert any("FinancePersona" in statement for statement in statements)
    assert any(
        "CREATE CONSTRAINT finance_persona_uuid" in statement
        for statement in statements
    )


def test_finance_index_records_its_own_metadata(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_finance(tmp_path)

    assert not cli.METADATA_PATH.exists()
    metadata = json.loads(_finance_metadata_path().read_text(encoding="utf-8"))
    assert metadata["dataset"] == "finance"
    assert metadata["qdrant"]["collection"] == "finance_personas"
    assert metadata["embedder"]["persona_fields"] == FINANCE_TEXT_FIELDS


def test_finance_documents_carry_all_nine_text_fields(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_finance(tmp_path)

    bulk = next(r for r in store_requests if r["path"].endswith("/_bulk"))
    document = json.loads(bulk["body"].splitlines()[1])
    for field in FINANCE_TEXT_FIELDS:
        assert document[field] == FINANCE_ROWS[0][field]
    mapping = next(
        r
        for r in store_requests
        if r["method"] == "PUT" and r["path"] == "/finance_personas"
    )
    properties = mapping["body"]["mappings"]["properties"]
    assert all(properties[field] == {"type": "text"} for field in FINANCE_TEXT_FIELDS)


def test_clearing_finance_leaves_japan_stores_and_metadata(
    store_requests: list[dict[str, Any]],
) -> None:
    cli.METADATA_PATH.write_text('{"dataset": "japan"}', encoding="utf-8")
    _finance_metadata_path().write_text('{"dataset": "finance"}', encoding="utf-8")

    cli.main(["clear-emulators", "--dataset-profile", "finance"])

    assert _store_paths(store_requests) == [
        "/collections/finance_personas",
        "/finance_personas",
    ]
    assert all(
        r["method"] == "DELETE"
        for r in store_requests
        if r["path"] != "/db/neo4j/tx/commit"
    )
    (statement,) = _cypher(store_requests)
    assert statement.startswith("MATCH (p:FinancePersona)")
    assert cli.METADATA_PATH.exists()
    assert not _finance_metadata_path().exists()


@pytest.mark.parametrize(
    "argv",
    [
        [
            "search",
            "--dataset-profile",
            "finance",
            "--query",
            "q",
            "--qdrant-collection",
            "personas",
        ],
        [
            "search",
            "--dataset-profile",
            "finance",
            "--query",
            "q",
            "--es-index",
            "personas",
        ],
        ["search", "--query", "q", "--qdrant-collection", "finance_personas"],
        ["clear-emulators", "--dataset-profile", "finance", "--es-index", "personas"],
        [
            "repair-metadata",
            "--dataset-profile",
            "finance",
            "--embedder",
            "hashed",
            "--qdrant-collection",
            "personas",
        ],
    ],
)
def test_a_profile_refuses_another_profiles_store_names(
    store_requests: list[dict[str, Any]],
    argv: list[str],
    printed: Callable[[], str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([*argv, "--embedder", "hashed"] if argv[0] == "search" else argv)

    assert excinfo.value.code == 2
    assert "belongs to another dataset profile" in printed()
    assert store_requests == []


def test_refused_index_sends_nothing(
    store_requests: list[dict[str, Any]],
    tmp_path: Path,
    printed: Callable[[], str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _index_finance(tmp_path, "--qdrant-collection", "personas")

    assert excinfo.value.code == 2
    assert "belongs to another dataset profile" in printed()
    assert store_requests == []


def test_clearing_a_custom_collection_keeps_the_default_metadata(
    store_requests: list[dict[str, Any]],
) -> None:
    cli.METADATA_PATH.write_text('{"dataset": "japan"}', encoding="utf-8")

    cli.main(
        ["clear-emulators", "--qdrant-collection", "scratch", "--es-index", "scratch"]
    )

    assert cli.METADATA_PATH.exists()
    assert "/collections/scratch" in _store_paths(store_requests)


def test_metadata_recorded_for_another_dataset_fails_closed(
    store_requests: list[dict[str, Any]], printed: Callable[[], str]
) -> None:
    japan_metadata = {
        "dataset": "japan",
        "embedder": {
            "preset": "hashed",
            "type": "hashed",
            "persona_fields": ["persona"],
        },
        "qdrant": {"collection": "finance_personas", "distance": "Cosine"},
        "schema_version": cli.INDEX_METADATA_SCHEMA_VERSION,
    }
    _finance_metadata_path().write_text(json.dumps(japan_metadata), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["search", "--dataset-profile", "finance", "--query", "q"])

    assert excinfo.value.code == 2
    assert "describes dataset 'japan', not 'finance'" in printed()
    assert store_requests == []


def test_finance_search_reuses_finance_metadata(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_finance(tmp_path)
    store_requests.clear()

    cli.main(
        [
            "search",
            "--dataset-profile",
            "finance",
            "--query",
            "住宅ローンに不安がある会社員",
            "--prefecture",
            "大阪府",
            "--format",
            "json",
        ]
    )

    assert _store_paths(store_requests) == [
        "/collections/finance_personas/points/search",
        "/finance_personas/_search",
    ]
