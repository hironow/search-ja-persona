"""The pii profile indexes Nemotron-PII-Japan documents into their own stores."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from search_ja_persona import cli
from search_ja_persona.dataset_profiles import PII, detect_profile
from search_ja_persona.embeddings import HashedNgramEmbedder
from search_ja_persona.indexer import PersonaIndexer
from search_ja_persona.repository import PersonaRepository

PII_ROWS: list[dict[str, Any]] = [
    {
        "uid": "5ede5c5d6bcc493683495725bbfff54b",
        "domain": "金融",
        "document_type": "ローン審査書類",
        "document_description": "申込者と審査結果を記載する審査報告書です。",
        "document_format": "unstructured",
        "locale": "jp",
        "text": "審査受付日: 令和6年7月22日 申込者は中島 様、口座番号は1234567です。",
        "spans": [
            {"start": 7, "end": 16, "text": "令和6年7月22日", "label": "date"},
            {"start": 21, "end": 23, "text": "中島", "label": "last_name"},
            {"start": 32, "end": 39, "text": "1234567", "label": "account_number"},
            {"start": 0, "end": 2, "text": "審査", "label": "date"},
        ],
        "text_tagged": "審査受付日: [令和6年7月22日]date ...",
    }
]


def _write_pii(tmp_path: Path) -> Path:
    path = tmp_path / "data.parquet"
    PersonaRepository.write_sample(path, PII_ROWS)
    return path


def _cypher(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        statement
        for request in requests
        if request["path"] == "/db/neo4j/tx/commit"
        for statement in request["body"]["statements"]
    ]


def _index_pii(tmp_path: Path) -> None:
    cli.main(
        [
            "index",
            "--dataset-profile",
            "pii",
            "--dataset",
            str(_write_pii(tmp_path)),
            "--embedder",
            "hashed",
            "--vector-dimension",
            "8",
            "--persona-fields",
            "all",
        ]
    )


def test_pii_source_columns_are_detected() -> None:
    assert detect_profile(PII_ROWS[0].keys()) is PII


def test_pii_records_use_the_shared_record_keys(tmp_path: Path) -> None:
    (record,) = PersonaRepository([_write_pii(tmp_path)], profile=PII).iter_personas()

    assert record["uuid"] == PII_ROWS[0]["uid"]
    assert record["document_text"] == PII_ROWS[0]["text"]
    assert "text" not in record
    assert record["spans"] == PII_ROWS[0]["spans"]
    json.dumps(record, ensure_ascii=False)  # plain Python values only


def test_pii_index_writes_only_pii_stores(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_pii(tmp_path)

    for request in store_requests:
        if request["path"] != "/db/neo4j/tx/commit":
            assert request["path"].startswith(
                ("/collections/pii_documents", "/pii_documents")
            )
    for statement in _cypher(store_requests):
        assert not re.search(r"\b(Persona|Prefecture|Region)\b", statement["statement"])

    payload_indexes = [
        r["body"]["field_name"]
        for r in store_requests
        if r["path"].endswith("/index?wait=true")
    ]
    assert payload_indexes == ["domain", "document_type"]

    mapping = next(
        r
        for r in store_requests
        if r["method"] == "PUT" and r["path"] == "/pii_documents"
    )["body"]["mappings"]["properties"]
    assert mapping["domain"] == {"type": "keyword"}
    assert mapping["document_type"] == {"type": "keyword"}
    assert mapping["document_text"] == {"type": "text"}
    assert mapping["document_description"] == {"type": "text"}

    metadata = json.loads(
        cli.METADATA_PATH.with_name("index_metadata.pii.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["dataset"] == "pii"
    assert metadata["embedder"]["embedding_text_policy"] == "raw-text-v1"


def test_pii_documents_keep_text_and_link_domain_type_and_labels(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_pii(tmp_path)

    bulk = next(r for r in store_requests if r["path"].endswith("/_bulk"))
    document = json.loads(bulk["body"].splitlines()[1])
    row = PII_ROWS[0]
    assert document["document_text"] == row["text"]
    assert document["text"] == f"{row['text']}\n\n{row['document_description']}"
    assert document["domain"] == "金融"
    assert "spans" not in document

    statements = [s["statement"] for s in _cypher(store_requests)]
    assert any("CREATE CONSTRAINT pii_document_uuid" in s for s in statements)
    for name in ("pii_domain_name", "pii_document_type_name", "pii_label_name"):
        assert any(f"CREATE CONSTRAINT {name}" in s for s in statements), name
    merge = next(s for s in _cypher(store_requests) if "UNWIND" in s["statement"])
    for relationship in (":IN_DOMAIN", ":OF_TYPE", ":CONTAINS_PII"):
        assert relationship in merge["statement"]
    (parameters,) = merge["parameters"]["documents"]
    assert parameters == {
        "uuid": row["uid"],
        "text": row["document_description"],
        "domain": "金融",
        "document_type": "ローン審査書類",
        "pii_labels": ["account_number", "date", "last_name"],
    }


def test_pii_embedding_input_keeps_names() -> None:
    captured: list[str] = []

    class RecordingEmbedder(HashedNgramEmbedder):
        def embed_documents(self, texts):  # type: ignore[override]
            captured.extend(texts)
            return super().embed_documents(texts)

    class Sink:
        def __getattr__(self, name: str):
            return lambda *args, **kwargs: None

    record = {
        "uuid": "1",
        "document_text": "中島 様は、中島 様の口座を開設した。",
        "document_description": "中島 様は申込者だ。",
        "domain": "金融",
        "document_type": "申込書",
    }

    class OneRecord:
        def iter_personas(self, limit=None):
            yield record

    PersonaIndexer(
        repository=OneRecord(),  # type: ignore[arg-type]
        embedder=RecordingEmbedder(dimension=8, ngram_sizes=(2, 3)),
        qdrant=Sink(),  # type: ignore[arg-type]
        elasticsearch=Sink(),  # type: ignore[arg-type]
        neo4j=Sink(),  # type: ignore[arg-type]
        persona_fields=PII.text_fields,
        profile=PII,
    ).index()

    assert captured == [
        f"{record['document_text']}\n\n{record['document_description']}"
    ]


def test_pii_search_filters_by_domain_and_document_type(
    store_requests: list[dict[str, Any]], tmp_path: Path
) -> None:
    _index_pii(tmp_path)
    store_requests.clear()

    cli.main(
        [
            "search",
            "--dataset-profile",
            "pii",
            "--query",
            "口座番号を含む審査書類",
            "--domain",
            "金融",
            "--document-type",
            "ローン審査書類",
            "--format",
            "json",
        ]
    )

    qdrant, elasticsearch = (
        r for r in store_requests if r["path"] != "/db/neo4j/tx/commit"
    )
    assert qdrant["path"] == "/collections/pii_documents/points/search"
    assert qdrant["body"]["filter"]["must"] == [
        {"key": "domain", "match": {"value": "金融"}},
        {"key": "document_type", "match": {"value": "ローン審査書類"}},
    ]
    assert elasticsearch["body"]["query"]["bool"]["filter"] == [
        {"term": {"domain": "金融"}},
        {"term": {"document_type": "ローン審査書類"}},
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["--dataset-profile", "pii", "--prefecture", "東京都"],
        ["--domain", "金融"],
        ["--dataset-profile", "finance", "--document-type", "申請書"],
    ],
)
def test_filters_outside_the_profile_are_refused(
    store_requests: list[dict[str, Any]],
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["search", "--query", "q", "--embedder", "hashed", *argv])

    assert excinfo.value.code == 2
    assert "does not support the" in capsys.readouterr().out
    assert store_requests == []


def test_clearing_pii_deletes_only_pii_documents(
    store_requests: list[dict[str, Any]],
) -> None:
    cli.main(["clear-emulators", "--dataset-profile", "pii"])

    paths = [r["path"] for r in store_requests if r["path"] != "/db/neo4j/tx/commit"]
    assert paths == ["/collections/pii_documents", "/pii_documents"]
    (statement,) = _cypher(store_requests)
    assert statement["statement"].startswith("MATCH (p:PiiDocument)")
