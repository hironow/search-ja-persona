"""Characterization test: everything the japan (Nemotron-Personas-Japan) flow
sends to the stores, byte for byte.

The live 1M index was built by exactly these requests. Any change to the
Qdrant / Elasticsearch / Neo4j bodies, Cypher text, collection or index
names, payload shape, or recorded metadata would make the next index or
search run disagree with it, so the snapshot must only change on purpose.

Every HTTP call is captured at ``SimpleHttpTransport.request``, below all
service code, and the flows are driven through the public entry points
(CLI ``main``, ``PersonaApplication``, the eval store counter).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.evaluate_search import collect_store_counts
from search_ja_persona import cli
from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.repository import PersonaRepository
from search_ja_persona.services import RequestDescriptor, SimpleHttpTransport

SNAPSHOT_PATH = Path(__file__).with_name("snapshots") / "japan_wire_contract.json"

PERSONAS: list[dict[str, Any]] = [
    {
        "uuid": "0a1b2c3d4e5f40718293a4b5c6d7e8f9",
        "professional_persona": "野本 花代子は、介護施設の主任として新人教育を担う。",
        "sports_persona": "野本 花代子は、週末に近所の公園を散歩する。",
        "arts_persona": "野本 花代子は、水彩画教室に通っている。",
        "travel_persona": "野本 花代子は、温泉地を巡る旅が好きだ。",
        "culinary_persona": "野本 花代子は、出汁をとった和食を作る。",
        "persona": "野本 花代子は、東京都で介護に携わる穏やかなリーダーだ。",
        "prefecture": "東京都",
        "region": "関東地方",
        "occupation": "介護職員",
        "age": 52,
        "sex": "女",
    },
    {
        "uuid": "f9e8d7c6b5a440329182736455463728",
        "professional_persona": "浅野 零士は、機械設計のエンジニアとして CAD を使う。",
        "sports_persona": "浅野 零士は、休日に登山をする。",
        "arts_persona": "",
        "travel_persona": "浅野 零士は、北海道を一人旅したい。",
        "culinary_persona": "浅野 零士は、ラーメンの食べ歩きが趣味だ。",
        "persona": "浅野 零士は、大阪府で働く几帳面な設計者だ。",
        "prefecture": "大阪府",
        "region": "近畿地方",
        "occupation": "機械技術者",
        "age": 34,
        "sex": "男",
    },
]


def _neo4j_response(body: dict[str, Any]) -> dict[str, Any]:
    statement = body["statements"][0]["statement"]
    if "RETURN count(p)" in statement and "UNWIND" not in statement:
        return {"results": [{"data": [{"row": [0]}]}], "errors": []}
    if "DETACH DELETE" in statement:
        return {"results": [{"data": [{"row": [0]}]}], "errors": []}
    if "OPTIONAL MATCH" in statement:
        uuid = body["statements"][0]["parameters"]["uuid"]
        return {
            "results": [
                {
                    "data": [
                        {
                            "row": [
                                uuid,
                                "summary",
                                "東京都",
                                [{"type": "LIVES_IN", "target": "東京都"}],
                            ]
                        }
                    ]
                }
            ],
            "errors": [],
        }
    return {"results": [], "errors": []}


def _hyphenated(hex_uuid: str) -> str:
    return (
        f"{hex_uuid[:8]}-{hex_uuid[8:12]}-{hex_uuid[12:16]}-"
        f"{hex_uuid[16:20]}-{hex_uuid[20:]}"
    )


def _respond(method: str, path: str, body: Any) -> dict[str, Any]:
    if path == "/db/neo4j/tx/commit":
        return _neo4j_response(body)
    if method == "GET" and path.startswith("/collections/"):
        return {
            "result": {
                "points_count": 0,
                "config": {"params": {"vectors": {"size": 8}}},
                "payload_schema": {},
            }
        }
    if path.endswith("/points/search"):
        persona = PERSONAS[0]
        return {
            "result": [
                {
                    "id": _hyphenated(persona["uuid"]),
                    "score": 0.9,
                    "payload": {
                        "uuid": persona["uuid"],
                        "text": persona["persona"],
                        "persona_fields": {"persona": persona["persona"]},
                        "prefecture": persona["prefecture"],
                        "region": persona["region"],
                    },
                }
            ]
        }
    if path.endswith("/_search"):
        persona = PERSONAS[1]
        return {
            "hits": {
                "hits": [
                    {
                        "_id": persona["uuid"],
                        "_score": 3.5,
                        "_source": {
                            "uuid": persona["uuid"],
                            "text": persona["persona"],
                            "prefecture": persona["prefecture"],
                            "region": persona["region"],
                            "persona": persona["persona"],
                        },
                    }
                ]
            }
        }
    if path.endswith("/_count"):
        return {"count": 0}
    if path.endswith("/_bulk"):
        return {"errors": False, "items": []}
    return {}


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []

    def record(self: SimpleHttpTransport, descriptor: RequestDescriptor) -> dict:
        body = descriptor.body
        if isinstance(body, (dict, list)):
            body = json.loads(json.dumps(body))
        recorded.append(
            {
                "host": self.host,
                "port": self.port,
                "method": descriptor.method,
                "path": descriptor.path,
                "body": body,
                "headers": descriptor.headers,
            }
        )
        return _respond(descriptor.method, descriptor.path, body)

    monkeypatch.setattr(SimpleHttpTransport, "request", record)
    return recorded


def _drain(wire: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drained = list(wire)
    wire.clear()
    return drained


def _record_japan_flows(
    wire: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    dataset = tmp_path / "personas.parquet"
    PersonaRepository.write_sample(dataset, PERSONAS)
    snapshot: dict[str, Any] = {}

    cli.main(
        [
            "index",
            "--dataset",
            str(dataset),
            "--embedder",
            "hashed",
            "--vector-dimension",
            "8",
            "--persona-fields",
            "all",
        ]
    )
    snapshot["cli_index_requests"] = _drain(wire)
    metadata = json.loads(cli.METADATA_PATH.read_text(encoding="utf-8"))
    metadata.pop("timestamp")
    snapshot["cli_index_metadata"] = metadata

    cli.main(
        [
            "search",
            "--query",
            "介護の経験があるリーダー",
            "--limit",
            "2",
            "--prefecture",
            "東京都",
            "--format",
            "json",
        ]
    )
    snapshot["cli_search_requests"] = _drain(wire)

    app = PersonaApplication.build(ApplicationConfig(vector_dimension=8))
    app.index([dataset], batch_size=1)
    snapshot["app_index_requests"] = _drain(wire)
    snapshot["app_search_results"] = app.search("登山が趣味の設計者", limit=2)
    snapshot["app_search_requests"] = _drain(wire)

    cli.METADATA_PATH.unlink()
    cli.main(
        [
            "repair-metadata",
            "--embedder",
            "hashed",
            "--vector-dimension",
            "8",
        ]
    )
    snapshot["cli_repair_requests"] = _drain(wire)
    repaired = json.loads(cli.METADATA_PATH.read_text(encoding="utf-8"))
    repaired.pop("timestamp")
    snapshot["cli_repair_metadata"] = repaired

    snapshot["eval_store_counts"] = collect_store_counts(app)
    snapshot["eval_store_count_requests"] = _drain(wire)

    monkeypatch.setattr(cli, "_confirm_reset", lambda: True)
    cli.main(["clear-emulators"])
    snapshot["cli_clear_requests"] = _drain(wire)
    snapshot["cli_clear_removed_metadata"] = not cli.METADATA_PATH.exists()
    return json.loads(json.dumps(snapshot, ensure_ascii=False))


def test_japan_flows_send_exactly_the_recorded_requests(
    wire: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual = _record_japan_flows(wire, tmp_path, monkeypatch)

    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        pytest.fail(f"snapshot created at {SNAPSHOT_PATH}; review it and rerun")

    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert actual == expected


def test_live_japan_metadata_stays_usable_without_reset() -> None:
    # Verbatim copy of the live index's .cache/index_metadata.json
    # (recorded 2026-08-27 for the 1M ruri-v3-310m index). A schema bump or a
    # changed comparison would force a destructive reset prompt on the next
    # index run and make search fail closed.
    live = {
        "timestamp": "2026-08-27T12:40:52+00:00",
        "embedder": {
            "preset": "ruri-v3-310m",
            "type": "sentence",
            "model": "cl-nagoya/ruri-v3-310m",
            "dimension": 768,
            "normalize": True,
            "vector_dimension": 768,
            "ngram_sizes": None,
            "device": None,
            "fastembed_cache_dir": ".cache",
            "persona_fields": [
                "professional_persona",
                "sports_persona",
                "arts_persona",
                "travel_persona",
                "culinary_persona",
                "persona",
            ],
            "embedding_text_policy": "strip-person-names-v1",
        },
        "qdrant": {"collection": "personas", "distance": "Cosine"},
        "schema_version": "2025-03-05",
    }
    rerun_embedder = {
        key: live["embedder"][key]
        for key in (
            "type",
            "model",
            "vector_dimension",
            "ngram_sizes",
            "normalize",
            "persona_fields",
        )
    }

    assert cli._is_usable_metadata(live)
    assert not cli._should_reset(live, rerun_embedder)
