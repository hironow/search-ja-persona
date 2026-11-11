from __future__ import annotations

from typing import Any

from scripts.store_fingerprint import canonical_digest, collect_fingerprint
from search_ja_persona.application import ApplicationConfig, PersonaApplication


def test_canonical_digest_ignores_key_order_but_not_values() -> None:
    first = {"a": 1, "b": [{"x": "東京都", "y": 0.5}]}
    reordered = {"b": [{"y": 0.5, "x": "東京都"}], "a": 1}

    assert canonical_digest(first) == canonical_digest(reordered)
    assert canonical_digest(first) != canonical_digest(
        {"a": 1, "b": [{"x": "大阪府", "y": 0.5}]}
    )


def test_fingerprint_reads_only_the_profiles_stores(
    store_requests: list[dict[str, Any]],
) -> None:
    app = PersonaApplication.build(ApplicationConfig(dataset_profile="finance"))

    fingerprint = collect_fingerprint(app, sample=10)

    assert fingerprint["dataset"] == "finance"
    assert {r["method"] for r in store_requests} <= {"GET", "POST"}
    for request in store_requests:
        if request["path"] == "/db/neo4j/tx/commit":
            for statement in request["body"]["statements"]:
                text = statement["statement"]
                assert "FinancePersona" in text
                assert not any(
                    word in text for word in ("CREATE", "MERGE", "DELETE", "SET ")
                )
        else:
            assert request["path"].startswith(
                ("/collections/finance_personas", "/finance_personas")
            )
