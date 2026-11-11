from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_ja_persona import cli
from search_ja_persona.services import RequestDescriptor, SimpleHttpTransport


def _real_metadata_digests() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / ".cache").glob("index_metadata*.json"))
    }


@pytest.fixture(autouse=True, scope="session")
def _real_index_metadata_stays_untouched() -> Iterator[None]:
    # The live index's metadata lives in the developer's .cache/: a test
    # that reaches it (a reset unlinks it) silently breaks every later
    # search. Fail the session if any test changed it.
    before = _real_metadata_digests()
    yield
    assert _real_metadata_digests() == before, (
        "a test modified or deleted the real .cache/index_metadata*.json"
    )


@pytest.fixture(autouse=True)
def _isolated_metadata_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    metadata_dir = tmp_path_factory.mktemp("metadata")
    monkeypatch.setattr(cli, "METADATA_PATH", metadata_dir / "index_metadata.json")


def _respond(method: str, path: str, body: Any) -> dict[str, Any]:
    if path == "/db/neo4j/tx/commit":
        statement = body["statements"][0]["statement"]
        if "count(p)" in statement and "UNWIND" not in statement:
            return {"results": [{"data": [{"row": [0]}]}], "errors": []}
        return {"results": [{"data": []}], "errors": []}
    if method == "GET" and path.startswith("/collections/"):
        return {
            "result": {
                "points_count": 0,
                "config": {"params": {"vectors": {"size": 8}}},
                "payload_schema": {},
            }
        }
    if path.endswith("/points/scroll"):
        return {"result": {"points": []}}
    if path.endswith("/_stats/indexing"):
        indexing = {"index_total": 0, "delete_total": 0}
        return {"_all": {"primaries": {"indexing": indexing}}}
    if path.endswith("/_count"):
        return {"count": 0}
    if path.endswith("/_bulk"):
        return {"errors": False, "items": []}
    if path.endswith("/points/search"):
        return {"result": []}
    if path.endswith("/_search"):
        return {"hits": {"hits": []}}
    return {}


@pytest.fixture
def store_requests(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every emulator request instead of sending it; confirm resets."""

    recorded: list[dict[str, Any]] = []

    def record(self: SimpleHttpTransport, descriptor: RequestDescriptor) -> dict:
        recorded.append(
            {
                "method": descriptor.method,
                "path": descriptor.path,
                "body": descriptor.body,
            }
        )
        return _respond(descriptor.method, descriptor.path, descriptor.body)

    monkeypatch.setattr(SimpleHttpTransport, "request", record)
    monkeypatch.setattr(cli, "_confirm_reset", lambda: True)
    return recorded
