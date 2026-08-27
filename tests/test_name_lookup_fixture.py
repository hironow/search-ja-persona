from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "scripts" / "name_lookup_queries.json"


def test_name_lookup_fixture_is_well_formed() -> None:
    entries = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert len(entries) == 40
    uuids = [entry["uuid"] for entry in entries]
    assert len(set(uuids)) == 40
    for entry in entries:
        assert entry["name"].strip()
        assert " " in entry["name"] or "　" in entry["name"]
        assert entry["uuid"].strip()
