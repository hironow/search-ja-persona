from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_ja_persona import cli


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
