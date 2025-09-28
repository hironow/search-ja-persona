from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    row_count: int | None


def build_manifest(entries: Iterable[str | Path]) -> list[ManifestEntry]:
    """Return deterministic list of parquet shards with row counts when available."""

    collected: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            children = sorted(
                child for child in path.glob("*.parquet") if child.is_file()
            )
            for child in children:
                if child not in seen:
                    collected.append(child)
                    seen.add(child)
        elif path.suffix == ".parquet" and path.is_file():
            if path not in seen:
                collected.append(path)
                seen.add(path)

    collected.sort(key=lambda value: value.name)

    manifest: list[ManifestEntry] = []
    for path in collected:
        manifest.append(ManifestEntry(path=path, row_count=_count_rows(path)))
    return manifest


def sample_manifest(
    entries: Sequence[ManifestEntry], limit: int | None
) -> list[ManifestEntry]:
    if limit is None or limit >= len(entries):
        return list(entries)
    return list(entries[:limit])


def _count_rows(path: Path) -> int | None:
    try:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        if metadata is None:
            return None
        return metadata.num_rows
    except Exception:  # pragma: no cover - defensive, e.g., corrupted file
        return None
