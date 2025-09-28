from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from pyarrow.lib import ArrowInvalid


@dataclass
class PersonaRepository:
    """Stream persona records from parquet shards."""

    parquet_paths: Iterable[Path]
    columns: tuple[str, ...] = (
        "uuid",
        "persona",
        "prefecture",
        "region",
        "occupation",
        "age",
        "sex",
    )

    def __post_init__(self) -> None:
        self._paths = [Path(path) for path in self.parquet_paths]

    def iter_personas(self, limit: int | None = None) -> Iterator[dict]:
        yielded = 0
        for path in self._paths:
            frame = self._read_frame(path)
            for record in frame.to_dict(orient="records"):
                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def _read_frame(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, columns=list(self.columns))
        except (ValueError, KeyError, ArrowInvalid):
            frame = pd.read_parquet(path)
            for column in self.columns:
                if column not in frame.columns:
                    frame[column] = None
            return frame[list(self.columns)]

    @staticmethod
    def write_sample(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
