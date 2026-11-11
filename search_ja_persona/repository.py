from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyarrow.lib import ArrowInvalid

from .dataset_profiles import JAPAN, DatasetProfile, detect_profile


@dataclass
class PersonaRepository:
    """Stream persona records from parquet shards."""

    parquet_paths: Iterable[Path]
    profile: DatasetProfile = JAPAN

    @property
    def columns(self) -> tuple[str, ...]:
        return self.profile.columns

    def __post_init__(self) -> None:
        self._paths = [Path(path) for path in self.parquet_paths]

    def iter_personas(self, limit: int | None = None) -> Iterator[dict]:
        yielded = 0
        for path in self._paths:
            self._require_profile_match(path)
            frame = self._read_frame(path)
            for record in frame.to_dict(orient="records"):
                # List columns (PII spans) arrive as numpy arrays.
                yield {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in record.items()
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    def _require_profile_match(self, path: Path) -> None:
        # Shards of different datasets share file names and most columns;
        # reading one under another profile would index it silently into
        # the wrong stores, so tell them apart by their source columns.
        detected = detect_profile(pq.read_schema(path).names)
        if detected is None:
            raise ValueError(
                f"{path} matches no known dataset (missing "
                f"{sorted(self.profile.marker_columns)} columns)"
            )
        if detected.name != self.profile.name:
            raise ValueError(
                f"{path} looks like the '{detected.name}' dataset, not "
                f"'{self.profile.name}': pass --dataset-profile {detected.name}"
            )

    def _read_frame(self, path: Path) -> pd.DataFrame:
        renames = dict(self.profile.source_renames)
        to_source = {record_key: source for source, record_key in renames.items()}
        source_columns = [to_source.get(column, column) for column in self.columns]
        try:
            frame = pd.read_parquet(path, columns=source_columns)
        except (ValueError, KeyError, ArrowInvalid):
            frame = pd.read_parquet(path)
            for column in source_columns:
                if column not in frame.columns:
                    frame[column] = None
            frame = frame[source_columns]
        return frame.rename(columns=renames)

    @staticmethod
    def write_sample(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
