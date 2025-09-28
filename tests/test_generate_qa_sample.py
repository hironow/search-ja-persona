from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import scripts.generate_qa_sample as generator
from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS
from search_ja_persona.repository import PersonaRepository


def test_generate_qa_sample_accepts_limit_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    captured: dict[str, Any] = {}

    def fake_write_sample(path: Path, rows: list[dict]) -> None:
        captured["path"] = Path(path)
        captured["rows"] = rows

    monkeypatch.setattr(
        generator.PersonaRepository,
        "write_sample",
        staticmethod(fake_write_sample),
    )

    records = []
    for index in range(3):
        row: dict[str, Any] = {}
        for column in PersonaRepository.columns:
            row[column] = f"{column}-{index}"
        for field in PERSONA_TEXT_FIELDS:
            row[field] = f"{field}-{index}"
        records.append(row)

    def fake_load_dataset(name: str, **kwargs: Any):
        captured["split"] = kwargs.get("split")
        return records

    monkeypatch.setattr(generator, "load_dataset", fake_load_dataset)

    generator.run(["--limit", "7"])

    assert captured["split"] == "train[:7]"
    assert captured["path"] == Path("qa_samples/qa_sample.parquet")
    assert len(captured["rows"]) == len(records)
