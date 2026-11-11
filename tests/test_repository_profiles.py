from __future__ import annotations

from pathlib import Path

import pytest

from search_ja_persona.dataset_profiles import FINANCE, JAPAN, detect_profile
from search_ja_persona.repository import PersonaRepository


def _write(path: Path, row: dict) -> Path:
    PersonaRepository.write_sample(path, [row])
    return path


def _japan_row() -> dict:
    return {
        "uuid": "0a1b2c3d4e5f40718293a4b5c6d7e8f9",
        "persona": "野本 花代子は、東京都で働く。",
        "prefecture": "東京都",
        "region": "関東地方",
    }


def _finance_row() -> dict:
    return {**_japan_row(), "finance_persona": "野本 花代子は、堅実に貯蓄する。"}


def test_detect_profile_tells_finance_from_japan() -> None:
    assert detect_profile(_finance_row().keys()) is FINANCE
    assert detect_profile(_japan_row().keys()) is JAPAN
    assert detect_profile(["uid", "text"]) is None


def test_japan_profile_refuses_finance_shards(tmp_path: Path) -> None:
    path = _write(tmp_path / "train-00000-of-00008.parquet", _finance_row())

    with pytest.raises(ValueError, match="'finance' dataset, not 'japan'"):
        list(PersonaRepository([path], profile=JAPAN).iter_personas())


def test_finance_profile_refuses_japan_shards(tmp_path: Path) -> None:
    path = _write(tmp_path / "train-00000-of-00008.parquet", _japan_row())

    with pytest.raises(ValueError, match="'japan' dataset, not 'finance'"):
        list(PersonaRepository([path], profile=FINANCE).iter_personas())


def test_parquet_without_an_id_column_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "data.parquet", {"uid": "x", "text": "本文"})

    with pytest.raises(ValueError, match="no known dataset"):
        list(PersonaRepository([path], profile=JAPAN).iter_personas())


def test_partial_japan_sample_still_reads(tmp_path: Path) -> None:
    path = _write(tmp_path / "sample.parquet", _japan_row())

    (record,) = PersonaRepository([path], profile=JAPAN).iter_personas()

    assert record["uuid"] == _japan_row()["uuid"]
    assert record["sports_persona"] is None
