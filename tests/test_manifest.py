from __future__ import annotations

from pathlib import Path

import pytest

from search_ja_persona.repository import PersonaRepository


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    PersonaRepository.write_sample(
        dataset_dir / "b.parquet",
        [
            {
                "uuid": "2",
                "persona": "関西で菓子職人として活躍するクリエイター",
                "prefecture": "大阪府",
                "region": "近畿地方",
            }
        ],
    )
    PersonaRepository.write_sample(
        dataset_dir / "a.parquet",
        [
            {
                "uuid": "1",
                "persona": "東京で介護に従事するリーダー",
                "prefecture": "東京都",
                "region": "関東地方",
            }
        ],
    )
    return dataset_dir


def test_build_manifest_orders_entries_and_counts(dataset: Path) -> None:
    from search_ja_persona import manifest

    entries = manifest.build_manifest([dataset])

    assert [item.path.name for item in entries] == ["a.parquet", "b.parquet"]
    assert [item.row_count for item in entries] == [1, 1]


def test_sample_manifest_respects_limit(dataset: Path) -> None:
    from search_ja_persona import manifest

    entries = manifest.build_manifest([dataset])

    sample = manifest.sample_manifest(entries, limit=1)

    assert len(sample) == 1
    assert sample[0].path.name == "a.parquet"


def test_sample_manifest_returns_all_when_limit_missing(dataset: Path) -> None:
    from search_ja_persona import manifest

    entries = manifest.build_manifest([dataset])

    sample = manifest.sample_manifest(entries, limit=None)

    assert sample == entries
