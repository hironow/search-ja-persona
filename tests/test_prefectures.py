from __future__ import annotations

import pytest

from search_ja_persona.prefectures import PREFECTURES, validate_prefecture


def test_all_47_official_prefecture_names_are_known() -> None:
    assert len(PREFECTURES) == 47
    assert "北海道" in PREFECTURES
    assert "東京都" in PREFECTURES
    assert "京都府" in PREFECTURES
    assert "沖縄県" in PREFECTURES


def test_validate_prefecture_accepts_official_names_and_strips() -> None:
    assert validate_prefecture("北海道") == "北海道"
    assert validate_prefecture(" 沖縄県 ") == "沖縄県"


def test_validate_prefecture_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="沖縄"):
        validate_prefecture("沖縄")
    with pytest.raises(ValueError, match="prefecture"):
        validate_prefecture("")
