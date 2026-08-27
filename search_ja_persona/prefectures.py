"""Official prefecture names and input validation.

The ``prefecture`` field is stored as an exact keyword in every backend, so
a colloquial form ("沖縄") would silently match nothing. Validate at the
entry points (CLI, golden set) instead of the mechanical service layer.
"""

from __future__ import annotations

PREFECTURES: frozenset[str] = frozenset(
    {
        "北海道",
        "青森県",
        "岩手県",
        "宮城県",
        "秋田県",
        "山形県",
        "福島県",
        "茨城県",
        "栃木県",
        "群馬県",
        "埼玉県",
        "千葉県",
        "東京都",
        "神奈川県",
        "新潟県",
        "富山県",
        "石川県",
        "福井県",
        "山梨県",
        "長野県",
        "岐阜県",
        "静岡県",
        "愛知県",
        "三重県",
        "滋賀県",
        "京都府",
        "大阪府",
        "兵庫県",
        "奈良県",
        "和歌山県",
        "鳥取県",
        "島根県",
        "岡山県",
        "広島県",
        "山口県",
        "徳島県",
        "香川県",
        "愛媛県",
        "高知県",
        "福岡県",
        "佐賀県",
        "長崎県",
        "熊本県",
        "大分県",
        "宮崎県",
        "鹿児島県",
        "沖縄県",
    }
)


def validate_prefecture(value: str) -> str:
    """Return the stripped official name, or raise on anything else."""

    stripped = value.strip()
    if stripped not in PREFECTURES:
        raise ValueError(
            f"unknown prefecture {value!r}: use the official name "
            "(e.g. 沖縄県, not 沖縄)"
        )
    return stripped
