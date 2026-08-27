from __future__ import annotations

from search_ja_persona.name_stripping import detect_person_name, strip_person_names

FIELDS_SPACED = [
    "野本 花代子は、介護サービスの品質向上を推進する。",
    "野本 花代子は、シニア向け健康体操を主催し、地域で活動する。",
    "野本 花代子（72歳）は、季節の花を愛でる。",
]


def test_detect_spaced_name_with_cross_field_agreement() -> None:
    assert detect_person_name(FIELDS_SPACED) == "野本 花代子"


def test_detect_requires_agreement_across_fields() -> None:
    assert detect_person_name(["野本 花代子は、介護を担う。"]) is None


def test_strip_removes_name_and_leading_particle() -> None:
    stripped = strip_person_names(FIELDS_SPACED)

    assert stripped[0] == "介護サービスの品質向上を推進する。"
    assert stripped[2] == "季節の花を愛でる。"
    assert all("野本" not in text for text in stripped)


def test_detect_unspaced_name_needs_agreement() -> None:
    fields = [
        "西口甲一は温泉地で得た感覚を大切にする。",
        "西口甲一は計画書とチェックリストを用いる。",
    ]
    assert detect_person_name(fields) == "西口甲一"

    only_one = [
        "西口甲一は温泉地で得た感覚を大切にする。",
        "東京は大都市である。",
    ]
    assert detect_person_name(only_one) is None


def test_pronoun_subjects_are_not_names() -> None:
    fields = ["彼らは地域で活動する。", "彼らは計画を立てる。"]
    assert detect_person_name(fields) is None


def test_strip_keeps_place_names_matching_a_surname() -> None:
    fields = [
        "福岡 太也は、油圧配管の保守を担う。",
        "福岡 太也は、福岡の屋台文化を愛する。",
    ]

    stripped = strip_person_names(fields)

    assert all("福岡 太也" not in text for text in stripped)
    assert "福岡の屋台文化" in stripped[1]


def test_strip_removes_mid_text_mentions() -> None:
    fields = [
        "中村 明鈴は、機械修理の現場で働く。",
        "中村 明鈴は、指導も行う。地域は中村 明鈴の工房を頼る。",
    ]

    stripped = strip_person_names(fields)

    assert all("中村 明鈴" not in text for text in stripped)
    assert "工房を頼る" in stripped[1]


def test_no_name_text_is_unchanged() -> None:
    fields = ["料理が好きな人の記録。", "旅行の記録。"]
    assert strip_person_names(fields) == fields


def test_conflicting_candidates_are_a_noop() -> None:
    fields = [
        "田中 太郎は、営業を担う。",
        "佐藤 次郎は、経理を担う。",
        "田中 太郎は、地域で活動する。",
        "佐藤 次郎は、司会を務める。",
    ]
    assert detect_person_name(fields) is None
    assert strip_person_names(fields) == fields


def test_empty_fields_keep_their_positions() -> None:
    fields = ["", "田中 太郎は、営業を担う。", "田中 太郎は、地域で活動する。"]

    stripped = strip_person_names(fields)

    assert stripped[0] == ""
    assert stripped[1] == "営業を担う。"
