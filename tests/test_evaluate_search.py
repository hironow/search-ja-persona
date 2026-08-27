from __future__ import annotations

from typing import Any

from scripts.evaluate_search import run_golden


class FakeApp:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def search(
        self, query: str, *, limit: int, prefecture: str | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((query, prefecture))
        if prefecture is not None:
            return [{"text": "", "prefecture": prefecture, "region": ""}]
        return []


def test_run_golden_adds_filtered_run_for_filtered_entries() -> None:
    app = FakeApp()
    golden = [
        {
            "query": "海が好き",
            "tier": "hard",
            "expect": {"prefecture": "沖縄県"},
            "filters": {"prefecture": "沖縄県"},
        },
        {"query": "介護", "tier": "basic", "expect": {"text_any": ["介護"]}},
    ]

    per_query, filtered = run_golden(app, golden, k=1)

    assert [row["precision_at_k"] for row in per_query] == [0.0, 0.0]
    assert filtered == [
        {
            "query": "海が好き",
            "tier": "hard",
            "filters": {"prefecture": "沖縄県"},
            "unfiltered_precision_at_k": 0.0,
            "filtered_precision_at_k": 1.0,
            "delta": 1.0,
        }
    ]
    assert ("海が好き", None) in app.calls
    assert ("海が好き", "沖縄県") in app.calls
    assert ("介護", None) in app.calls


def test_run_golden_without_filters_returns_empty_filtered_list() -> None:
    app = FakeApp()
    golden = [{"query": "介護", "tier": "basic", "expect": {"text_any": ["介護"]}}]

    per_query, filtered = run_golden(app, golden, k=1)

    assert len(per_query) == 1
    assert filtered == []


def test_run_golden_rows_carry_canary_fields() -> None:
    class CanaryApp:
        def search(
            self, query: str, *, limit: int, prefecture: str | None = None
        ) -> list[dict]:
            return [
                {
                    "uuid": "1",
                    "text": "介護の記録",
                    "prefecture": "",
                    "region": "",
                    "sources": ["vector", "keyword"],
                    "context": {"relationships": [{"type": "LIVES_IN"}]},
                },
                {
                    "uuid": "2",
                    "text": "別の記録",
                    "prefecture": "",
                    "region": "",
                    "sources": ["vector"],
                    "context": {"relationships": []},
                },
            ]

    golden = [{"query": "介護", "tier": "basic", "expect": {"text_any": ["介護"]}}]

    per_query, _ = run_golden(CanaryApp(), golden, k=2)

    row = per_query[0]
    assert row["results_returned"] == 2
    assert row["results_with_context"] == 1
    assert row["keyword_sourced"] is True
