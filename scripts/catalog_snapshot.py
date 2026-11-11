"""Record fused-search results from a full index for the marimo catalog.

The catalog renders the committed snapshot even with the emulators down.
Run it after a full index of the dataset; it refuses when the three store
counts disagree (a partial index would make a misleading snapshot).

Usage:
    uv run --frozen python -m scripts.catalog_snapshot --dataset-profile finance
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.evaluate_search import collect_store_counts
from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.dataset_profiles import (
    JAPAN,
    PROFILES,
    DatasetProfile,
    get_profile,
)

MARIMO_DIR = Path(__file__).resolve().parents[1] / "marimo"
EXCERPT_CHARS = 200
RESULTS_PER_QUERY = 3

SNAPSHOT_QUERIES: dict[str, tuple[str, ...]] = {
    "japan": (
        "高齢者介護の経験豊富なマネージャー",
        "病院で働く看護師",
        "大阪に住んでいてたこ焼きなど粉もの料理が好きな人",
        "伝統的な和食を大切にする料理人",
        "登山やハイキングが趣味の人",
    ),
    "finance": (
        "住宅ローンの返済に不安を抱えている共働きの会社員",
        "退職金をどう運用するか悩んでいる定年前の人",
        "子どもの教育資金を計画的に貯めている親",
        "相続や資産の引き継ぎについて考えている高齢者",
        "株式の配当を受け取っている投資家",
    ),
    "pii": (
        "病院で医師が患者の病名を証明する書類",
        "銀行で新しく口座を作るための申込み",
        "引っ越しで住所が変わったときに役所へ出す届出",
        "不正アクセスが起きたときの社内報告",
        "従業員に毎月の給料の内訳を知らせる書類",
    ),
}


def snapshot_path(profile: DatasetProfile) -> Path:
    if profile.name == JAPAN.name:
        return MARIMO_DIR / "catalog_snapshot.json"
    return MARIMO_DIR / f"catalog_snapshot.{profile.name}.json"


def build_snapshot(
    profile: DatasetProfile,
    runs: list[tuple[str, float, list[dict[str, Any]]]],
    *,
    embedder: str,
    dimension: int,
    corpus_size: int,
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the snapshot JSON from (query, latency_ms, hits) runs."""

    return {
        "dataset": profile.name,
        "generated_at": generated_at,
        "embedder": embedder,
        "dimension": dimension,
        "corpus_size": corpus_size,
        "environment": "Windows 11 / RTX 4090 / local emulator stack",
        "embedding_text_policy": profile.embedding_text_policy,
        "queries": [
            {
                "query": query,
                "latency_ms": round(latency_ms),
                "results": [
                    {
                        "score": round(float(hit.get("score", 0.0)), 4),
                        **{field: hit.get(field) for field in profile.payload_fields},
                        "excerpt": " ".join(str(hit.get("text") or "").split())[
                            :EXCERPT_CHARS
                        ],
                    }
                    for hit in hits
                ],
            }
            for query, latency_ms, hits in runs
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record the catalog snapshot")
    parser.add_argument("--dataset-profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--embedder", default="ruri-v3-310m")
    args = parser.parse_args(argv)
    profile = get_profile(args.dataset_profile)

    app = PersonaApplication.build(
        ApplicationConfig(embedder=args.embedder, dataset_profile=profile.name)
    )
    counts = collect_store_counts(app)
    if not counts or len(set(counts.values())) != 1 or not counts["qdrant"]:
        raise SystemExit(f"store counts disagree or are empty: {counts}")

    app.search("warm-up", limit=1)  # load the model outside the timings
    runs = []
    for query in SNAPSHOT_QUERIES[profile.name]:
        started = time.perf_counter()
        hits = app.search(query, limit=RESULTS_PER_QUERY)
        runs.append((query, (time.perf_counter() - started) * 1000, hits))

    snapshot = build_snapshot(
        profile,
        runs,
        embedder=args.embedder,
        dimension=app.embedder.dimension,
        corpus_size=counts["qdrant"],
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    path = snapshot_path(profile)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} ({len(runs)} queries, corpus {counts['qdrant']:,})")


if __name__ == "__main__":
    main()
