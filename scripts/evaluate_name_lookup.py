"""Lookup benchmark: exact-string queries that identify one record.

With person names stripped from embeddings, name retrieval is owed by the
keyword leg (plus the keyword-preference tie-break). This measures that
guarantee on a fixed 40-entry fixture per dataset: full names for japan
and finance (8-shard stratified), PII values that occur in one document
only for pii (5 per domain).

Usage:
    uv run --frozen python -m scripts.evaluate_name_lookup [--dataset-profile pii]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.dataset_profiles import (
    JAPAN,
    PII,
    PROFILES,
    DatasetProfile,
    get_profile,
)
from search_ja_persona.evaluation import recall_at_k

FIXTURE_PATH = Path(__file__).with_name("name_lookup_queries.json")


def fixture_path(profile: DatasetProfile) -> Path:
    if profile.name == JAPAN.name:
        return FIXTURE_PATH
    if profile.name == PII.name:
        return FIXTURE_PATH.with_name("value_lookup_queries.pii.json")
    return FIXTURE_PATH.with_name(f"name_lookup_queries.{profile.name}.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-string lookup benchmark")
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(PROFILES),
        default=JAPAN.name,
        help="Dataset whose index and golden set to evaluate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    profile = get_profile(parse_args(argv).dataset_profile)
    entries = json.loads(fixture_path(profile).read_text(encoding="utf-8"))
    app = PersonaApplication.build(
        ApplicationConfig(embedder="ruri-v3-310m", dataset_profile=profile.name)
    )

    recall_1 = recall_10 = 0
    started = time.perf_counter()
    for entry in entries:
        query = entry["name"] if "name" in entry else entry["value"]
        ranked = [hit.get("uuid") for hit in app.search(query, limit=10)]
        hit_1 = recall_at_k(ranked, entry["uuid"], k=1)
        hit_10 = recall_at_k(ranked, entry["uuid"], k=10)
        recall_1 += hit_1
        recall_10 += hit_10
        if not hit_1:
            print(f"miss@1 {'(miss@10!)' if not hit_10 else ''} {query}")
    elapsed = time.perf_counter() - started

    report = {
        "dataset": profile.name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "samples": len(entries),
        "recall_at_1": round(recall_1 / len(entries), 4),
        "recall_at_10": round(recall_10 / len(entries), 4),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        f"\n{profile.name} lookup recall@1: {report['recall_at_1']} "
        f"recall@10: {report['recall_at_10']} (n={len(entries)}) | {elapsed:.1f}s"
    )
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"name_lookup-{profile.name}-{stamp}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
