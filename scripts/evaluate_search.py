"""Search-quality benchmark against the live index.

Runs two objective measurements (see search_ja_persona/evaluation.py):
golden-query precision@k and self-retrieval recall@k, then prints a
summary and writes a JSON report under outputs/.

Usage:
    uv run --frozen python -m scripts.evaluate_search [--k 5] [--samples 100]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.evaluation import (
    build_report,
    load_golden_queries,
    precision_at_k,
    recall_at_k,
)

GOLDEN_PATH = Path(__file__).with_name("golden_queries.json")
SELF_RETRIEVAL_SHARD = Path(
    "datasets/Nemotron-Personas-Japan/data/train-00000-of-00008.parquet"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fused search quality")
    parser.add_argument("--embedder", default="ruri-v3-310m")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for precision@k")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Personas sampled for self-retrieval recall",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Validate the golden set before touching any live service.
    golden = load_golden_queries(GOLDEN_PATH)
    app = PersonaApplication.build(ApplicationConfig(embedder=args.embedder))

    per_query: list[dict] = []
    started = time.perf_counter()
    for entry in golden:
        results = app.search(entry["query"], limit=args.k)
        score = precision_at_k(results, entry["expect"], k=args.k)
        per_query.append(
            {"query": entry["query"], "tier": entry["tier"], "precision_at_k": score}
        )
        print(f"precision@{args.k} {score:.2f}  [{entry['tier']}]  {entry['query']}")

    recall_1 = recall_10 = 0
    sampled = 0
    if SELF_RETRIEVAL_SHARD.exists():
        table = pq.read_table(SELF_RETRIEVAL_SHARD, columns=["uuid", "persona"])
        rng = random.Random(args.seed)
        rows = rng.sample(range(table.num_rows), args.samples)
        for row in rows:
            uuid = table.column("uuid")[row].as_py()
            summary = (table.column("persona")[row].as_py() or "").strip()
            if not uuid or not summary:
                continue
            ranked = [hit.get("uuid") for hit in app.search(summary, limit=10)]
            sampled += 1
            recall_1 += recall_at_k(ranked, uuid, k=1)
            recall_10 += recall_at_k(ranked, uuid, k=10)
    else:
        print(f"self-retrieval skipped: {SELF_RETRIEVAL_SHARD} not found")

    elapsed = time.perf_counter() - started
    report = build_report(
        per_query=per_query,
        self_retrieval={
            "samples": sampled,
            "recall_at_1": round(recall_1 / sampled, 4) if sampled else None,
            "recall_at_10": round(recall_10 / sampled, 4) if sampled else None,
        },
        embedder=args.embedder,
        k=args.k,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        elapsed_seconds=elapsed,
    )

    by_tier = report["golden_mean_precision_by_tier"]
    tier_summary = " | ".join(f"{tier} {value:.3f}" for tier, value in by_tier.items())
    print(
        f"\ngolden precision@{args.k}: {tier_summary} | "
        f"overall {report['golden_overall_mean_precision']:.3f} "
        f"(bar metric = basic) | "
        f"self-retrieval recall@1: {report['self_retrieval']['recall_at_1']} "
        f"recall@10: {report['self_retrieval']['recall_at_10']} "
        f"(n={sampled}) | {elapsed:.1f}s"
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"search_eval-{stamp}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
