"""Golden-set discriminative-power diagnostic.

Scores every golden query's predicate against three rankings:

- ``fused``:   the production vector-first fusion (``PersonaApplication.search``;
  with a fully indexed corpus Qdrant fills the top-k, so this is the
  vector-ranked list — keyword-only hits are appended after and truncated)
- ``keyword``: the Elasticsearch leg alone (BM25 top-k)
- ``random``:  fixed-seed random personas from shard 0, i.e. the predicate's
  base match rate over the corpus

A predicate that scores high on ``keyword`` or ``random`` is doing the work
itself (lexically non-independent with the query, or too broad) — the eval
then measures the predicate, not the retrieval. The report also records the
uuid overlap between the fused and keyword top-k, plus the fused top-k
snippets with per-hit predicate verdicts for human review.

Usage:
    uv run --frozen python -m scripts.diagnose_golden [--k 5] [--random-samples 200]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pyarrow.parquet as pq

from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.evaluation import (
    load_golden_queries,
    matches_expectation,
    precision_at_k,
)
from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS

GOLDEN_PATH = Path(__file__).with_name("golden_queries.json")
RANDOM_SHARD = Path(
    "datasets/Nemotron-Personas-Japan/data/train-00000-of-00008.parquet"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose golden-set predicates")
    parser.add_argument("--embedder", default="ruri-v3-310m")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--random-samples",
        type=int,
        default=200,
        help="Random personas sampled for the predicate base rate",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args(argv)


def _hex(value: object) -> str:
    try:
        return UUID(str(value)).hex
    except (ValueError, TypeError):
        return str(value)


def _keyword_results(
    app: PersonaApplication, query: str, k: int
) -> list[dict[str, Any]]:
    response = app.elasticsearch.search(query, limit=k)
    results: list[dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        results.append(
            {
                "uuid": hit.get("_id"),
                "text": source.get("text"),
                "prefecture": source.get("prefecture"),
                "region": source.get("region"),
            }
        )
    return results


def _random_pool(samples: int, seed: int) -> list[dict[str, Any]]:
    # Mirror index-time aggregation: text is the "\n\n" join of the six
    # persona fields (see PersonaIndexer._compose_text).
    columns = [*PERSONA_TEXT_FIELDS, "prefecture", "region"]
    table = pq.read_table(RANDOM_SHARD, columns=columns)
    rng = random.Random(seed)
    rows = rng.sample(range(table.num_rows), samples)
    pool: list[dict[str, Any]] = []
    for row in rows:
        texts = [
            value
            for field in PERSONA_TEXT_FIELDS
            if (value := table.column(field)[row].as_py())
        ]
        pool.append(
            {
                "text": "\n\n".join(texts),
                "prefecture": table.column("prefecture")[row].as_py(),
                "region": table.column("region")[row].as_py(),
            }
        )
    return pool


def _snippet(text: object) -> str:
    return " ".join(str(text or "").split())[:100]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    golden = load_golden_queries(GOLDEN_PATH)
    app = PersonaApplication.build(ApplicationConfig(embedder=args.embedder))
    pool = _random_pool(args.random_samples, args.seed) if RANDOM_SHARD.exists() else []
    if not pool:
        print(f"random baseline skipped: {RANDOM_SHARD} not found")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for entry in golden:
        query, expect, tier = entry["query"], entry["expect"], entry["tier"]
        fused = app.search(query, limit=args.k)
        keyword = _keyword_results(app, query, args.k)
        fused_rate = precision_at_k(fused, expect, k=args.k)
        keyword_rate = precision_at_k(keyword, expect, k=args.k)
        random_rate = precision_at_k(pool, expect, k=len(pool)) if pool else None
        filters = entry.get("filters")
        filtered_rate = None
        if filters is not None:
            filtered_hits = app.search(
                query, limit=args.k, prefecture=filters["prefecture"]
            )
            filtered_rate = precision_at_k(filtered_hits, expect, k=args.k)
        overlap = (
            len(
                {_hex(hit.get("uuid")) for hit in fused}
                & {_hex(hit.get("uuid")) for hit in keyword}
            )
            / args.k
        )
        rows.append(
            {
                "query": query,
                "tier": tier,
                "fused": fused_rate,
                "keyword": keyword_rate,
                "random": random_rate,
                "filtered": filtered_rate,
                "fused_keyword_overlap": overlap,
                "fused_hits": [
                    {
                        "uuid": _hex(hit.get("uuid")),
                        "prefecture": hit.get("prefecture"),
                        "matched": matches_expectation(hit, expect),
                        "snippet": _snippet(hit.get("text")),
                    }
                    for hit in fused
                ],
            }
        )
        random_label = f"{random_rate:.3f}" if random_rate is not None else "n/a"
        filtered_label = (
            f" | filt {filtered_rate:.2f}" if filtered_rate is not None else ""
        )
        print(
            f"[{tier}] fused {fused_rate:.2f} | kw {keyword_rate:.2f} | "
            f"rand {random_label} | overlap {overlap:.2f}{filtered_label}  {query}"
        )

    def _tier_mean(tier: str, key: str) -> float | None:
        values = [
            row[key] for row in rows if row["tier"] == tier and row[key] is not None
        ]
        return sum(values) / len(values) if values else None

    print()
    for tier in ("basic", "hard"):
        means = {key: _tier_mean(tier, key) for key in ("fused", "keyword", "random")}
        parts = " | ".join(
            f"{key} {value:.3f}" if value is not None else f"{key} n/a"
            for key, value in means.items()
        )
        print(f"{tier}: {parts}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": args.embedder,
        "k": args.k,
        "random_samples": len(pool),
        "seed": args.seed,
        "queries": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"golden_diagnose-{stamp}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
