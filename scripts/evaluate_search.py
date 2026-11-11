"""Search-quality benchmark against the live index.

Runs two objective measurements (see search_ja_persona/evaluation.py):
golden-query precision@k and self-retrieval recall@k, then prints a
summary and writes a JSON report under outputs/.

Usage:
    uv run --frozen python -m scripts.evaluate_search [--dataset-profile finance]
        [--k 5] [--samples 100]
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
from search_ja_persona.dataset_profiles import (
    JAPAN,
    PROFILES,
    DatasetProfile,
    get_profile,
)
from search_ja_persona.evaluation import (
    build_report,
    check_thresholds,
    load_golden_queries,
    precision_at_k,
    recall_at_k,
)
from search_ja_persona.services import RequestDescriptor

GOLDEN_PATH = Path(__file__).with_name("golden_queries.json")
# A pii document description is shared by ~10 documents, so it cannot
# identify its own document; value lookup (just eval-names) measures that.
SELF_RETRIEVAL_PROFILES = frozenset({"japan", "finance"})


def golden_path(profile: DatasetProfile) -> Path:
    if profile.name == JAPAN.name:
        return GOLDEN_PATH
    return GOLDEN_PATH.with_name(f"golden_queries.{profile.name}.json")


def self_retrieval_source(profile: DatasetProfile) -> Path | None:
    """The persona shard whose summaries query for their own persona."""

    if profile.name not in SELF_RETRIEVAL_PROFILES:
        return None
    return Path(profile.source_dir) / "train-00000-of-00008.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fused search quality")
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(PROFILES),
        default=JAPAN.name,
        help="Dataset whose index and golden set to evaluate",
    )
    parser.add_argument("--embedder", default="ruri-v3-310m")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for precision@k")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Personas sampled for self-retrieval recall",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--rrf-weights",
        default=None,
        help="Comma-separated vector,keyword RRF weights (default: production)",
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Exit non-zero when a ratified intent.md bar or a required metric is unmet",
    )
    return parser.parse_args(argv)


def run_golden(
    app: PersonaApplication, golden: list[dict], *, k: int
) -> tuple[list[dict], list[dict]]:
    """Score every query unfiltered; rerun filtered variants separately.

    Tier means stay comparable across baselines because they only ever see
    the unfiltered scores; entries carrying ``filters`` additionally run
    with the filter applied and land in the second list with their paired
    delta.
    """

    per_query: list[dict] = []
    filtered_rows: list[dict] = []
    for entry in golden:
        results = app.search(entry["query"], limit=k)
        score = precision_at_k(results, entry["expect"], k=k)
        per_query.append(
            {
                "query": entry["query"],
                "tier": entry["tier"],
                "precision_at_k": score,
                # Canary fields: dead graph context and a dead keyword leg
                # were both silent for a long time before they were found.
                "results_returned": len(results),
                "results_with_context": sum(
                    1
                    for hit in results
                    if (hit.get("context") or {}).get("relationships")
                ),
                "keyword_sourced": any(
                    "keyword" in (hit.get("sources") or []) for hit in results
                ),
            }
        )
        print(f"precision@{k} {score:.2f}  [{entry['tier']}]  {entry['query']}")

        filters = entry.get("filters")
        if filters is None:
            continue
        filtered_results = app.search(entry["query"], limit=k, filters=filters)
        filtered_score = precision_at_k(filtered_results, entry["expect"], k=k)
        filtered_rows.append(
            {
                "query": entry["query"],
                "tier": entry["tier"],
                "filters": filters,
                "unfiltered_precision_at_k": score,
                "filtered_precision_at_k": filtered_score,
                "delta": round(filtered_score - score, 4),
            }
        )
        print(
            f"precision@{k} {filtered_score:.2f}  [{entry['tier']}/filtered "
            f"{' '.join(f'{key}={value}' for key, value in filters.items())}]  "
            f"{entry['query']}"
        )
    return per_query, filtered_rows


def collect_store_counts(app: PersonaApplication) -> dict[str, int] | None:
    """Persona counts from all three stores (None when any is unreachable)."""

    try:
        qdrant = app.qdrant.transport.request(
            RequestDescriptor(
                method="GET", path=f"/collections/{app.qdrant.collection}"
            )
        )
        elasticsearch = app.elasticsearch.transport.request(
            RequestDescriptor(method="GET", path=f"/{app.elasticsearch.index}/_count")
        )
        neo4j = app.neo4j.count_nodes()
    except RuntimeError:
        return None
    return {
        "qdrant": int(qdrant.get("result", {}).get("points_count", 0)),
        "elasticsearch": int(elasticsearch.get("count", 0)),
        "neo4j": neo4j,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile = get_profile(args.dataset_profile)
    # Validate the golden set before touching any live service.
    golden = load_golden_queries(golden_path(profile), profile=profile)
    config_kwargs: dict = {"embedder": args.embedder, "dataset_profile": profile.name}
    if args.rrf_weights is not None:
        vector_weight, keyword_weight = (
            float(part) for part in args.rrf_weights.split(",")
        )
        config_kwargs["rrf_weights"] = (vector_weight, keyword_weight)
    config = ApplicationConfig(**config_kwargs)
    app = PersonaApplication.build(config)

    started = time.perf_counter()
    per_query, filtered_rows = run_golden(app, golden, k=args.k)

    recall_1 = recall_10 = 0
    sampled = 0
    source = self_retrieval_source(profile)
    if source is not None and source.exists():
        summary_field = profile.summary_field
        table = pq.read_table(source, columns=["uuid", summary_field])
        rng = random.Random(args.seed)
        rows = rng.sample(range(table.num_rows), args.samples)
        for row in rows:
            uuid = table.column("uuid")[row].as_py()
            summary = (table.column(summary_field)[row].as_py() or "").strip()
            if not uuid or not summary:
                continue
            ranked = [hit.get("uuid") for hit in app.search(summary, limit=10)]
            sampled += 1
            recall_1 += recall_at_k(ranked, uuid, k=1)
            recall_10 += recall_at_k(ranked, uuid, k=10)
    elif source is None:
        print(f"self-retrieval not applicable to the '{profile.name}' dataset")
    else:
        print(f"self-retrieval skipped: {source} not found")

    elapsed = time.perf_counter() - started
    report = build_report(
        per_query=per_query,
        filtered=filtered_rows,
        self_retrieval={
            "samples": sampled,
            "recall_at_1": round(recall_1 / sampled, 4) if sampled else None,
            "recall_at_10": round(recall_10 / sampled, 4) if sampled else None,
        },
        embedder=args.embedder,
        k=args.k,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        elapsed_seconds=elapsed,
        fusion={"rrf_weights": list(config.rrf_weights)},
        store_counts=collect_store_counts(app),
    )
    report = {"dataset": profile.name, **report}

    by_tier = report["golden_mean_precision_by_tier"]
    tier_summary = " | ".join(f"{tier} {value:.3f}" for tier, value in by_tier.items())
    filtered_summary = ""
    if report["filtered_mean_precision"] is not None:
        filtered_summary = (
            f"filtered geo {report['filtered_mean_precision']:.3f} "
            f"(n={len(report['filtered'])}) | "
        )
    print(
        f"\ngolden precision@{args.k}: {tier_summary} | "
        f"overall {report['golden_overall_mean_precision']:.3f} "
        f"(bar metric = basic) | "
        f"{filtered_summary}"
        f"self-retrieval recall@1: {report['self_retrieval']['recall_at_1']} "
        f"recall@10: {report['self_retrieval']['recall_at_10']} "
        f"(n={sampled}) | ctx {report['context_coverage']} | "
        f"kw-rate {report['keyword_contribution_rate']} | "
        f"stores {report['store_counts']} | {elapsed:.1f}s"
    )

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"search_eval-{profile.name}-{stamp}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")

    if args.check_thresholds:
        failures = check_thresholds(report, profile=profile)
        if failures:
            for failure in failures:
                print(f"THRESHOLD FAIL: {failure}")
            raise SystemExit(1)
        if profile.name == JAPAN.name:
            print(
                "thresholds OK (basic >= 0.85, hard >= 0.55, filtered >= 0.90, "
                "recall@1 >= 0.90, recall@10 >= 0.99)"
            )
        else:
            print(
                f"canaries OK; quality bars for '{profile.name}' are not "
                "ratified yet (report only)"
            )


if __name__ == "__main__":
    main()
