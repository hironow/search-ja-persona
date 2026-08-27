"""Name-lookup benchmark: exact full-name queries against the live index.

With person names stripped from embeddings, name retrieval is owed by the
keyword leg (plus the keyword-preference tie-break). This measures that
guarantee on a fixed 40-name stratified fixture.

Usage:
    uv run --frozen python -m scripts.evaluate_name_lookup
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.evaluation import recall_at_k

FIXTURE_PATH = Path(__file__).with_name("name_lookup_queries.json")


def main() -> None:
    entries = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    app = PersonaApplication.build(ApplicationConfig(embedder="ruri-v3-310m"))

    recall_1 = recall_10 = 0
    started = time.perf_counter()
    for entry in entries:
        ranked = [hit.get("uuid") for hit in app.search(entry["name"], limit=10)]
        hit_1 = recall_at_k(ranked, entry["uuid"], k=1)
        hit_10 = recall_at_k(ranked, entry["uuid"], k=10)
        recall_1 += hit_1
        recall_10 += hit_10
        if not hit_1:
            print(f"miss@1 {'(miss@10!)' if not hit_10 else ''} {entry['name']}")
    elapsed = time.perf_counter() - started

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "samples": len(entries),
        "recall_at_1": round(recall_1 / len(entries), 4),
        "recall_at_10": round(recall_10 / len(entries), 4),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        f"\nname lookup recall@1: {report['recall_at_1']} "
        f"recall@10: {report['recall_at_10']} (n={len(entries)}) | {elapsed:.1f}s"
    )
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"name_lookup-{stamp}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
