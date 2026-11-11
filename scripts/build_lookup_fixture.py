"""Build the 40-entry exact-lookup fixture for the finance or pii dataset.

- finance: full person names, 5 per shard (8-shard stratified like the
  japan fixture), restricted to names that occur once in the corpus so
  every query has exactly one right answer.
- pii: PII span values, 5 per domain, restricted to values that occur in
  one document's text only.

Selection is seed-fixed, so a rebuild reproduces the committed fixture.

Usage:
    uv run --frozen python -m scripts.build_lookup_fixture --dataset-profile finance
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from scripts.evaluate_name_lookup import fixture_path
from search_ja_persona.dataset_profiles import FINANCE, PII, get_profile
from search_ja_persona.name_stripping import detect_person_name
from search_ja_persona.repository import PersonaRepository

# Fields whose openings carry the persona's name (two must agree).
NAME_FIELDS = ("persona", "professional_persona", "finance_persona")
# Span labels that name one thing (not dates, ages, or prefectures).
IDENTIFIER_LABELS = frozenset(
    {
        "account_number",
        "customer_id",
        "email",
        "employee_id",
        "medical_record_number",
        "phone_number",
        "street_address",
        "unique_id",
        "user_name",
    }
)
SEED = 20261111


def pick_unique_names(
    shards: Sequence[Sequence[dict[str, Any]]], *, per_shard: int, seed: int
) -> list[dict[str, str]]:
    named: list[list[tuple[str, str]]] = []
    counts: Counter[str] = Counter()
    for shard in shards:
        rows = []
        for record in shard:
            name = detect_person_name([record.get(f) or "" for f in NAME_FIELDS])
            if name and (" " in name or "　" in name):
                rows.append((name, record["uuid"]))
                counts[name] += 1
        named.append(rows)

    rng = random.Random(seed)
    picked: list[dict[str, str]] = []
    for rows in named:
        unique = sorted(row for row in rows if counts[row[0]] == 1)
        for name, uuid in rng.sample(unique, per_shard):
            picked.append({"name": name, "uuid": uuid})
    return picked


def pick_unique_values(
    documents: Iterable[dict[str, Any]], *, per_domain: int, seed: int
) -> list[dict[str, str]]:
    documents = list(documents)
    texts = {document["uuid"]: document["document_text"] for document in documents}
    annotated_in: Counter[str] = Counter()
    for document in documents:
        annotated_in.update({span["text"] for span in document.get("spans") or []})

    # Cheap filter first: identifier spans annotated in one document only.
    candidates: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for document in documents:
        for span in document.get("spans") or []:
            value = span["text"]
            if (
                span["label"] in IDENTIFIER_LABELS
                and len(value) >= 6
                and annotated_in[value] == 1
            ):
                candidates[document["domain"]].add(
                    (value, span["label"], document["uuid"])
                )

    # Then the exact check, only for the candidates drawn: the value must
    # not occur unannotated in any other document's text either.
    corpus = "\x00".join(texts.values())
    rng = random.Random(seed)
    picked: list[dict[str, str]] = []
    for domain in sorted(candidates):
        pool = sorted(candidates[domain])
        rng.shuffle(pool)
        accepted = 0
        for value, label, uuid in pool:
            if accepted == per_domain:
                break
            if corpus.count(value) != texts[uuid].count(value):
                continue
            picked.append(
                {"value": value, "label": label, "domain": domain, "uuid": uuid}
            )
            accepted += 1
        if accepted < per_domain:
            raise ValueError(f"only {accepted} unique values in domain {domain!r}")
    return picked


def _finance_shards() -> list[list[dict[str, Any]]]:
    # Only the name-bearing columns: the full records of 1M personas do not
    # need to sit in memory.
    paths = sorted(Path(FINANCE.source_dir).glob("train-*.parquet"))
    return [
        pq.read_table(path, columns=["uuid", *NAME_FIELDS]).to_pylist()
        for path in paths
    ]


def _pii_documents() -> list[dict[str, Any]]:
    path = Path(PII.source_dir) / "data.parquet"
    return list(PersonaRepository([path], profile=PII).iter_personas())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build an exact-lookup fixture")
    parser.add_argument(
        "--dataset-profile", choices=[FINANCE.name, PII.name], required=True
    )
    profile = get_profile(parser.parse_args(argv).dataset_profile)

    if profile.name == FINANCE.name:
        entries = pick_unique_names(_finance_shards(), per_shard=5, seed=SEED)
    else:
        entries = pick_unique_values(_pii_documents(), per_domain=5, seed=SEED)

    path = fixture_path(profile)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(entries)} entries to {path}")


if __name__ == "__main__":
    main()
