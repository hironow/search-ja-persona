"""Read-only fingerprint of one dataset's stores, to prove they did not change.

Records counts and write statistics plus digests of a fixed sample (the
first ``--sample`` records by id) from Qdrant (payload and vector),
Elasticsearch (source), and Neo4j (node properties and outgoing
relationships), and the digest of the dataset's metadata file. Run it
before and after working on another dataset and compare the two JSON
files; they must be equal.

The Elasticsearch ``index_total`` counts writes since the node started,
so compare fingerprints taken without a restart in between.

Usage:
    uv run --frozen python -m scripts.store_fingerprint --dataset-profile japan
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from search_ja_persona import cli
from search_ja_persona.application import ApplicationConfig, PersonaApplication
from search_ja_persona.dataset_profiles import JAPAN, PROFILES
from search_ja_persona.services import RequestDescriptor


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _qdrant(app: PersonaApplication, sample: int) -> dict[str, Any]:
    collection = app.qdrant.collection
    info = app.qdrant.transport.request(
        RequestDescriptor("GET", f"/collections/{collection}")
    )["result"]
    scroll = app.qdrant.transport.request(
        RequestDescriptor(
            "POST",
            f"/collections/{collection}/points/scroll",
            body={"limit": sample, "with_payload": True, "with_vector": True},
        )
    )["result"]["points"]
    return {
        "points_count": info.get("points_count"),
        "vectors": info.get("config", {}).get("params", {}).get("vectors"),
        "payload_schema": sorted(info.get("payload_schema", {})),
        "sample_size": len(scroll),
        "sample_digest": canonical_digest(scroll),
    }


def _elasticsearch(app: PersonaApplication, sample: int) -> dict[str, Any]:
    index = app.elasticsearch.index
    transport = app.elasticsearch.transport
    count = transport.request(RequestDescriptor("GET", f"/{index}/_count"))["count"]
    stats = transport.request(RequestDescriptor("GET", f"/{index}/_stats/indexing"))
    hits = transport.request(
        RequestDescriptor(
            "GET",
            f"/{index}/_search",
            body={
                "size": sample,
                "sort": [{"uuid": "asc"}],
                "query": {"match_all": {}},
            },
        )
    )["hits"]["hits"]
    return {
        "count": count,
        "index_total": stats["_all"]["primaries"]["indexing"]["index_total"],
        "delete_total": stats["_all"]["primaries"]["indexing"]["delete_total"],
        "sample_size": len(hits),
        "sample_digest": canonical_digest(
            [{"_id": hit["_id"], "_source": hit["_source"]} for hit in hits]
        ),
    }


def _neo4j_rows(app: PersonaApplication, statement: str, **parameters: Any) -> list:
    response = app.neo4j.transport.request(
        RequestDescriptor(
            "POST",
            "/db/neo4j/tx/commit",
            body={"statements": [{"statement": statement, "parameters": parameters}]},
        )
    )
    return [row["row"] for row in response["results"][0]["data"]]


def _neo4j(app: PersonaApplication, sample: int) -> dict[str, Any]:
    label = app.neo4j.label
    relationships = _neo4j_rows(
        app,
        f"MATCH (p:{label})-[r]->() RETURN type(r), count(r) ORDER BY type(r)",
    )
    nodes = _neo4j_rows(
        app,
        f"MATCH (p:{label}) WITH p ORDER BY p.uuid LIMIT $sample "
        "OPTIONAL MATCH (p)-[r]->(t) "
        "WITH p, collect([type(r), coalesce(t.name, t.uuid)]) AS links "
        "RETURN properties(p), links ORDER BY p.uuid",
        sample=sample,
    )
    for _properties, links in nodes:
        links.sort()
    return {
        "nodes": app.neo4j.count_nodes(),
        "relationships": {name: count for name, count in relationships},
        "sample_size": len(nodes),
        "sample_digest": canonical_digest(nodes),
    }


def collect_fingerprint(app: PersonaApplication, *, sample: int) -> dict[str, Any]:
    return {
        "dataset": app.profile.name,
        "qdrant": _qdrant(app, sample),
        "elasticsearch": _elasticsearch(app, sample),
        "neo4j": _neo4j(app, sample),
    }


def _metadata_digest(profile_name: str) -> str | None:
    profile = PROFILES[profile_name]
    path = (
        cli.METADATA_PATH
        if profile.name == JAPAN.name
        else cli.METADATA_PATH.with_name(profile.metadata_filename)
    )
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fingerprint one dataset's stores")
    parser.add_argument(
        "--dataset-profile", choices=sorted(PROFILES), default=JAPAN.name
    )
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args(argv)

    app = PersonaApplication.build(
        ApplicationConfig(dataset_profile=args.dataset_profile)
    )
    fingerprint = collect_fingerprint(app, sample=args.sample)
    fingerprint["metadata_sha256"] = _metadata_digest(args.dataset_profile)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"store_fingerprint-{args.dataset_profile}-{stamp}.json"
    out_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(fingerprint, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
