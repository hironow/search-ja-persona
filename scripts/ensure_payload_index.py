"""One-off migration: prefecture keyword payload index on an existing collection.

New collections get the index at index time (PersonaIndexer setup); this
script backfills collections created before the filter existed. Idempotent —
an existing keyword index is a noop, an incompatible one fails loudly.

Usage:
    uv run --frozen python -m scripts.ensure_payload_index \
        [--qdrant-host 127.0.0.1] [--qdrant-port 6333] \
        [--qdrant-collection personas]
"""

from __future__ import annotations

import argparse
import json

from search_ja_persona.services import QdrantService, RequestDescriptor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure the prefecture payload index exists"
    )
    parser.add_argument("--qdrant-host", default="127.0.0.1")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--qdrant-collection", default="personas")
    return parser.parse_args(argv)


def _print_schema(service: QdrantService, label: str) -> None:
    info = service.transport.request(
        RequestDescriptor(method="GET", path=f"/collections/{service.collection}")
    )
    schema = info.get("result", {}).get("payload_schema", {})
    print(f"{label} payload_schema: {json.dumps(schema, ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # vector_size is only used by ensure_collection, which this script never calls.
    service = QdrantService(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
        vector_size=0,
    )
    print(
        f"target: http://{args.qdrant_host}:{args.qdrant_port}"
        f"/collections/{args.qdrant_collection}"
    )
    _print_schema(service, "before")
    result = service.ensure_payload_index()
    print(f"ensure_payload_index: {result.get('status', result)}")
    _print_schema(service, "after")


if __name__ == "__main__":
    main()
