from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.table import Table

from . import datasets
from .embeddings import (
    EMBEDDER_PRESETS,
    FastEmbedder,
    HashedNgramEmbedder,
    SentenceTransformerEmbedder,
)
from .indexer import PersonaIndexer
from .repository import PersonaRepository
from .search import PersonaSearchService
from .services import (
    ElasticsearchService,
    Neo4jService,
    QdrantService,
    RequestDescriptor,
)
from .persona_fields import PERSONA_TEXT_FIELDS

console = Console()
METADATA_PATH = Path(".cache/index_metadata.json")
PERSONA_FIELD_SET = set(PERSONA_TEXT_FIELDS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        _run_index(args)
    elif args.command == "search":
        _run_search(args)
    elif args.command == "download-dataset":
        _run_download_dataset(args)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command: {args.command}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="search-ja-persona")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.required = True

    index_parser = subcommands.add_parser("index", help="Index personas into emulator services")
    index_parser.add_argument("--dataset", action="append", required=True, help="Parquet file or directory path")
    index_parser.add_argument("--batch-size", type=int, default=64)
    index_parser.add_argument("--limit", type=int, default=None)
    index_parser.add_argument(
        "--embedder",
        choices=sorted(EMBEDDER_PRESETS.keys()),
        default="hashed",
        help="Embedding preset (hashed, mini-lm, mpnet, e5-small, e5-large, fast-e5-small, fast-e5-large)",
    )
    index_parser.add_argument("--vector-dimension", type=int, default=256)
    index_parser.add_argument("--ngram-sizes", default="2,3", help="Comma-separated n-gram sizes")
    index_parser.add_argument(
        "--persona-fields",
        default="persona",
        help="Comma-separated persona columns or 'all' (defaults to 'persona')",
    )
    index_parser.add_argument("--embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    index_parser.add_argument("--embedder-device", default=None)
    index_parser.add_argument(
        "--embedder-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable normalization for sentence-transformer embeddings",
    )
    index_parser.add_argument("--fastembed-cache-dir", type=Path, default=Path(".cache"))
    index_parser.add_argument("--qdrant-host", default="localhost")
    index_parser.add_argument("--qdrant-port", type=int, default=6333)
    index_parser.add_argument("--qdrant-collection", default="personas")
    index_parser.add_argument("--qdrant-distance", default="Cosine")
    index_parser.add_argument("--es-host", default="localhost")
    index_parser.add_argument("--es-port", type=int, default=9200)
    index_parser.add_argument("--es-index", default="personas")
    index_parser.add_argument("--neo4j-host", default="localhost")
    index_parser.add_argument("--neo4j-port", type=int, default=7474)
    index_parser.add_argument("--neo4j-user", default="neo4j")
    index_parser.add_argument("--neo4j-password", default="password")

    search_parser = subcommands.add_parser("search", help="Run a free-text persona search")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument(
        "--embedder",
        choices=sorted(EMBEDDER_PRESETS.keys()),
        default=None,
        help="Embedding preset (leave unset to reuse last indexed preset automatically)",
    )
    search_parser.add_argument("--vector-dimension", type=int, default=256)
    search_parser.add_argument("--ngram-sizes", default="2,3")
    search_parser.add_argument(
        "--persona-fields",
        default=None,
        help="Comma-separated persona columns, 'all', or omit to reuse last indexed setting",
    )
    search_parser.add_argument("--embedder-model", default="sentence-transformers/all-MiniLM-L6-v2")
    search_parser.add_argument("--embedder-device", default=None)
    search_parser.add_argument(
        "--embedder-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable normalization for sentence-transformer embeddings",
    )
    search_parser.add_argument("--fastembed-cache-dir", type=Path, default=Path(".cache"))
    search_parser.add_argument("--qdrant-host", default="localhost")
    search_parser.add_argument("--qdrant-port", type=int, default=6333)
    search_parser.add_argument("--qdrant-collection", default="personas")
    search_parser.add_argument("--qdrant-distance", default="Cosine")
    search_parser.add_argument("--es-host", default="localhost")
    search_parser.add_argument("--es-port", type=int, default=9200)
    search_parser.add_argument("--es-index", default="personas")
    search_parser.add_argument("--neo4j-host", default="localhost")
    search_parser.add_argument("--neo4j-port", type=int, default=7474)
    search_parser.add_argument("--neo4j-user", default="neo4j")
    search_parser.add_argument("--neo4j-password", default="password")
    search_parser.add_argument("--format", choices=("table", "json"), default="table")
    search_parser.add_argument("--verbose", action="store_true")

    download_parser = subcommands.add_parser("download-dataset", help="Download dataset into the Hugging Face cache")
    download_parser.add_argument("--dataset-name", default=datasets.DEFAULT_DATASET_NAME)
    download_parser.add_argument("--split", default=datasets.DEFAULT_SPLIT)
    download_parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    download_parser.add_argument("--revision", default=None)
    download_parser.add_argument("--token", default=None)
    download_parser.add_argument("--force", action="store_true")

    return parser


def _parse_ngram_sizes(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def _expand_paths(entries: Iterable[str]) -> list[Path]:
    expanded: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.parquet")))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("No parquet files found for provided dataset paths")
    return expanded


def _load_index_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    try:
        return json.loads(METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return None


def _parse_persona_fields(value: str) -> tuple[str, ...]:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("Persona fields must not be empty")
    if cleaned.lower() == "all":
        return PERSONA_TEXT_FIELDS
    fields = tuple(part.strip() for part in cleaned.split(",") if part.strip())
    if not fields:
        raise ValueError("Persona fields must not be empty")
    for field in fields:
        if field not in PERSONA_FIELD_SET:
            raise ValueError(f"Unknown persona field: {field}")
    return fields


def _resolve_persona_fields(
    args: argparse.Namespace,
    *,
    existing_metadata: dict | None = None,
    default: tuple[str, ...] = ("persona",),
    allow_metadata_default: bool = False,
) -> tuple[tuple[str, ...], str | None]:
    note: str | None = None
    if getattr(args, "persona_fields", None) not in (None, ""):
        fields = _parse_persona_fields(args.persona_fields)
    else:
        metadata_fields = (
            (existing_metadata or {}).get("embedder", {}).get("persona_fields")
            if existing_metadata
            else None
        )
        if metadata_fields:
            fields = tuple(metadata_fields)
            if allow_metadata_default:
                note = (
                    f"Using persona fields {', '.join(fields)} recorded in {METADATA_PATH}"
                )
        else:
            fields = default
    return fields, note


def _write_index_metadata(embedder_info: dict, args: argparse.Namespace, embedder) -> None:
    ngram_sizes = embedder_info.get("ngram_sizes")
    if ngram_sizes is not None and not isinstance(ngram_sizes, list):
        ngram_sizes = list(ngram_sizes)
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": {
            "preset": embedder_info.get("preset"),
            "type": embedder_info.get("type"),
            "model": embedder_info.get("model"),
            "dimension": embedder.dimension,
            "normalize": embedder_info.get("normalize"),
            "vector_dimension": embedder_info.get("vector_dimension"),
            "ngram_sizes": ngram_sizes,
            "device": embedder_info.get("device"),
            "fastembed_cache_dir": embedder_info.get("fastembed_cache_dir"),
            "persona_fields": embedder_info.get("persona_fields"),
        },
        "qdrant": {
            "collection": args.qdrant_collection,
            "distance": args.qdrant_distance,
        },
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _collect_index_stats(
    qdrant: QdrantService,
    elastic: ElasticsearchService,
    neo4j: Neo4jService,
) -> tuple[dict[str, int], dict[str, int]]:
    stats: dict[str, int] = {}
    meta: dict[str, int] = {}

    try:
        response = qdrant.transport.request(
            RequestDescriptor(
                method="GET",
                path=f"/collections/{qdrant.collection}",
            )
        )
        stats["qdrant_points"] = int(
            response.get("result", {}).get("points_count", 0)
        )
        size = (
            response.get("result", {})
            .get("config", {})
            .get("params", {})
            .get("vectors", {})
            .get("size")
        )
        if size is not None:
            meta["qdrant_vector_size"] = int(size)
    except Exception:
        stats["qdrant_points"] = -1

    try:
        response = elastic.transport.request(
            RequestDescriptor(
                method="GET",
                path=f"/{elastic.index}/_count",
            )
        )
        stats["elasticsearch_docs"] = int(response.get("count", 0))
    except Exception:
        stats["elasticsearch_docs"] = -1

    try:
        response = neo4j.transport.request(
            RequestDescriptor(
                method="POST",
                path="/db/neo4j/tx/commit",
                body={
                    "statements": [
                        {
                            "statement": "MATCH (p:Persona) RETURN count(p)",
                        }
                    ]
                },
            )
        )
        count = (
            response.get("results", [{}])[0]
            .get("data", [{}])[0]
            .get("row", [0])[0]
        )
        stats["neo4j_nodes"] = int(count)
    except Exception:
        stats["neo4j_nodes"] = -1

    return stats, meta


def _should_reset(existing: dict | None, new_embedder: dict) -> bool:
    if not existing:
        return False
    current = existing.get("embedder") or {}
    for key in (
        "type",
        "model",
        "vector_dimension",
        "ngram_sizes",
        "normalize",
        "persona_fields",
    ):
        if key in current or key in new_embedder:
            if current.get(key) != new_embedder.get(key):
                return True
    return False


def _confirm_reset() -> bool:
    response = console.input(
        "Existing index metadata differs. Reset Qdrant/Elasticsearch/Neo4j before indexing? [y/N] (type 'yes' or 'no'): "
    )
    return response.strip().lower() in {"y", "yes"}


def _reset_indexes(qdrant: QdrantService, elastic: ElasticsearchService, neo4j: Neo4jService, args: argparse.Namespace) -> None:
    console.log("Resetting existing indexes...")
    cleared: list[str] = []
    try:
        qdrant.transport.request(
            RequestDescriptor("DELETE", f"/collections/{args.qdrant_collection}")
        )
        cleared.append(f"Qdrant collection '{args.qdrant_collection}'")
    except Exception:
        pass

    try:
        elastic.transport.request(RequestDescriptor("DELETE", f"/{args.es_index}"))
        cleared.append(f"Elasticsearch index '{args.es_index}'")
    except Exception:
        pass

    try:
        neo4j.transport.request(
            RequestDescriptor(
                method="POST",
                path="/db/neo4j/tx/commit",
                body={
                    "statements": [
                        {
                            "statement": "MATCH (p:Persona) DETACH DELETE p",
                        }
                    ]
                },
            )
        )
        cleared.append("Neo4j Persona nodes")
    except Exception:
        pass

    try:
        METADATA_PATH.unlink()
    except FileNotFoundError:
        pass

    if cleared:
        console.print("[yellow]Cleared " + ", ".join(cleared) + "[/yellow]")
    else:
        console.print("[yellow]No existing resources were removed[/yellow]")


def _build_embedder(
    args: argparse.Namespace,
) -> tuple[
    HashedNgramEmbedder | SentenceTransformerEmbedder,
    dict,
    str | None,
]:
    metadata_note: str | None = None
    preset_name = args.embedder
    metadata = None
    if preset_name is None:
        metadata = _load_index_metadata()
        embed_info = (metadata or {}).get("embedder") if metadata else None
        if embed_info:
            preset_name = embed_info.get("preset")
            embedder_type = embed_info.get("type", preset_name)
            model_name = embed_info.get("model", args.embedder_model)
            args.embedder_normalize = embed_info.get("normalize", args.embedder_normalize)
            if embedder_type == "sentence" and embed_info.get("device") is not None:
                args.embedder_device = embed_info.get("device")
            if embedder_type == "fast" and embed_info.get("fastembed_cache_dir"):
                args.fastembed_cache_dir = Path(embed_info["fastembed_cache_dir"])
            if embedder_type == "hashed":
                args.vector_dimension = embed_info.get("vector_dimension", args.vector_dimension)
                if embed_info.get("ngram_sizes"):
                    args.ngram_sizes = ",".join(str(n) for n in embed_info["ngram_sizes"])
            metadata_note = f"Using embedder preset '{preset_name or embedder_type}' recorded in {METADATA_PATH}"
        else:
            preset_name = "hashed"

    if not preset_name:
        preset_name = embedder_type if 'embedder_type' in locals() else 'hashed'
    preset = EMBEDDER_PRESETS.get(preset_name, {})
    embedder_type = preset.get("type", preset_name)
    model_name = preset.get("model", args.embedder_model)

    info: dict = {
        "preset": preset_name,
        "type": embedder_type,
        "model": model_name,
        "normalize": args.embedder_normalize,
        "device": args.embedder_device,
        "fastembed_cache_dir": str(args.fastembed_cache_dir) if args.fastembed_cache_dir else None,
    }

    if embedder_type == "sentence":
        embedder = SentenceTransformerEmbedder(
            model_name=model_name,
            device=args.embedder_device,
            normalize_embeddings=args.embedder_normalize,
        )
    elif embedder_type == "fast":
        embedder = FastEmbedder(
            model_name=model_name,
            cache_dir=str(args.fastembed_cache_dir) if args.fastembed_cache_dir else None,
            normalize_embeddings=args.embedder_normalize,
        )
    else:
        ngram_sizes = _parse_ngram_sizes(args.ngram_sizes)
        if not ngram_sizes:
            raise ValueError("At least one n-gram size must be provided")
        embedder = HashedNgramEmbedder(dimension=args.vector_dimension, ngram_sizes=ngram_sizes)
        info["vector_dimension"] = args.vector_dimension
        info["ngram_sizes"] = list(ngram_sizes)

    if embedder_type in {"sentence", "fast"}:
        info["vector_dimension"] = embedder.dimension

    args.embedder = preset_name
    return embedder, info, metadata_note


def _run_index(args: argparse.Namespace) -> None:
    dataset_paths = _expand_paths(args.dataset)
    console.log(f"Resolved {len(dataset_paths)} dataset path(s)")

    existing_metadata = _load_index_metadata()
    persona_fields, persona_note = _resolve_persona_fields(
        args, existing_metadata=existing_metadata, default=("persona",)
    )
    setattr(args, "_persona_fields_resolved", persona_fields)

    repository = PersonaRepository(dataset_paths)
    embedder, embedder_info, _ = _build_embedder(args)
    embedder_info["persona_fields"] = list(persona_fields)

    qdrant = QdrantService(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
        vector_size=embedder.dimension,
        distance=args.qdrant_distance,
    )
    elastic = ElasticsearchService(host=args.es_host, port=args.es_port, index=args.es_index)
    neo4j = Neo4jService(
        host=args.neo4j_host,
        port=args.neo4j_port,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    before_stats, before_meta = _collect_index_stats(qdrant, elastic, neo4j)
    if not existing_metadata and before_meta.get("qdrant_vector_size"):
        existing_metadata = {
            "embedder": {
                "vector_dimension": before_meta["qdrant_vector_size"],
                "persona_fields": list(persona_fields),
            }
        }
    if _should_reset(existing_metadata, embedder_info):
        current = (existing_metadata or {}).get("embedder", {}) if existing_metadata else {}
        console.print(
            f"[yellow]Existing index uses '{current.get('preset', current.get('type', 'unknown'))}'. "
            f"New preset '{embedder_info.get('preset')}' selected.[/yellow]"
        )
        if not _confirm_reset():
            console.print("[red]Indexing aborted by user.[/red]")
            return
        _reset_indexes(qdrant, elastic, neo4j, args)
        before_stats, _ = _collect_index_stats(qdrant, elastic, neo4j)

    indexer = PersonaIndexer(
        repository=repository,
        embedder=embedder,
        persona_fields=persona_fields,
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
    )

    console.log(
        f"Indexing personas (batch_size={args.batch_size}, limit={args.limit or '∞'})"
    )
    indexer.index(batch_size=args.batch_size, limit=args.limit)
    _write_index_metadata(embedder_info, args, embedder)
    after_stats, _ = _collect_index_stats(qdrant, elastic, neo4j)

    def _format_count(value: int) -> str:
        if value < 0:
            return "unknown"
        return str(value)

    console.print("[bold green]Indexing completed[/bold green]")
    console.print(
        f"[dim]Qdrant points: {_format_count(before_stats['qdrant_points'])} -> {_format_count(after_stats['qdrant_points'])} | "
        f"Elasticsearch docs: {_format_count(before_stats['elasticsearch_docs'])} -> {_format_count(after_stats['elasticsearch_docs'])} | "
        f"Neo4j persona nodes: {_format_count(before_stats['neo4j_nodes'])} -> {_format_count(after_stats['neo4j_nodes'])}[/dim]"
    )


def _run_search(args: argparse.Namespace) -> None:
    existing_metadata = _load_index_metadata()
    persona_fields, persona_note = _resolve_persona_fields(
        args,
        existing_metadata=existing_metadata,
        default=("persona",),
        allow_metadata_default=True,
    )
    setattr(args, "_persona_fields_resolved", persona_fields)
    embedder, embedder_info, metadata_note = _build_embedder(args)
    embedder_info["persona_fields"] = list(persona_fields)
    qdrant = QdrantService(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
        vector_size=embedder.dimension,
        distance=args.qdrant_distance,
    )
    elastic = ElasticsearchService(host=args.es_host, port=args.es_port, index=args.es_index)
    neo4j = Neo4jService(
        host=args.neo4j_host,
        port=args.neo4j_port,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    service = PersonaSearchService(
        embedder=embedder,
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
        persona_fields=persona_fields,
    )
    notes = []
    if metadata_note and args.embedder is None:
        notes.append(metadata_note)
    if persona_note and args.persona_fields is None:
        notes.append(persona_note)
    if notes:
        console.print("[dim]" + " | ".join(notes) + "[/dim]")
    if args.verbose:
        results, stats = service.search(args.query, limit=args.limit, return_stats=True)
        console.print(
            f"[dim]Qdrant candidates: {stats['vector_hits']} | Elasticsearch hits: {stats['keyword_hits']} | Context lookups: {stats['context_calls']}[/dim]"
        )
    else:
        results = service.search(args.query, limit=args.limit)
        stats = None
    if args.format == "json":
        console.print_json(json.dumps(results, ensure_ascii=False))
        return

    if not results:
        console.print("[bold yellow]No results found[/bold yellow]")
        return

    table = Table(title="Search Results", show_lines=True)
    table.add_column("UUID", overflow="fold", max_width=40)
    table.add_column("Score", justify="right")
    table.add_column("Prefecture")
    table.add_column("Region")

    include_combined = len(persona_fields) > 1
    if include_combined:
        table.add_column("Combined", overflow="fold", max_width=80)

    for field in persona_fields:
        label = "Persona" if field == "persona" else field.replace("_", " ").title()
        table.add_column(label, overflow="fold", max_width=80)

    for entry in results:
        per_field = entry.get("persona_fields") or {}
        row = [
            str(entry.get("uuid", "")),
            f"{entry.get('score', 0.0):.4f}",
            entry.get("prefecture") or "",
            entry.get("region") or "",
        ]
        combined_text = (entry.get("text") or "").strip()
        if include_combined:
            row.append(combined_text)
        for field in persona_fields:
            row.append((per_field.get(field) or "").strip())
        table.add_row(*row)

    console.print(table)

    if args.verbose and stats is not None:
        console.print(f"[dim]Returned {stats['results']} combined result(s)[/dim]")


def _run_download_dataset(args: argparse.Namespace) -> None:
    config = datasets.DatasetCacheConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
        force_download=args.force,
        revision=args.revision,
        token=args.token,
    )
    console.log("Downloading dataset shards...")
    datasets.ensure_dataset_cached(config)
    console.log("Dataset download completed")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
