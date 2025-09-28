from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from . import datasets
from .indexer import PersonaIndexer
from .repository import PersonaRepository
from .search import PersonaSearchService
from .services import ElasticsearchService, Neo4jService, QdrantService
from .vectorizer import HashedNgramVectorizer


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
    index_parser.add_argument("--vector-dimension", type=int, default=256)
    index_parser.add_argument("--ngram-sizes", default="2,3", help="Comma-separated n-gram sizes")
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
    search_parser.add_argument("--vector-dimension", type=int, default=256)
    search_parser.add_argument("--ngram-sizes", default="2,3")
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

    download_parser = subcommands.add_parser("download-dataset", help="Download dataset into the Hugging Face cache")
    download_parser.add_argument("--dataset-name", default=datasets.DEFAULT_DATASET_NAME)
    download_parser.add_argument("--split", default=datasets.DEFAULT_SPLIT)
    download_parser.add_argument("--cache-dir", type=Path, default=None)
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


def _build_vectorizer(dimension: int, ngram_sizes_raw: str) -> HashedNgramVectorizer:
    ngram_sizes = _parse_ngram_sizes(ngram_sizes_raw)
    if not ngram_sizes:
        raise ValueError("At least one n-gram size must be provided")
    return HashedNgramVectorizer(dimension=dimension, ngram_sizes=ngram_sizes)


def _run_index(args: argparse.Namespace) -> None:
    dataset_paths = _expand_paths(args.dataset)
    repository = PersonaRepository(dataset_paths)
    vectorizer = _build_vectorizer(args.vector_dimension, args.ngram_sizes)

    qdrant = QdrantService(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
        vector_size=vectorizer.dimension,
        distance=args.qdrant_distance,
    )
    elastic = ElasticsearchService(host=args.es_host, port=args.es_port, index=args.es_index)
    neo4j = Neo4jService(
        host=args.neo4j_host,
        port=args.neo4j_port,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    indexer = PersonaIndexer(
        repository=repository,
        vectorizer=vectorizer,
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
    )
    indexer.index(batch_size=args.batch_size, limit=args.limit)


def _run_search(args: argparse.Namespace) -> None:
    vectorizer = _build_vectorizer(args.vector_dimension, args.ngram_sizes)
    qdrant = QdrantService(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection=args.qdrant_collection,
        vector_size=vectorizer.dimension,
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
        vectorizer=vectorizer,
        qdrant=qdrant,
        elasticsearch=elastic,
        neo4j=neo4j,
    )
    results = service.search(args.query, limit=args.limit)
    print(json.dumps(results, ensure_ascii=False))


def _run_download_dataset(args: argparse.Namespace) -> None:
    config = datasets.DatasetCacheConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
        force_download=args.force,
        revision=args.revision,
        token=args.token,
    )
    datasets.ensure_dataset_cached(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
