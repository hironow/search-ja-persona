from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .embeddings import (
    EMBEDDER_PRESETS,
    FastEmbedder,
    HashedNgramEmbedder,
    SentenceTransformerEmbedder,
)
from .indexer import PersonaIndexer
from .repository import PersonaRepository
from .search import PersonaSearchService
from .services import ElasticsearchService, Neo4jService, QdrantService
from .persona_fields import PERSONA_TEXT_FIELDS


@dataclass(frozen=True)
class ApplicationConfig:
    """Configuration covering vectorizer and emulator endpoints."""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "personas"
    qdrant_distance: str = "Cosine"
    es_host: str = "localhost"
    es_port: int = 9200
    es_index: str = "personas"
    neo4j_host: str = "localhost"
    neo4j_port: int = 7474
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    embedder: str = "hashed"
    vector_dimension: int = 256
    ngram_sizes: tuple[int, ...] = field(default_factory=lambda: (2, 3))
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sentence_device: str | None = None
    normalize_embeddings: bool = True
    fastembed_cache_dir: str | None = ".cache"
    persona_fields: tuple[str, ...] = ("persona",)


@dataclass
class PersonaApplication:
    """High-level orchestration helpers for indexing and search flows."""

    config: ApplicationConfig
    embedder: HashedNgramEmbedder | SentenceTransformerEmbedder | FastEmbedder
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService
    search_service: PersonaSearchService

    @classmethod
    def build(cls, config: ApplicationConfig) -> PersonaApplication:
        embedder: HashedNgramEmbedder | SentenceTransformerEmbedder | FastEmbedder
        preset = EMBEDDER_PRESETS.get(config.embedder)
        if preset:
            embedder_type = preset["type"]
            model_name = preset.get("model", config.sentence_model)
        else:
            embedder_type = config.embedder
            model_name = config.sentence_model

        if embedder_type == "sentence":
            embedder = SentenceTransformerEmbedder(
                model_name=model_name,
                device=config.sentence_device,
                normalize_embeddings=config.normalize_embeddings,
            )
        elif embedder_type == "fast":
            embedder = FastEmbedder(
                model_name=model_name,
                cache_dir=config.fastembed_cache_dir,
                normalize_embeddings=config.normalize_embeddings,
            )
        else:
            embedder = HashedNgramEmbedder(
                dimension=config.vector_dimension,
                ngram_sizes=config.ngram_sizes,
            )

        qdrant = QdrantService(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection=config.qdrant_collection,
            vector_size=embedder.dimension,
            distance=config.qdrant_distance,
        )
        elasticsearch = ElasticsearchService(
            host=config.es_host,
            port=config.es_port,
            index=config.es_index,
        )
        neo4j = Neo4jService(
            host=config.neo4j_host,
            port=config.neo4j_port,
            user=config.neo4j_user,
            password=config.neo4j_password,
        )
        persona_fields = config.persona_fields or PERSONA_TEXT_FIELDS
        search_service = PersonaSearchService(
            embedder=embedder,
            qdrant=qdrant,
            elasticsearch=elasticsearch,
            neo4j=neo4j,
            persona_fields=persona_fields,
        )
        return cls(
            config=config,
            embedder=embedder,
            qdrant=qdrant,
            elasticsearch=elasticsearch,
            neo4j=neo4j,
            search_service=search_service,
        )

    def index(
        self,
        dataset_entries: Sequence[str | Path],
        *,
        batch_size: int = 64,
        limit: int | None = None,
    ) -> None:
        paths = self._expand_dataset_paths(dataset_entries)
        repository = PersonaRepository(paths)
        indexer = PersonaIndexer(
            repository=repository,
            embedder=self.embedder,
            persona_fields=self.config.persona_fields or PERSONA_TEXT_FIELDS,
            qdrant=self.qdrant,
            elasticsearch=self.elasticsearch,
            neo4j=self.neo4j,
        )
        indexer.index(batch_size=batch_size, limit=limit)

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self.search_service.search(query, limit=limit)

    @staticmethod
    def _expand_dataset_paths(entries: Iterable[str | Path]) -> list[Path]:
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
