from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .indexer import PersonaIndexer
from .repository import PersonaRepository
from .search import PersonaSearchService
from .services import ElasticsearchService, Neo4jService, QdrantService
from .vectorizer import HashedNgramVectorizer


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
    vector_dimension: int = 256
    ngram_sizes: tuple[int, ...] = field(default_factory=lambda: (2, 3))


@dataclass
class PersonaApplication:
    """High-level orchestration helpers for indexing and search flows."""

    config: ApplicationConfig
    vectorizer: HashedNgramVectorizer
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService
    search_service: PersonaSearchService

    @classmethod
    def build(cls, config: ApplicationConfig) -> PersonaApplication:
        vectorizer = HashedNgramVectorizer(
            dimension=config.vector_dimension,
            ngram_sizes=config.ngram_sizes,
        )
        qdrant = QdrantService(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection=config.qdrant_collection,
            vector_size=vectorizer.dimension,
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
        search_service = PersonaSearchService(
            vectorizer=vectorizer,
            qdrant=qdrant,
            elasticsearch=elasticsearch,
            neo4j=neo4j,
        )
        return cls(
            config=config,
            vectorizer=vectorizer,
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
            vectorizer=self.vectorizer,
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
