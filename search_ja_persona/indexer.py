from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .repository import PersonaRepository
from .services import ElasticsearchService, Neo4jService, QdrantService
from .vectorizer import HashedNgramVectorizer


@dataclass
class PersonaIndexer:
    repository: PersonaRepository
    vectorizer: HashedNgramVectorizer
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService

    def index(self, *, batch_size: int = 64, limit: int | None = None) -> None:
        self.qdrant.ensure_collection()
        self.elasticsearch.ensure_index()

        batch: list[dict] = []
        processed = 0
        for persona in self.repository.iter_personas(limit=limit):
            batch.append(persona)
            if len(batch) >= batch_size:
                self._process_batch(batch)
                processed += len(batch)
                batch = []
        if batch:
            self._process_batch(batch)
            processed += len(batch)

        if processed == 0:
            return

    def _process_batch(self, batch: list[dict]) -> None:
        qdrant_points = [self._build_qdrant_point(persona) for persona in batch]
        es_documents = [self._build_elasticsearch_document(persona) for persona in batch]

        self.qdrant.upsert_points(qdrant_points)
        self.elasticsearch.bulk_index(es_documents)
        for persona in batch:
            self.neo4j.merge_persona(persona)

    def _build_qdrant_point(self, persona: dict) -> dict:
        text = persona.get("persona", "")
        vector = self.vectorizer.embed(text)
        return {
            "id": persona["uuid"],
            "vector": vector,
            "payload": {
                "uuid": persona.get("uuid"),
                "text": text,
                "prefecture": persona.get("prefecture"),
                "region": persona.get("region"),
            },
        }

    def _build_elasticsearch_document(self, persona: dict) -> dict:
        return {
            "uuid": persona.get("uuid"),
            "text": persona.get("persona", ""),
            "prefecture": persona.get("prefecture"),
            "region": persona.get("region"),
        }
