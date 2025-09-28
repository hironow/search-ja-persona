from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .repository import PersonaRepository
from .services import ElasticsearchService, Neo4jService, QdrantService
from .embeddings import Embedder
from .persona_fields import PERSONA_TEXT_FIELDS


@dataclass
class PersonaIndexer:
    repository: PersonaRepository
    embedder: Embedder
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService
    persona_fields: tuple[str, ...] = PERSONA_TEXT_FIELDS

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
        es_documents = [
            self._build_elasticsearch_document(persona) for persona in batch
        ]

        self.qdrant.upsert_points(qdrant_points)
        self.elasticsearch.bulk_index(es_documents)
        for persona in batch:
            self.neo4j.merge_persona(persona)

    def _build_qdrant_point(self, persona: dict) -> dict:
        aggregated_text, per_field = self._compose_text(persona)
        vector = self.embedder.embed(aggregated_text)
        return {
            "id": persona["uuid"],
            "vector": vector,
            "payload": {
                "uuid": persona.get("uuid"),
                "text": aggregated_text,
                "persona_fields": per_field,
                "prefecture": persona.get("prefecture"),
                "region": persona.get("region"),
            },
        }

    def _build_elasticsearch_document(self, persona: dict) -> dict:
        aggregated_text, per_field = self._compose_text(persona)
        document: dict[str, Any] = {
            "uuid": persona.get("uuid"),
            "text": aggregated_text,
            "prefecture": persona.get("prefecture"),
            "region": persona.get("region"),
        }
        for field in self.persona_fields:
            document[field] = per_field.get(field)
        return document

    def _compose_text(self, persona: dict) -> tuple[str, dict[str, str]]:
        per_field: dict[str, str] = {}
        texts: list[str] = []
        for field in self.persona_fields:
            value = (persona.get(field) or "").strip()
            per_field[field] = value
            if value:
                texts.append(value)
        aggregated = "\n\n".join(texts)
        return aggregated, per_field
