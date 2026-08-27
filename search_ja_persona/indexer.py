from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embeddings import Embedder
from .name_stripping import strip_person_names
from .persona_fields import PERSONA_TEXT_FIELDS
from .repository import PersonaRepository
from .services import ElasticsearchService, Neo4jService, QdrantService


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
        self.qdrant.ensure_payload_index()
        self.elasticsearch.ensure_index()
        self.neo4j.ensure_constraints()

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
        composed = [self._compose_text(persona) for persona in batch]
        # One model call per batch: single-item embed calls would leave
        # GPU/ONNX backends dominated by per-call overhead. The embedding
        # input has person names stripped (vectors chase names otherwise);
        # the stored text keeps them for BM25 lookup and display.
        vectors = self.embedder.embed_documents(
            [
                "\n\n".join(text for text in strip_person_names(field_texts) if text)
                for _, _, field_texts in composed
            ]
        )
        qdrant_points = [
            self._build_qdrant_point(persona, aggregated, per_field, vector)
            for persona, (aggregated, per_field, _), vector in zip(
                batch, composed, vectors, strict=True
            )
        ]
        es_documents = [
            self._build_elasticsearch_document(persona, aggregated, per_field)
            for persona, (aggregated, per_field, _) in zip(batch, composed, strict=True)
        ]

        self.qdrant.upsert_points(qdrant_points)
        self.elasticsearch.bulk_index(es_documents)
        self.neo4j.merge_personas(batch)

    def _build_qdrant_point(
        self,
        persona: dict,
        aggregated_text: str,
        per_field: dict[str, str],
        vector: list[float],
    ) -> dict:
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

    def _build_elasticsearch_document(
        self, persona: dict, aggregated_text: str, per_field: dict[str, str]
    ) -> dict:
        document: dict[str, Any] = {
            "uuid": persona.get("uuid"),
            "text": aggregated_text,
            "prefecture": persona.get("prefecture"),
            "region": persona.get("region"),
        }
        for field in self.persona_fields:
            document[field] = per_field.get(field)
        return document

    def _compose_text(self, persona: dict) -> tuple[str, dict[str, str], list[str]]:
        per_field: dict[str, str] = {}
        texts: list[str] = []
        for field in self.persona_fields:
            value = (persona.get(field) or "").strip()
            per_field[field] = value
            if value:
                texts.append(value)
        aggregated = "\n\n".join(texts)
        return aggregated, per_field, texts
