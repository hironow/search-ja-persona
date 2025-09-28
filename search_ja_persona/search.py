from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embeddings import Embedder
from .persona_fields import PERSONA_TEXT_FIELDS
from .services import ElasticsearchService, Neo4jService, QdrantService


@dataclass
class PersonaSearchService:
    embedder: Embedder
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService
    persona_fields: tuple[str, ...] = PERSONA_TEXT_FIELDS

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        return_stats: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        if limit <= 0:
            results: list[dict[str, Any]] = []
            return (
                (results, {"vector_hits": 0, "keyword_hits": 0, "context_calls": 0})
                if return_stats
                else results
            )

        query_vector = self.embedder.embed(query)
        vector_hits = self.qdrant.search(query_vector, limit=limit)
        keyword_response = self.elasticsearch.search(query, limit=limit)
        keyword_hits = keyword_response.get("hits", {}).get("hits", [])

        keyword_map: dict[str, dict[str, Any]] = {}
        for hit in keyword_hits:
            source = hit.get("_source", {})
            per_field = {field: source.get(field, "") for field in self.persona_fields}
            keyword_map[str(hit.get("_id"))] = {
                "uuid": source.get("uuid") or hit.get("_id"),
                "text": source.get("text"),
                "prefecture": source.get("prefecture"),
                "region": source.get("region"),
                "score": hit.get("_score", 0.0),
                "persona_fields": per_field,
            }

        stats: dict[str, Any] = {
            "vector_hits": len(vector_hits),
            "keyword_hits": len(keyword_hits),
            "context_calls": 0,
        }

        combined: list[dict[str, Any]] = []
        seen: set[str] = set()

        for hit in vector_hits:
            uuid = str(hit.get("id"))
            payload = hit.get("payload", {})
            doc = keyword_map.get(uuid, payload)
            context = self.neo4j.fetch_persona_context(uuid)
            stats["context_calls"] += 1
            combined.append(
                {
                    "uuid": doc.get("uuid", uuid),
                    "score": hit.get("score", 0.0),
                    "text": doc.get("text"),
                    "prefecture": doc.get("prefecture"),
                    "region": doc.get("region"),
                    "context": context,
                    "persona_fields": doc.get(
                        "persona_fields", payload.get("persona_fields", {})
                    ),
                }
            )
            seen.add(uuid)

        for uuid, doc in keyword_map.items():
            if uuid in seen:
                continue
            context = self.neo4j.fetch_persona_context(uuid)
            stats["context_calls"] += 1
            combined.append(
                {
                    "uuid": doc.get("uuid", uuid),
                    "score": doc.get("score", 0.0),
                    "text": doc.get("text"),
                    "prefecture": doc.get("prefecture"),
                    "region": doc.get("region"),
                    "context": context,
                    "persona_fields": doc.get("persona_fields", {}),
                }
            )
            seen.add(uuid)

        results = combined[:limit]
        stats["results"] = len(results)

        if return_stats:
            return results, stats
        return results
