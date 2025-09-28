from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .services import ElasticsearchService, Neo4jService, QdrantService
from .vectorizer import HashedNgramVectorizer


@dataclass
class PersonaSearchService:
    vectorizer: HashedNgramVectorizer
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        query_vector = self.vectorizer.embed(query)
        vector_hits = self.qdrant.search(query_vector, limit=limit)
        keyword_response = self.elasticsearch.search(query, limit=limit)
        keyword_hits = keyword_response.get("hits", {}).get("hits", [])

        keyword_map = {
            str(hit.get("_id")): {
                "uuid": hit.get("_source", {}).get("uuid") or hit.get("_id"),
                "text": hit.get("_source", {}).get("text"),
                "prefecture": hit.get("_source", {}).get("prefecture"),
                "region": hit.get("_source", {}).get("region"),
                "score": hit.get("_score", 0.0),
            }
            for hit in keyword_hits
        }

        combined: list[dict[str, Any]] = []
        seen: set[str] = set()

        for hit in vector_hits:
            uuid = str(hit.get("id"))
            payload = hit.get("payload", {})
            doc = keyword_map.get(uuid, payload)
            context = self.neo4j.fetch_persona_context(uuid)
            combined.append(
                {
                    "uuid": doc.get("uuid", uuid),
                    "score": hit.get("score", 0.0),
                    "text": doc.get("text"),
                    "prefecture": doc.get("prefecture"),
                    "region": doc.get("region"),
                    "context": context,
                }
            )
            seen.add(uuid)

        for uuid, doc in keyword_map.items():
            if uuid in seen:
                continue
            context = self.neo4j.fetch_persona_context(uuid)
            combined.append(
                {
                    "uuid": doc.get("uuid", uuid),
                    "score": doc.get("score", 0.0),
                    "text": doc.get("text"),
                    "prefecture": doc.get("prefecture"),
                    "region": doc.get("region"),
                    "context": context,
                }
            )
            seen.add(uuid)

        return combined[:limit]
