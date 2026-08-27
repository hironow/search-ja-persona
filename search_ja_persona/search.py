from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .embeddings import Embedder
from .persona_fields import PERSONA_TEXT_FIELDS
from .services import ElasticsearchService, Neo4jService, QdrantService

# Reciprocal Rank Fusion: score(d) = Σ_leg w_leg / (RRF_K + rank_leg(d)).
# K=60 is the standard constant; the weights are (vector, keyword) and the
# production default was ratified by the 2026-08-27 A/B evaluation.
RRF_K = 60
DEFAULT_RRF_WEIGHTS: tuple[float, float] = (1.0, 1.0)

# Both legs are asked for more candidates than the caller wants: RRF gets
# more rank evidence, capped so latency stays bounded, but never below the
# requested limit (the limit contract wins over the cap).
_FETCH_CAP = 30


def _fetch_depth(limit: int) -> int:
    return max(limit, min(limit * 3, _FETCH_CAP))


def _normalize_uuid(value: object) -> str:
    # Qdrant echoes UUID point ids in the canonical hyphenated form, while
    # the dataset, Elasticsearch ids, and Neo4j nodes all use the hyphen-less
    # hex. Fuse everything on the dataset form so dedup and graph lookups
    # match across backends.
    try:
        return UUID(str(value)).hex
    except (ValueError, TypeError):
        return str(value)


@dataclass
class PersonaSearchService:
    embedder: Embedder
    qdrant: QdrantService
    elasticsearch: ElasticsearchService
    neo4j: Neo4jService
    persona_fields: tuple[str, ...] = PERSONA_TEXT_FIELDS
    rrf_weights: tuple[float, float] = DEFAULT_RRF_WEIGHTS

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        return_stats: bool = False,
        prefecture: str | None = None,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fuse both retrieval legs with weighted RRF.

        Ordering is fully specified: RRF score desc, then number of source
        legs desc, then best single-leg rank asc, then uuid asc — so results
        never depend on dict insertion order. ``score`` keeps its historical
        meaning (Qdrant score when the vector leg saw the hit, otherwise the
        Elasticsearch ``_score``); ranking uses ``rrf_score``. Neo4j context
        is fetched only for the returned top-``limit``.
        """

        if limit <= 0:
            results: list[dict[str, Any]] = []
            return (
                (results, {"vector_hits": 0, "keyword_hits": 0, "context_calls": 0})
                if return_stats
                else results
            )

        depth = _fetch_depth(limit)
        query_vector = self.embedder.embed_query(query)
        vector_hits = self.qdrant.search(
            query_vector, limit=depth, prefecture=prefecture
        )
        keyword_response = self.elasticsearch.search(
            query, limit=depth, prefecture=prefecture
        )
        keyword_hits = keyword_response.get("hits", {}).get("hits", [])

        vector_ranks: dict[str, int] = {}
        vector_docs: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            uuid = _normalize_uuid(hit.get("id"))
            if uuid in vector_ranks:
                continue
            vector_ranks[uuid] = rank
            vector_docs[uuid] = hit

        keyword_ranks: dict[str, int] = {}
        keyword_docs: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(keyword_hits, start=1):
            uuid = _normalize_uuid(hit.get("_id"))
            if uuid in keyword_ranks:
                continue
            keyword_ranks[uuid] = rank
            keyword_docs[uuid] = hit

        vector_weight, keyword_weight = self.rrf_weights
        fused: list[tuple[tuple[float, int, int, str], str]] = []
        for uuid in set(vector_ranks) | set(keyword_ranks):
            rrf_score = 0.0
            source_count = 0
            best_rank = _FETCH_CAP + 1
            if uuid in vector_ranks:
                rrf_score += vector_weight / (RRF_K + vector_ranks[uuid])
                source_count += 1
                best_rank = min(best_rank, vector_ranks[uuid])
            if uuid in keyword_ranks:
                rrf_score += keyword_weight / (RRF_K + keyword_ranks[uuid])
                source_count += 1
                best_rank = min(best_rank, keyword_ranks[uuid])
            fused.append(((-rrf_score, -source_count, best_rank, uuid), uuid))
        fused.sort(key=lambda item: item[0])

        stats: dict[str, Any] = {
            "vector_hits": len(vector_hits),
            "keyword_hits": len(keyword_hits),
            "context_calls": 0,
        }

        results = []
        for sort_key, uuid in fused[:limit]:
            vector_hit = vector_docs.get(uuid)
            keyword_hit = keyword_docs.get(uuid)
            source = (keyword_hit or {}).get("_source", {})
            payload = (vector_hit or {}).get("payload", {})
            doc = source or payload

            if vector_hit is not None:
                score = vector_hit.get("score", 0.0)
            else:
                score = (keyword_hit or {}).get("_score", 0.0)

            per_field = {field: source.get(field, "") for field in self.persona_fields}
            persona_fields = per_field if source else payload.get("persona_fields", {})

            sources = []
            if vector_hit is not None:
                sources.append("vector")
            if keyword_hit is not None:
                sources.append("keyword")

            context = self.neo4j.fetch_persona_context(uuid)
            stats["context_calls"] += 1
            results.append(
                {
                    "uuid": doc.get("uuid", uuid),
                    "score": score,
                    "rrf_score": -sort_key[0],
                    "sources": sources,
                    "text": doc.get("text"),
                    "prefecture": doc.get("prefecture"),
                    "region": doc.get("region"),
                    "context": context,
                    "persona_fields": persona_fields,
                }
            )

        stats["results"] = len(results)

        if return_stats:
            return results, stats
        return results
