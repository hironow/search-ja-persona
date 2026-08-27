# Architecture

This document describes the current architecture of the search-ja-persona system.

## System Overview

search-ja-persona is a CLI tool for indexing and searching the Nemotron Personas Japan dataset (1M Japanese personas) using three complementary search backends:

- **Qdrant** - Vector similarity search (semantic matching)
- **Elasticsearch** - Keyword search (text matching)
- **Neo4j** - Knowledge graph (relationship context)

## Data Flow

```
Parquet Shards
      |
      v
PersonaRepository (streams records with batch/limit)
      |
      v
PersonaIndexer (normalizes text, builds embeddings)
      |
      +---> QdrantService (vector points)
      +---> ElasticsearchService (keyword documents)
      +---> Neo4jService (graph nodes)
      |
      v
PersonaSearchService (query execution)
      |
      +---> Qdrant (vector search)
      +---> Elasticsearch (keyword fallback)
      +---> Neo4j (context enrichment)
      |
      v
Merged Results (score + text + context)
```

Legend:
- PersonaRepository: Repository for streaming Parquet data
- PersonaIndexer: Indexer for batch ingestion
- PersonaSearchService: Service for query orchestration

## Module Structure

| Module | Responsibility |
|--------|----------------|
| `cli.py` | CLI entry point, argument parsing, command dispatch |
| `application.py` | High-level API, factory pattern for service assembly |
| `repository.py` | Parquet streaming with batch and limit support |
| `indexer.py` | Batch ingestion into all three backends |
| `search.py` | Query execution and result fusion |
| `embeddings.py` | Text vectorization (hashed, SentenceTransformers, FastEmbed) |
| `services.py` | HTTP transports for emulator APIs |
| `datasets.py` | HuggingFace dataset download helpers |
| `manifest.py` | Parquet file manifest utilities |
| `persona_fields.py` | Persona text field definitions |

## Embedder System

The system supports multiple embedding backends through a common protocol:

```python
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
```

Queries and documents embed asymmetrically: retrieval models expect a
side-specific prefix, declared per preset and applied automatically
(search uses the query prefix, indexing the document prefix).

Available presets:

| Preset | Type | Model | Dimensions | Prefixes (query / document) |
|--------|------|-------|------------|------------------------------|
| `hashed` | Hashed n-gram | N/A | 256 (configurable) | — |
| `mini-lm` | SentenceTransformers | all-MiniLM-L6-v2 | 384 | — |
| `mpnet` | SentenceTransformers | all-mpnet-base-v2 | 768 | — |
| `e5-small` | SentenceTransformers | multilingual-e5-small | 384 | `query: ` / `passage: ` |
| `e5-large` | SentenceTransformers | multilingual-e5-large | 1024 | `query: ` / `passage: ` |
| `fast-e5-small` | FastEmbed (ONNX) | multilingual-e5-small | 384 | `query: ` / `passage: ` |
| `fast-e5-large` | FastEmbed (ONNX) | multilingual-e5-large | 1024 | `query: ` / `passage: ` |
| `ruri-v3-310m` | SentenceTransformers | cl-nagoya/ruri-v3-310m | 768 | `検索クエリ: ` / `検索文書: ` |
| `ruri-v3-130m` | SentenceTransformers | cl-nagoya/ruri-v3-130m | 512 | `検索クエリ: ` / `検索文書: ` |

The Ruri presets also cap the encode batch size (16 / 32) at their
measured RTX 4090 throughput optima.

## Persona Data Model

Each persona record contains:

| Field | Description |
|-------|-------------|
| `uuid` | Unique identifier |
| `professional_persona` | Professional background |
| `sports_persona` | Sports interests |
| `arts_persona` | Arts and culture interests |
| `travel_persona` | Travel preferences |
| `culinary_persona` | Food and dining preferences |
| `persona` | Aggregated persona text |
| `prefecture` | Japanese prefecture |
| `region` | Japanese region |
| `occupation` | Occupation |
| `age` | Age |
| `sex` | Sex |

## Search Result Format

```python
{
    "uuid": "...",
    "score": 0.87,  # cosine similarity (Qdrant) or ES _score
    "rrf_score": 0.0325,  # weighted RRF score used for ranking
    "sources": ["vector", "keyword"],  # which legs returned the hit
    "text": "...",  # aggregated persona text
    "prefecture": "...",
    "region": "...",
    "context": {...},  # Neo4j graph relationships
    "persona_fields": {...},  # per-field breakdown
}
```

## Fusion and Score Semantics

Results are ordered by weighted Reciprocal Rank Fusion over both legs:
`rrf_score(d) = Σ_leg w_leg / (60 + rank_leg(d))`, with production weights
1:1. Both legs are queried at `max(limit, min(limit * 3, 30))` depth for
rank evidence; ties break by source count, then best single-leg rank, then
uuid. Neo4j context is fetched only for the returned top-`limit`.

- `rrf_score` = the fusion score the ranking is based on
- `score` keeps its historical meaning: Qdrant cosine similarity when the
  vector leg saw the hit, otherwise the Elasticsearch relevance score
- `sources` lists the legs that returned the hit
- `--verbose` mode reveals `vector_hits`, `keyword_hits`, `context_calls`
  counts (leg counts are fetch-depth sized; context is top-`limit` only)

## Metadata Persistence

Index metadata is cached in `.cache/index_metadata.json`:

- Embedder preset and configuration
- Vector dimensions
- Persona fields used
- Collection/index names
- Schema version

This enables automatic reuse of settings across `index` and `search` commands.

## Emulator Infrastructure

The system requires three local emulators, provided by a standalone Docker
Compose stack in `emulator/` (`cd emulator && docker compose up -d`). It is a
vendored minimal subset of github.com/hironow/emulator-set. See
`emulator/README.md`.

| Service | Port | Purpose |
|---------|------|---------|
| Qdrant | 6333 | Vector search with cosine distance |
| Elasticsearch | 9200 | Full-text keyword search |
| Neo4j | 7474 | Graph database for context |
