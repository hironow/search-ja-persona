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
PersonaRepository (stream with batch/limit)
      |
      v
PersonaIndexer
      |  compose text --> strip person names --> embed (batched)
      |
      +---> QdrantService (vector points; original text in payload)
      +---> ElasticsearchService (keyword documents, names included)
      +---> Neo4jService (graph nodes)

Query text
      |
      v
PersonaSearchService
      |
      +---> Qdrant vector leg  ---+   both legs fetch
      +---> ES keyword leg     ---+   max(limit, min(3*limit, 30))
      |                           |
      |                     RRF fusion (K=60, weights 1:1)
      |                           |
      +---> Neo4j context for the fused top-limit only
      |
      v
Fused Results (rrf_score + score + sources + text + context)
```

Legend:
- Parquet Shards: データセットの分割ファイル
- PersonaRepository: Parquet ストリーミング読み出し
- PersonaIndexer: 一括インデクサ（埋め込み入力からは人名を除去）
- vector leg / keyword leg: ベクトル検索側 / キーワード検索側
- RRF fusion: 順位ベースの融合（Reciprocal Rank Fusion）
- Fused Results: 融合済み検索結果

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
| `evaluation.py` | Golden-query loading/validation, precision/recall metrics, report assembly, threshold gate |
| `name_stripping.py` | Person-name detection and removal for embedding inputs |
| `prefectures.py` | Official 47-prefecture names and input validation |
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
keyword-leg presence (a BM25 rank-1 is a stronger identity signal than a
vector rank-1 among near-duplicates), then uuid. Neo4j context is fetched
only for the returned top-`limit`.

- `rrf_score` = the fusion score the ranking is based on
- `score` keeps its historical meaning: Qdrant cosine similarity when the
  vector leg saw the hit, otherwise the Elasticsearch relevance score
- `sources` lists the legs that returned the hit
- `--verbose` mode reveals `vector_hits`, `keyword_hits`, `context_calls`
  counts (leg counts are fetch-depth sized; context is top-`limit` only)

Embedding inputs exclude person names (`strip-person-names-v1`,
`search_ja_persona/name_stripping.py`): vectors stop chasing surnames and
given names, while stored text keeps them so BM25 name lookup and display
are unchanged (ADR 0003).

## Prefecture Filter

`search --prefecture <official name>` restricts both legs to residents of
one prefecture: a Qdrant payload filter (backed by a keyword payload index,
created at index time; `just ensure-payload-index` backfills older
collections) plus an Elasticsearch `term` filter. Inputs are validated
against the official 47 names at the entry points (CLI, golden set), so a
colloquial form like 沖縄 fails fast instead of silently matching nothing.

## Search-Quality Gate

`just eval --check-thresholds` runs the golden benchmark against the live
index and exits non-zero when a ratified bar or a required metric is
missing or unmet (`search_ja_persona/evaluation.py:check_thresholds`):

- golden mean precision@5, basic tier >= 0.85 / hard tier >= 0.55
- filtered geo mean >= 0.90
- self-retrieval recall@1 >= 0.90 and recall@10 >= 0.99
- silent-death canaries: 3-store count agreement, graph-context
  coverage >= 0.99, keyword-leg contribution > 0

Companion harnesses: `just diagnose` scores every golden predicate against
fused / vector-only / keyword-only / random rankings (discriminative-power
check), and `just eval-names` tracks exact full-name lookup on a fixed
40-name fixture. Golden data lives in `scripts/golden_queries.json`
(runtime-validated); measurement history lives in `docs/research/`.

## Metadata Persistence

Index metadata is cached in `.cache/index_metadata.json` (CWD-relative):

- Embedder preset and configuration
- Vector dimensions
- Persona fields used
- Embedding text policy (`strip-person-names-v1`)
- Collection/index names
- Schema version

This enables automatic reuse of settings across `index` and `search`
commands. When the file is missing or unusable, `search` without
`--embedder` and `index` against a populated collection both fail closed
(guessing the embedder risks wrong-dimension queries or destructive
resets); the read-only `repair-metadata` subcommand re-records the file
after verifying the declared preset against the live collection's vector
dimension and a stored point's persona fields.

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
