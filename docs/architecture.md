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
| `dataset_profiles.py` | Per-dataset settings: columns, text and payload fields, filters, and the dataset's own Qdrant collection, Elasticsearch index, Neo4j label, and metadata file |

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

## Dataset Profiles

Three datasets are indexed side by side. Each has a profile
(`search_ja_persona/dataset_profiles.py`, CLI `--dataset-profile`, default
`japan`) that owns its stores, so indexing, searching, or clearing one never
touches another:

| Profile | Source | Text fields | Payload / filters | Stores (Qdrant = ES name) | Neo4j graph | Embedding input |
|---------|--------|-------------|-------------------|---------------------------|-------------|-----------------|
| `japan` | Nemotron-Personas-Japan (1M) | 6 persona fields | `prefecture`, `region` / `prefecture` | `personas` | `Persona` -LIVES_IN-> `Prefecture`, -LOCATED_IN-> `Region` | `strip-person-names-v1` |
| `finance` | Nemotron-Personas-Finance-Japan (1M) | 3 finance personas + the 6 above | same as japan | `finance_personas` | `FinancePersona`, same relationships | `strip-person-names-v1` |
| `pii` | Nemotron-PII-Japan (100k documents) | `document_text`, `document_description` | `domain`, `document_type` / both | `pii_documents` | `PiiDocument` -IN_DOMAIN-> `PiiDomain`, -OF_TYPE-> `PiiDocumentType`, -CONTAINS_PII-> `PiiLabel` | `raw-text-v1` |

- Every record is keyed by `uuid`; the pii profile renames the source `uid`
  to `uuid` and `text` to `document_text` (the aggregated `text` field
  stays unambiguous). PII span labels become `PiiLabel` nodes; spans are not
  stored in Qdrant or Elasticsearch.
- `Prefecture` / `Region` nodes are shared reference data. Constraints are
  named after the label (`persona_uuid`, `finance_persona_uuid`,
  `pii_document_uuid`), and clearing a dataset deletes only its own nodes.
- Guards: the repository detects a parquet file's dataset from its columns
  (`uid` + `spans` → pii, `finance_persona` → finance, `uuid` → japan) and
  refuses a mismatch; a profile refuses another profile's collection or index
  name; metadata records its dataset, and a mismatch fails closed.

## Persona Data Model

Each japan persona record contains (finance adds `finance_persona`,
`financial_behavior_persona`, and `financial_concerns_persona`):

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
    "text": "...",  # aggregated text of the indexed fields
    "prefecture": "...",  # the profile's payload fields
    "region": "...",  # (pii: "domain", "document_type")
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

## Search Filters

Filters restrict both legs to an exact payload value: a Qdrant payload
filter (backed by a keyword payload index per filter field, created at index
time; `just ensure-payload-index` backfills older collections) plus an
Elasticsearch `term` filter. Each profile accepts only its own filter flags:

- japan / finance: `search --prefecture <official name>`. Inputs are
  validated against the official 47 names at the entry points (CLI, golden
  set), so a colloquial form like 沖縄 fails fast instead of silently
  matching nothing.
- pii: `search --domain <domain>` and `--document-type <type>`.

## Search-Quality Gate

`just eval --check-thresholds` runs the golden benchmark against the live
index and exits non-zero when a ratified bar or a required metric is
missing or unmet (`search_ja_persona/evaluation.py:check_thresholds`).
`just profile=<name> eval` evaluates another dataset against its own golden
set (`--dataset-profile`); every report records its `dataset`.

- japan only (ratified): golden mean precision@5, basic tier >= 0.85 /
  hard tier >= 0.55; filtered geo mean >= 0.90; self-retrieval
  recall@1 >= 0.90 and recall@10 >= 0.99
- every dataset: silent-death canaries — 3-store count agreement,
  graph-context coverage >= 0.99, keyword-leg contribution > 0. finance
  and pii report their quality metrics without a bar until the requester
  ratifies one.

Golden expectations may use `text_any` / `text_all` plus the profile's
payload fields, filters only the profile's filter fields, and a filtered
entry must expect something beyond its filter (otherwise it scores 1.0 by
construction). Self-retrieval queries a persona's `persona` summary for
japan and finance; pii skips it because one document description is shared
by about ten documents.

Companion harnesses: `just diagnose` scores every golden predicate against
fused / vector-only / keyword-only / random rankings (discriminative-power
check), and `just eval-names` tracks exact full-name lookup on a fixed
40-name fixture. Golden data lives in `scripts/golden_queries.json`
(runtime-validated); measurement history lives in `docs/research/`.

## Metadata Persistence

Index metadata is cached per dataset and collection (CWD-relative):
`.cache/index_metadata.json` (japan), `.cache/index_metadata.finance.json`,
`.cache/index_metadata.pii.json`, and
`.cache/index_metadata.<profile>.<collection>.json` for a custom collection.
Each file records:

- Dataset profile (`dataset`; files without it are japan)
- Embedder preset and configuration
- Vector dimensions
- Persona fields used
- Embedding text policy (the profile's)
- Collection name
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
