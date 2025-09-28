# search-ja-persona

Self-contained tooling for indexing and searching the Nemotron Personas Japan dataset with local emulators (Qdrant, Elasticsearch, Neo4j). The CLI coordinates data ingestion, vector and keyword search, and persona graph context so developers can explore personas without reading the source first.

## Pipeline at a Glance

Data flow

1. Parquet shards -> `PersonaRepository` (stream records with optional limits)
2. `PersonaIndexer` -> Qdrant (vector), Elasticsearch (keyword), Neo4j (context graph)
3. Query text -> embedder -> `PersonaSearchService` -> merge vector hits, keyword fallbacks, graph context

Key components

- `PersonaRepository` streams records from one or more parquet files (batch aware, limit ready).
- `PersonaIndexer` normalizes persona text fields, builds embeddings, and writes to each emulator service.
- `PersonaSearchService` embeds the query, runs Qdrant vector search, enriches results with Elasticsearch keyword hits and Neo4j persona context, then returns a combined list.

## Score Semantics

Each search result exposes a `score` field:

- For candidates returned by Qdrant, `score` is the cosine similarity reported by Qdrant (higher is better).
- When Elasticsearch supplies a fallback persona that was not in the vector shortlist, the Elasticsearch `_score` is mapped into the same `score` field.
- `--verbose` mode prints hit counts: `vector_hits`, `keyword_hits`, `context_calls`, and `results` so you can diagnose which backend produced the answer set.

## Repository Map

| Path | Purpose |
|------|---------|
| `search_ja_persona/cli.py` | Rich CLI entry point for indexing, searching, downloading, and clearing emulators |
| `search_ja_persona/indexer.py` | Batch ingestion into Qdrant, Elasticsearch, and Neo4j |
| `search_ja_persona/search.py` | Query orchestration and hit fusion logic |
| `search_ja_persona/embeddings.py` | Embedding backends (hashed n-gram, SentenceTransformers, fastembed) |
| `search_ja_persona/services.py` | Thin HTTP transports for emulator APIs |
| `qa_samples/qa_sample.parquet` | 1k-row sample used by quick QA flows |
| `scripts/generate_qa_sample.py` | Regenerate the QA sample parquet from Hugging Face |

## Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) for dependency management (recommended)
- Local emulators running: change into `emulator/` and use `./start-emulators.sh` (or `docker compose up -d`).

## Getting the Dataset

1. Hugging Face cache: run the CLI downloader (safe to rerun).
   ```bash
   uv run python -m search_ja_persona.cli download-dataset \
       --dataset-name nvidia/Nemotron-Personas-Japan \
       --split train \
       --cache-dir .cache
   ```
   The cached parquet shards sit under `~/.cache/huggingface`. You can also copy them into `datasets/Nemotron-Personas-Japan/data/` as this repo already demonstrates.

2. Optional: regenerate the bundled 1k sample parquet.
   ```bash
   uv run python -m scripts.generate_qa_sample --limit 1000
   ```

## Generating the QA Sample

The repository ships with `qa_samples/qa_sample.parquet`. Regenerate it whenever you
need a fresh slice or after updating the source dataset:

1. Ensure the Hugging Face cache already contains `nvidia/Nemotron-Personas-Japan`.
   - Run the `download-dataset` command above, or allow the script to fall back to
     an existing `.arrow` shard in the cache directory.
2. Execute the helper script (idempotent; overwrites the existing parquet):
   ```bash
   uv run python -m scripts.generate_qa_sample --limit 1000
   ```
   The script writes up to 1,000 rows into `qa_samples/qa_sample.parquet` using the persona
   text fields defined in `search_ja_persona/persona_fields.py`.

Override the default count with `--limit` when needed (for example, `--limit 2000`).

## Indexing Personas

### Full corpus (about 1,000,000 rows)

Ingest every shard in `datasets/Nemotron-Personas-Japan/data/` using a SentenceTransformer preset. Adjust `--batch-size` to match available memory; leaving `--limit` unset consumes all rows.

```bash
uv run python -m search_ja_persona.cli index \
    --dataset datasets/Nemotron-Personas-Japan/data \
    --batch-size 512 \
    --embedder mini-lm \
    --persona-fields all \
    --qdrant-host localhost --qdrant-port 6333 \
    --es-host localhost --es-port 9200 \
    --neo4j-host localhost --neo4j-port 7474
```

### QA sample (1,000 rows)

Limit ingestion to the bundled sample parquet. The `--limit` guard ensures only the first 1,000 personas are processed even if you regenerate the sample with more rows.

```bash
uv run python -m search_ja_persona.cli index \
    --dataset qa_samples/qa_sample.parquet \
    --batch-size 128 \
    --limit 1000 \
    --embedder mini-lm \
    --persona-fields all
```

> Tip: `just qa-index embedder="mini-lm" persona_fields="all"` wraps the same command.

### Metadata Tracking

After every indexing run, `.cache/index_metadata.json` records the embedder preset, dimensions, persona fields, and collection/index names. Subsequent `search` runs reuse this metadata when you omit `--embedder` or `--persona-fields`.

## Running Searches

Once indexing completes, issue free-text queries with combined vector plus keyword retrieval.

```bash
uv run python -m search_ja_persona.cli search \
    --query "care manager with elder care experience" \
    --limit 5 \
    --format table \
    --verbose
```

- `--format table` renders a Rich table; `--format json` prints structured JSON.
- `--verbose` surfaces per-backend hit statistics alongside the unified result list.
- To reuse the last indexed embedder or persona field set, omit `--embedder` and `--persona-fields` (the CLI will read `.cache/index_metadata.json`).

## Maintenance Utilities

- `uv run python -m search_ja_persona.cli clear-emulators` drops the Qdrant collection, Elasticsearch index, Neo4j persona nodes, and deletes cached metadata (asks for confirmation).
- `just test` runs the full pytest suite (emulator integration tests are skipped unless the emulators are up and the dataset cache is populated).

## Troubleshooting Checklist

- Ensure `./start-emulators.sh` completed and ports 6333, 9200, 7474 are reachable.
- Hugging Face downloads require authentication when the dataset is gated; pass `--token` to `download-dataset` if needed.
- If you switch embedder presets or persona field subsets, the CLI prompts to reset existing indexes so vector dimensions stay aligned across services.

---

With the pipeline indexed, you can explore prompts against the million-persona corpus or the 1k QA slice by swapping the dataset path and `--limit` flag in the commands above.
