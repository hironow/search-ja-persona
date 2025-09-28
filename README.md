# search-ja-persona

## Overview

`search-ja-persona` orchestrates the Nemotron-Personas-Japan dataset across Qdrant (vector), Elasticsearch (keyword), and Neo4j (graph) emulator services. It provides:

- Streaming repository + hashed n-gram embedder for deterministic vectors.
- Service clients and an indexer that batch-sync personas to each emulator.
- A persona search service (and CLI) that fuses vector/keyword hits and enriches results with graph context.
- TDD/Tidy-First development workflow (see `plan.md` and `AGENTS.md`).

### Dataset Highlights

- Source: `nvidia/Nemotron-Personas-Japan` (v1.0, released 2025-09-23).
- Scale: 1,000,000 personas (~1.73 GB parquet shards).
- Rich schema: narrative personas plus 16 demographic/context attributes spanning all 47 prefectures and >1,500 occupations.
- License: CC BY 4.0 synthetic data aimed at bias reduction and sovereign-AI scenarios.

## Requirements

- Python ≥ 3.12 (managed via `uv`).
- Docker Desktop (for emulator stack).
- Sufficient storage for the Hugging Face dataset cache (~2 GB).

Install dependencies in a `uv` environment:

```
uv sync
```

## Quick Usage

```python
from pathlib import Path
from search_ja_persona.application import ApplicationConfig, PersonaApplication

config = ApplicationConfig(vector_dimension=256, ngram_sizes=(2, 3))
app = PersonaApplication.build(config)

# Index personas from parquet shards (emulators must be running)
dataset_dir = Path("datasets/Nemotron-Personas-Japan")
app.index([dataset_dir], batch_size=128, limit=10_000)

# Run a fused search and inspect graph context
results = app.search("高齢者介護の経験豊富なマネージャー", limit=3)
for entry in results:
    print(entry["uuid"], entry["prefecture"], entry["context"].get("relationships"))
```

### Dataset Download Helpers

Cache the full dataset locally (defaults to `.cache/` unless overridden with `--cache-dir`):

```
uv run python -m search_ja_persona.cli download-dataset --cache-dir .hf-cache
```

Sample a single persona from the cache:

```
uv run python load_dataset_example.py
```

## Command Line Interface

```
uv run python -m search_ja_persona.cli --help
```

Key subcommands:

- `download-dataset`: Wraps `datasets.load_dataset` (`--cache-dir`, `--force`, `--revision`, `--token`).
- `index`: Streams parquet shards into Qdrant/Elasticsearch/Neo4j. Choose `--embedder hashed` (default) or semantic presets such as `mini-lm`, `mpnet`, `e5-small`, `e5-large`, `fast-e5-small`, `fast-e5-large`. Advanced parameters like `--embedder-model`/`--embedder-device`, `--fastembed-cache-dir` (default `.cache/`), and `--persona-fields` (comma-separated list or `all`) let you shape the embedding input.
- `search`: Embeds a free-text query and prints Rich tables (`--format json` for raw output). Supports the same `--embedder` presets and reuses the latest indexed persona fields automatically when `--persona-fields` is omitted.

Example indexing run (after caching data and starting emulators):

```
uv run python -m search_ja_persona.cli index \
  --dataset datasets/Nemotron-Personas-Japan \
  --batch-size 256 --limit 5000
```

### End-to-End Search Flow

When you run the search command, the system performs three steps:

1. **Vector similarity (Qdrant)** – The query text is embedded via the hashed n-gram vectorizer and compared against each persona narrative to gather semantic candidates.
2. **Keyword refinement (Elasticsearch)** – The same query runs through a `multi_match` across `persona`, `prefecture`, and `region`, boosting candidates that also match lexically or geographically.
3. **Graph context (Neo4j)** – For each resulting persona `uuid`, Neo4j is queried so the response can include relationship metadata (e.g., linked regions or related entities).

The CLI merges these stages into a single response: Qdrant scores, Elasticsearch payload, and optional Neo4j relationships per persona. By default results render as a Rich table showing every persona narrative column (professional/sports/arts/travel/culinary/summary); append `--format json` to print the raw JSON payload instead. The most recent indexing run records its embedder preset and persona-field selection in `.cache/index_metadata.json`, so subsequent `search` invocations automatically reuse those choices when flags are omitted.

## Integration Testing

1. Start the emulator stack:

   ```
   (cd emulator && ./start-emulators.sh)
   ```

2. Provide persona data. The integration test automatically reuses `.hf-cache/` if present; you can override via:

   ```
   export PERSONA_DATASET_CACHE_DIR=$(pwd)/.hf-cache
   export PERSONA_DATASET_SAMPLE_PATH=/path/to/sample.parquet
   export PERSONA_DATASET_DIR=/path/to/parquet_dir
   ```

3. Run the integration test (also part of the default suite when prerequisites are satisfied):

   ```
   uv run pytest tests/test_integration_emulators.py -m integration
   ```

Running `UV_CACHE_DIR=.uv-cache uv run pytest` executes the entire test suite (unit + integration). The integration case skips only when emulators or dataset shards are unavailable.

## Development Notes

- `plan.md` documents the roadmap (manifest generation, resumable indexing, etc.).
- `tests/` follows TDD structure; emulator submodule tests remain excluded via `norecursedirs`.
- `AGENTS.md` captures Tidy First/TDD guidelines used throughout the repository.
