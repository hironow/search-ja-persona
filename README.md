# search-ja-persona

Index and search three NVIDIA Japanese datasets with local emulators: Qdrant (vector), Elasticsearch (keyword), and Neo4j (graph context).
The CLI ingests parquet files, runs vector and keyword search, fuses the two rankings, and adds graph context, so you can explore the data without reading the source first.

| Dataset | Records | Profile | What it holds |
|---------|---------|---------|---------------|
| [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) | 1,000,000 | `japan` (default) | Synthetic personas: 6 persona texts, prefecture, region, occupation, age, sex |
| [Nemotron-Personas-Finance-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Finance-Japan) | 1,000,000 | `finance` | The same persona fields plus 3 finance persona texts and household finance attributes (income, financial assets, financial literacy, …) |
| [Nemotron-PII-Japan](https://huggingface.co/datasets/nvidia/Nemotron-PII-Japan) | 100,000 | `pii` | Synthetic documents in 8 domains (medical, finance, government, …) with PII span annotations; not personas |

All three are CC BY 4.0.

## How a Search Works

1. Parquet files -> `PersonaRepository` streams records (batch aware, with an optional limit).
2. `PersonaIndexer` composes each record's text fields, embeds them in batches, and writes to Qdrant, Elasticsearch, and Neo4j.
3. A query -> the embedder -> `PersonaSearchService` runs the Qdrant vector leg and the Elasticsearch keyword leg, fuses both rankings with weighted Reciprocal Rank Fusion (RRF, K=60, production weights 1:1), and adds Neo4j context to the fused top hits.

For the persona datasets (japan, finance), embedding inputs exclude person names.
Otherwise vectors chase surnames: a 福岡-surnamed persona ranked high for 福岡県 queries.
Stored text keeps the names, so keyword name lookup and display are unchanged.
pii embeds the document text as-is.

## Each Dataset Has Its Own Stores

A **dataset profile** (`search_ja_persona/dataset_profiles.py`) owns each dataset's stores.
Indexing, searching, or clearing one dataset never touches another.
Select it with `--dataset-profile` on the CLI (default `japan`) or with the justfile's `profile` variable (`just profile=finance ...`).

| Profile | Source | Qdrant collection / ES index | Neo4j label | Metadata file |
|---------|--------|------------------------------|-------------|---------------|
| `japan` | `datasets/Nemotron-Personas-Japan/data` | `personas` | `Persona` | `.cache/index_metadata.json` |
| `finance` | `datasets/Nemotron-Personas-Finance-Japan/data` | `finance_personas` | `FinancePersona` | `.cache/index_metadata.finance.json` |
| `pii` | `datasets/Nemotron-PII-Japan` (`data.parquet`) | `pii_documents` | `PiiDocument` | `.cache/index_metadata.pii.json` |

Guards keep the datasets apart:

- Before reading a parquet file, the repository detects its dataset from the source columns (`uid` + `spans` → `pii`, `finance_persona` → `finance`, otherwise `uuid` → `japan`) and refuses a file that does not match the selected profile.
  Finance shards share file names with the japan shards, so indexing them needs `--dataset-profile finance`.
- A profile refuses `--qdrant-collection` / `--es-index` names that belong to another profile.
- Metadata records its dataset, and a mismatch stops the command.
- A custom collection name gets its own metadata file (`.cache/index_metadata.<profile>.<collection>.json`), so clearing it keeps the default index's metadata.
- `Prefecture` and `Region` graph nodes are shared reference data; clearing a dataset deletes only its own nodes.

See `docs/architecture.md` (Dataset Profiles) for each profile's fields and graph.

## Prerequisites

- Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) for dependency management.
- Local emulators: `cd emulator && docker compose up -d` (Qdrant, Elasticsearch, Neo4j). See [`emulator/README.md`](emulator/README.md). On Windows, Docker Desktop provides the engine.
- On Windows, `just` recipes run under `bash`, so Git for Windows' `bash.exe` must be on `PATH`. It lives in `C:\Program Files\Git\bin`; the default installer adds only `Git\cmd`.

## Getting the Data

Each dataset ships as a Git LFS submodule under `datasets/`.
Check out the one you need by path and pull its LFS blobs:

```bash
git submodule update --init datasets/Nemotron-Personas-Japan
git -C datasets/Nemotron-Personas-Japan lfs pull
git submodule update --init datasets/Nemotron-Personas-Finance-Japan
git -C datasets/Nemotron-Personas-Finance-Japan lfs pull
git submodule update --init datasets/Nemotron-PII-Japan
git -C datasets/Nemotron-PII-Japan lfs pull
```

Always pass a path.
A bare `git submodule update --init` checks out every dataset submodule, and the git-lfs smudge filter then downloads all of their files (about 3 GB).

| Submodule | Files | Working tree |
|-----------|-------|--------------|
| `datasets/Nemotron-Personas-Japan` | 8 parquet shards under `data/` | ~1.6 GB |
| `datasets/Nemotron-Personas-Finance-Japan` | 8 parquet shards under `data/` (50 fields) | ~1.4 GB |
| `datasets/Nemotron-PII-Japan` | one `data.parquet` | ~76 MB |

For japan only, a Hugging Face cache can replace (or sit next to) the submodule.
The command is safe to rerun, and the `.arrow` shards land under the `--cache-dir` you pass:

```bash
uv run python -m search_ja_persona.cli download-dataset \
    --dataset-name nvidia/Nemotron-Personas-Japan \
    --split train \
    --cache-dir .cache
```

## Generating the QA Sample

The QA flows read a 1,000-row sample per dataset.
The samples are git-ignored, so generate one before the first `just qa` run, and again whenever you want a fresh slice or the source dataset changed.
The command is idempotent and overwrites the existing file.

| Profile | Command | Output | Reads from |
|---------|---------|--------|------------|
| `japan` | `just qa-sample` | `qa_samples/qa_sample.parquet` | The Hugging Face cache (`download-dataset` above), falling back to an existing `.arrow` shard in the cache directory |
| `finance` | `just profile=finance qa-sample` | `qa_samples/qa_sample.finance.parquet` | The first submodule shard, all source columns |
| `pii` | `just profile=pii qa-sample` | `qa_samples/qa_sample.pii.parquet` | `data.parquet`, all source columns |

Change the row count with the recipe argument (`just qa-sample 2000`), or call the script directly:

```bash
uv run python -m scripts.generate_qa_sample --limit 2000 --profile japan
```

## Indexing

### Full dataset

```bash
just full-index                    # japan: 8 shards -> personas / Persona
just profile=finance full-index    # 8 shards -> finance_personas / FinancePersona
just profile=pii full-index        # data.parquet -> pii_documents / PiiDocument
```

The recipe indexes one parquet file at a time with the `ruri-v3-310m` preset and a batch size of 512.
Each file is a checkpoint: uuid-keyed upserts make reruns idempotent, so after a failure rerun from the failed shard.
On an RTX 4090, japan takes about 50 minutes; finance took 151 minutes while sharing the GPU with another model server, and pii 14 minutes.

`ruri-v3-310m` (cl-nagoya/ruri-v3-310m, 768 dims, Apache-2.0) is the recommended preset for Japanese.
Its tokenizer avoids the ~23% document truncation multilingual-e5 suffers on this corpus, and the preset applies the `検索クエリ: ` / `検索文書: ` prefixes and its measured encode batch cap automatically.
Emulator hosts and ports default to the local stack, so no host flags are needed.
The equivalent single-shard command:

```bash
uv run python -m search_ja_persona.cli index \
    --dataset-profile japan \
    --dataset datasets/Nemotron-Personas-Japan/data/train-00000-of-00008.parquet \
    --batch-size 512 \
    --embedder ruri-v3-310m \
    --persona-fields all
```

Adjust `--batch-size` (the ingest batch) to the available memory; leaving `--limit` unset reads all rows.

### QA sample

```bash
just qa-index mini-lm all                    # japan sample, mini-lm, all text fields
just profile=finance qa-index ruri-v3-310m   # finance sample
```

`qa-index` uses `--batch-size 64 --limit 1000`, so only the first 1,000 records are indexed even if the sample has more.
Recipe arguments are positional (embedder, then persona fields); only the top-level `profile` variable takes `name=value`.
The equivalent CLI command:

```bash
uv run python -m search_ja_persona.cli index     --dataset qa_samples/qa_sample.parquet     --batch-size 128     --limit 1000     --embedder mini-lm     --persona-fields all
```

### Index metadata

Every index run records the embedder preset, dimensions, text fields, embedding text policy, and collection name in the profile's metadata file (see the table above).
Later `search` runs reuse it when you omit `--embedder` or `--persona-fields`.
If you switch embedder presets or text field subsets, the CLI asks before resetting the profile's stores, so vector dimensions stay aligned.

## Searching

```bash
uv run python -m search_ja_persona.cli search \
    --query "care manager with elder care experience" \
    --limit 5 \
    --format table \
    --verbose
```

- `--format table` renders a Rich table; `--format json` prints structured JSON.
- `--dataset-profile finance` or `pii` searches that dataset's stores.
- Filters restrict both legs to an exact value (a Qdrant payload filter plus an Elasticsearch term filter). Each profile accepts only its own filters:
  - japan / finance: `--prefecture 北海道`. Use official names (`沖縄県`, not `沖縄`). Collections indexed before this filter existed need a one-off `just ensure-payload-index`.
  - pii: `--domain 金融` and `--document-type ローン審査書類`.
- `--verbose` prints per-backend hit counts (`vector_hits`, `keyword_hits`, `context_calls`, `results`), so you can see which backend produced the answers.

If the metadata file is missing or unusable, `search` without `--embedder` stops (the hashed default would not match the live index), and `index` refuses to touch a populated collection.
Recover with `uv run python -m search_ja_persona.cli repair-metadata --embedder <preset>` (add `--dataset-profile` for finance or pii).
It checks the collection's vector dimension and a stored point's text fields, then re-records the metadata without writing to any store.

### Score semantics

Results are ordered by `rrf_score`, the fusion score; `sources` lists the legs that returned each hit.
Each result also has a `score`:

- For a hit the vector leg returned, `score` is Qdrant's cosine similarity (higher is better).
- For a hit only Elasticsearch returned, `score` is its Elasticsearch `_score`.

## Evaluating Search Quality

```bash
just eval --check-thresholds                    # japan
just profile=finance eval --check-thresholds    # finance (or pii)
```

`just eval` benchmarks the live index and writes a report to `outputs/`.
With `--check-thresholds` it exits non-zero when a ratified bar or a required metric is unmet:

- japan only (ratified): golden precision@5 basic >= 0.85 and hard >= 0.55, filtered geo mean >= 0.90, self-retrieval recall@1 >= 0.90 and recall@10 >= 0.99.
- Every dataset: three silent-death canaries (3-store count agreement, graph-context coverage >= 0.99, keyword-leg contribution > 0). finance and pii report their quality metrics without a bar until the requester rules on one.

Companion harnesses, all per profile:

- `just diagnose` checks that the golden predicates measure retrieval rather than themselves (fused / vector / keyword / random rankings).
- `just eval-names` measures exact lookup on a fixed 40-entry fixture: full names for japan and finance, single-document PII values for pii. `just profile=<finance|pii> lookup-fixture` rebuilds a fixture (seed-fixed).
- `just store-fingerprint` records counts, write totals, and sample digests of one dataset's stores, read-only. Take it before and after working on another dataset; the two must be equal.

The bars live in `docs/intent.md`; measurements live in `docs/research/`.

## Notebooks

- `just notebook` opens the feature-catalog marimo notebook (`marimo/catalog.py`). Pick japan, finance, or pii in its dataset selector. It shows the data model, store counts, the recorded metadata, a committed snapshot of real results (renders with the emulators down), and button-triggered live search. `just profile=<name> catalog-snapshot` re-records the snapshot after a full index.
- `just panel` / `just panel-app` open the persona panel (`marimo/persona_panel.py`): the top M personas from japan or finance answer a text or image question through a local Ollama model. Answers land in `outputs/persona_panel-*.jsonl` with a `dataset` field. `marimo/panel_example.jsonl` is a bundled M=30 sample.

## Emulator Data Persistence

The emulators store data in named Docker volumes (`emulator_qdrant_data`, `emulator_elasticsearch_data`, `emulator_neo4j_data`).
Indexed data survives `docker compose stop`, `docker compose down`, Docker Desktop restarts, and OS reboots.
All three datasets together occupy about 27 GB across the volumes (japan alone about 14 GB).

Only these destroy the data:

- `docker compose down -v` (the `-v` flag deletes named volumes)
- `docker volume prune` / `docker system prune --volumes`
- Docker Desktop "Clean / Purge data" or a factory reset

If the volumes are lost, rebuild each dataset with its `full-index` recipe.
On Windows the volumes live inside Docker Desktop's WSL2 virtual disk, which grows with the data and does not shrink when data is deleted.

## Task Reference (justfile)

Run `just` (or `just help`) to list every task.
The recipes marked with * work on the dataset in the top-level `profile` variable (default `japan`), for example `just profile=finance eval --check-thresholds`.

| Task | What it does |
|------|--------------|
| `just check` | Local gate: prek hooks (ruff format and lint, file checks) and the unit suite |
| `just format` / `just lint` | Format / lint and auto-fix with ruff |
| `just test` | pytest; integration tests skip themselves unless the emulators are up |
| `just integration` | Integration tests against the running emulators |
| `just qa-sample [limit]` * | Generate the QA sample parquet |
| `just qa-index [embedder] [fields]` * | Index the QA sample |
| `just qa-search "query"` * | Search the QA sample's index |
| `just qa` * | `qa-index`, then `qa-search` |
| `just qa-clear` * | Clear the dataset's collection, index, nodes, and metadata (asks first) |
| `just full-index` * | Index the full dataset shard by shard (ruri-v3-310m) |
| `just eval` * | Search-quality benchmark; add `--check-thresholds` to enforce the bars |
| `just diagnose` * | Golden-predicate diagnostic |
| `just eval-names` * | Exact-lookup benchmark |
| `just lookup-fixture` * | Rebuild the finance / pii lookup fixture |
| `just store-fingerprint` * | Read-only fingerprint of the dataset's stores |
| `just catalog-snapshot` * | Record the catalog snapshot from the dataset's full index |
| `just ensure-payload-index` | Backfill the Qdrant prefecture payload index on an older collection (idempotent) |
| `just notebook` / `just notebook-export` | Feature-catalog notebook (edit / static HTML) |
| `just panel` / `just panel-app` | Persona-panel notebook (edit / read-mode app) |

`uv run python -m search_ja_persona.cli clear-emulators --dataset-profile <name>` is the CLI form of `qa-clear`.

## Repository Map

| Path | Purpose |
|------|---------|
| `search_ja_persona/cli.py` | CLI entry point: index, search, download-dataset, clear-emulators, repair-metadata |
| `search_ja_persona/application.py` | High-level programmatic API (`PersonaApplication`) |
| `search_ja_persona/dataset_profiles.py` | Per-dataset fields, filters, graph shape, and stores |
| `search_ja_persona/repository.py` | Parquet streaming with batch and limit support, and the dataset check |
| `search_ja_persona/indexer.py` | Batch ingestion into Qdrant, Elasticsearch, and Neo4j |
| `search_ja_persona/search.py` | Query orchestration and RRF fusion |
| `search_ja_persona/embeddings.py` | Embedding backends (hashed n-gram, SentenceTransformers, fastembed) |
| `search_ja_persona/services.py` | Thin HTTP clients for the emulator APIs |
| `search_ja_persona/evaluation.py` | Golden-set validation, precision/recall metrics, report assembly, threshold gate |
| `search_ja_persona/name_stripping.py` | Person-name removal for embedding inputs |
| `search_ja_persona/persona_fields.py` | Persona text fields (japan 6, finance 9) |
| `search_ja_persona/prefectures.py` | Official 47 prefecture names and validation |
| `search_ja_persona/datasets.py` / `manifest.py` | Hugging Face download helpers / parquet manifest utilities |
| `scripts/generate_qa_sample.py` | Build a dataset's QA sample |
| `scripts/golden_queries*.json` | Golden queries per dataset (basic/hard tiers, machine-checkable predicates) |
| `scripts/evaluate_search.py` / `diagnose_golden.py` | `just eval` / `just diagnose` |
| `scripts/evaluate_name_lookup.py` + `name_lookup_queries*.json` / `value_lookup_queries.pii.json` | `just eval-names` and its fixtures |
| `scripts/build_lookup_fixture.py` | `just lookup-fixture` |
| `scripts/store_fingerprint.py` | `just store-fingerprint` |
| `scripts/catalog_snapshot.py` | `just catalog-snapshot` |
| `scripts/ensure_payload_index.py` | One-off Qdrant payload-index migration (`just ensure-payload-index`) |
| `marimo/catalog.py` + `catalog_snapshot*.json` | Feature-catalog notebook and its committed snapshots |
| `marimo/persona_panel.py` + `panel_example.jsonl` | Persona-panel app and its bundled sample |
| `emulator/compose.yaml` | Standalone Qdrant / Elasticsearch / Neo4j stack (vendored minimal subset) |
| `docs/architecture.md` | System architecture |
| `docs/storage-footprint.md` | Disk needs and how to reclaim space |
| `docs/adr/` | Architecture Decision Records |
| `docs/research/` | Dated measurement records |

## Troubleshooting

- Check that `docker compose up -d` (in `emulator/`) finished and that ports 6333, 9200, and 7474 answer (`docker compose ps` shows health).
- If a Hugging Face dataset is gated, pass `--token` to `download-dataset`.
- If `index` says a file "looks like the '<name>' dataset", pass `--dataset-profile <name>`.
