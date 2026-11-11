# Storage Footprint

This document describes the disk capacity required to run search-ja-persona,
from the lightweight QA-sample path to a full 1M-persona corpus. Use it to
decide whether a machine has enough headroom before indexing, and which parts
of the on-disk state are safe to reclaim.

Numbers labelled **measured** were captured on a development Mac; numbers
labelled **approx** are public image/model sizes that vary by version and
platform. Emulator data-volume figures are estimates derived from the corpus
size and the chosen embedder dimension.

## TL;DR Budgets

| Scenario | Embedder | New disk needed | Notes |
|----------|----------|-----------------|-------|
| QA sample (1k rows) | `mini-lm` (384d) | **~5 GB** | Smallest end-to-end run; full corpus not required |
| Full corpus (1M rows) | `mini-lm` (384d) | **~16–21 GB** | Source parquet (two copies) + emulator volumes dominate |
| Full corpus (1M rows) | `e5-large` (1024d) | **~22–29 GB** | Larger vectors inflate Qdrant + model cache |
| japan + finance + pii, full (2.1M records) | `ruri-v3-310m` (768d) | **~37 GB** measured | ~27 GB emulator volumes + ~6.3 GB source parquet (two copies) + images |

"New disk needed" is on top of the Python virtualenv (~1 GB) and excludes the
redundant Hugging Face arrow cache, which can be deleted (see
[Reclaiming Space](#reclaiming-space)).

## Where the Bytes Live

```
 full-corpus run (mini-lm), approx GB
 |
 +-- venv (torch, pyarrow, ...)  ~1.0
 +-- source parquet (8 shards)   ~1.6
 +-- LFS object store (.git)     ~1.6
 +-- model cache (mini-lm)       ~0.1
 +-- docker images (3)           ~2.3
 +-- docker vm runtime           ~1-2
 +-- Qdrant volume               ~3-5
 +-- Elasticsearch volume        ~3-5
 +-- Neo4j volume                ~1.5-2.5
```

Legend:
- `venv`: Python 仮想環境 / 依存ライブラリ
- `source parquet`: 元データセットの parquet シャード
- `LFS object store`: Git LFS が `.git/modules/` に持つ parquet の複製
- `model cache`: 埋め込みモデルのローカルキャッシュ
- `docker images`: エミュレータのコンテナイメージ
- `docker vm runtime`: Docker VM のオーバーレイ・実行時領域
- `volume`: 各エミュレータがインデックスを保存する永続領域

## Component Breakdown

### Python environment (measured)

| Item | Size |
|------|------|
| `.venv` total | ~958 MB |
| — `torch` (CPU/MPS build) | 362 MB |
| — `pyarrow` | 108 MB |
| — `scipy` | 84 MB |
| — `onnxruntime` | 63 MB |
| — `transformers` | 58 MB |
| — `pandas` | 48 MB |
| — `sklearn` / `numpy` | 34 MB / 24 MB |
| `.uv-cache` (prunable) | ~55 MB |

The environment is CPU-only on macOS, so there is no CUDA bloat.

### Dataset (measured)

| Item | Size | Needed for |
|------|------|-----------|
| `qa_samples/qa_sample.parquet` | ~1.1 MB | QA path (generated via `just qa-sample`; git-ignored, not checked in) |
| `qa_samples/qa_sample.finance.parquet` / `qa_sample.pii.parquet` | ~2.2 MB / ~0.8 MB | finance / pii QA path (`just profile=<finance\|pii> qa-sample`) |
| `datasets/Nemotron-Personas-Japan/data/*.parquet` | ~1.6 GB (8 × 206 MB) | Full-corpus indexing |
| `.git/modules/datasets/Nemotron-Personas-Japan/lfs/objects` | ~1.6 GB | Git LFS object store behind the shards above (not hard-linked) |
| `.hf-cache/nvidia___nemotron-personas-japan` (arrow) | ~3.3 GB | `download-dataset` / QA-sample regeneration only |
| `datasets/Nemotron-Personas-Finance-Japan/data/*.parquet` | ~1.4 GB (8 × 174 MB) | finance indexing (`just profile=finance full-index`) |
| `.git/modules/datasets/Nemotron-Personas-Finance-Japan/lfs/objects` | ~1.4 GB | Git LFS object store behind the Finance shards and card images |
| `datasets/Nemotron-PII-Japan/data.parquet` | ~76 MB | pii indexing (`just profile=pii full-index`) |
| `.git/modules/datasets/Nemotron-PII-Japan/lfs/objects` | ~76 MB | Git LFS object store behind `data.parquet` |

The dataset is effectively stored **three times**: as compressed parquet shards
under `datasets/`, as the identical Git LFS objects under `.git/modules/`
(git-lfs keeps both a checked-out copy and its object store), and as an
expanded arrow cache under `.hf-cache/`. Indexing reads the parquet shards; the
arrow cache is only consumed by `scripts/generate_qa_sample.py` and the
`download-dataset` command.

### Embedding models (approx, downloaded on first use)

Sentence-transformer / FastEmbed weights are cached under `~/.cache/huggingface`
(and `~/.cache/torch`), **outside the repo**. Only the preset you actually use
is fetched.

| Preset | Type | Approx download |
|--------|------|-----------------|
| `hashed` | pure Python | 0 (no model) |
| `mini-lm` | SentenceTransformers | ~90 MB |
| `mpnet` | SentenceTransformers | ~420 MB |
| `e5-small` | SentenceTransformers | ~470 MB |
| `e5-large` | SentenceTransformers | ~2.2 GB |
| `fast-e5-small` | FastEmbed (ONNX) | ~470 MB |
| `fast-e5-large` | FastEmbed (ONNX) | ~2.2 GB |
| all presets combined | — | ~6 GB |

### Emulator container images (approx)

The three backends run as containers (Qdrant, Elasticsearch, Neo4j). Images are
pulled once and shared across runs.

| Image | Approx size |
|-------|-------------|
| `qdrant/qdrant` | 150–250 MB |
| `elasticsearch` 8.x | 1.3–1.7 GB |
| `neo4j` 5.x | 500–600 MB |
| **subtotal** | **~2.0–2.5 GB** |

Budget an additional ~1–2 GB for the Docker VM overlay and runtime scratch.

### Emulator data volumes after indexing (estimated)

These grow with the number of indexed personas and the vector dimension.

| Backend | QA sample (1k) | Full corpus (1M, `mini-lm` 384d) | What drives it |
|---------|----------------|----------------------------------|----------------|
| Qdrant | < 30 MB | ~3–5 GB | vectors (1M × dim × 4B) + HNSW graph + text payload |
| Elasticsearch | < 40 MB | ~3–5 GB | `_source` + inverted index over `text` and 6 persona fields |
| Neo4j | < 20 MB | ~1.5–2.5 GB | 1M `Persona` nodes (stores the `persona` field) + prefecture/region graph |
| **subtotal** | **< 100 MB** | **~8–12 GB** | |

Measured on Windows 11 / Docker Desktop with `ruri-v3-310m` (768d),
2026-11-11 (`docker system df -v`; Elasticsearch per index from
`_cat/indices`):

| Volume | japan only (1M) | japan + finance (1M) + pii (100k) | Increase |
|--------|-----------------|-----------------------------------|----------|
| `emulator_qdrant_data` | 5.77 GB | 12.31 GB | +6.54 GB |
| `emulator_elasticsearch_data` | 5.68 GB | 12.40 GB | +6.72 GB (indexes: personas 5.41, finance_personas 6.20, pii_documents 0.21 GB) |
| `emulator_neo4j_data` | 2.54 GB | 2.67 GB | +0.13 GB |
| **subtotal** | **~14.0 GB** | **~27.4 GB** | **+13.4 GB** |

Qdrant and Neo4j report only per volume, so their increase covers finance
and pii together.

Switching to `e5-large` (1024d) roughly doubles the raw vector store
(1M × 1024 × 4B ≈ 4 GB in Qdrant alone), pushing the Qdrant volume to ~7–9 GB.

## Recommended Path for a Space-Constrained Mac

1. **Stay on the QA sample.** `just qa-index` + `just qa-search` exercise the
   entire pipeline against `qa_samples/qa_sample.parquet` (1k rows) with
   emulator volumes under ~100 MB.
2. **Use `mini-lm`** (90 MB, 384d) — good quality at the smallest model + vector
   footprint. Use `hashed` for zero model download when quality does not matter.
3. **Delete the redundant arrow cache** and, if you do not need the full corpus,
   the parquet shards too (see below).

This keeps the total end-to-end footprint (beyond the existing venv) to roughly
**5 GB**, most of which is the Docker images shared with any other project.

## Reclaiming Space

| Action | Reclaims | Safe when |
|--------|----------|-----------|
| `rm -rf .hf-cache` | ~3.3 GB | Parquet shards under `datasets/` are present (indexing does not use the arrow cache) |
| `git submodule deinit -f datasets/Nemotron-Personas-Japan`, then `rm -rf .git/modules/datasets/Nemotron-Personas-Japan/lfs/objects` | ~3.2 GB (shards + LFS objects) | Running the QA path only; restore with `git submodule update --init` + `git lfs pull` (see the README) |
| `git submodule deinit -f datasets/Nemotron-Personas-Finance-Japan`, then `rm -rf .git/modules/datasets/Nemotron-Personas-Finance-Japan/lfs/objects` | ~2.8 GB | The finance index is built and no reindex, eval self-retrieval, diagnose, or `lookup-fixture` run is planned (they read the shards) |
| `git submodule deinit -f datasets/Nemotron-PII-Japan`, then `rm -rf .git/modules/datasets/Nemotron-PII-Japan/lfs/objects` | ~150 MB | The pii index is built and no reindex, diagnose, or `lookup-fixture` run is planned |
| `uv cache prune` (or `rm -rf .uv-cache`) | ~55 MB | Anytime; repopulated on next `uv` run |
| `just profile=<finance\|pii> qa-clear` | ~6.5 / ~0.3 GB | Drops only that dataset's collection, index, and nodes; japan stays |
| `docker image prune` / drop emulator volumes | up to ~27 GB | Between runs; re-index to rebuild |

## Verifying Available Space

macOS reports the real free space on the data volume, not the sealed system
volume:

```bash
df -h /System/Volumes/Data      # true available space + capacity %
du -sh .venv .cache .hf-cache datasets .git   # this repo's largest consumers
```

A capacity at or above ~90% is the practical warning line on macOS even when
the absolute free figure looks large, because the OS reserves headroom for
snapshots and swap.
