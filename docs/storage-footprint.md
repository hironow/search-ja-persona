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
| Full corpus (1M rows) | `mini-lm` (384d) | **~14–19 GB** | Source parquet + emulator volumes dominate |
| Full corpus (1M rows) | `e5-large` (1024d) | **~20–27 GB** | Larger vectors inflate Qdrant + model cache |

"New disk needed" is on top of the Python virtualenv (~1 GB) and excludes the
redundant Hugging Face arrow cache, which can be deleted (see
[Reclaiming Space](#reclaiming-space)).

## Where the Bytes Live

```
 full-corpus run (mini-lm), approx GB
 |
 +-- venv (torch, pyarrow, ...)  ~1.0
 +-- source parquet (8 shards)   ~1.6
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
| `qa_samples/qa_sample.parquet` | ~1.1 MB | QA path (bundled, always present) |
| `datasets/Nemotron-Personas-Japan/data/*.parquet` | ~1.6 GB (8 × 206 MB) | Full-corpus indexing |
| `.hf-cache/nvidia___nemotron-personas-japan` (arrow) | ~3.3 GB | `download-dataset` / QA-sample regeneration only |

The dataset is effectively stored **twice**: as compressed parquet shards under
`datasets/` and as an expanded arrow cache under `.hf-cache/`. Indexing reads
the parquet shards; the arrow cache is only consumed by
`scripts/generate_qa_sample.py` and the `download-dataset` command.

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
| Remove `datasets/Nemotron-Personas-Japan/data/*.parquet` | ~1.6 GB | Running the QA path only; re-fetch via the dataset submodule / `download-dataset` when needed |
| `uv cache prune` (or `rm -rf .uv-cache`) | ~55 MB | Anytime; repopulated on next `uv` run |
| `git gc` on `.git` (~1.8 GB) | varies | Anytime; large historical blobs limit how much is reclaimable |
| `docker image prune` / drop emulator volumes | up to ~12 GB | Between runs; re-index to rebuild |

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
