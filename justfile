# https://just.systems

# Use bash for consistent behavior
set shell := ["bash", "-cu"]

default: help

help:
    @just --list --unsorted

# Aggregate local gate (mirrors what git hooks + CI enforce today:
# prek hooks and the unit suite). mypy/semgrep join here when the full
# agent baseline lands (tracked in docs/handover.md).
check: pre-commit test

format:
    uv run --frozen ruff format .

lint:
    uv run --frozen ruff check --fix .

# Install prek-managed git hooks once per clone
install-hooks:
    mise exec -- prek install

# Run every prek hook against all files (matches what git invokes on commit)
pre-commit:
    mise exec -- prek run --all-files

# --frozen everywhere: a bare `uv run` re-resolves under machine-local uv
# config and rewrites uv.lock as a side effect; the lock only changes via
# an explicit `uv lock`.
test:
    UV_CACHE_DIR=.uv-cache uv run --frozen pytest

integration:
    UV_CACHE_DIR=.uv-cache uv run --frozen pytest tests/test_integration_emulators.py -m integration

qa-clear:
    uv run --frozen python -m search_ja_persona.cli clear-emulators

qa-sample limit="1000":
    uv run --frozen python -m scripts.generate_qa_sample --limit "{{limit}}"

qa-index embedder="mini-lm" persona_fields="all":
    uv run --frozen python -m search_ja_persona.cli index \
        --dataset qa_samples/qa_sample.parquet \
        --batch-size 64 \
        --limit 1000 \
        --embedder "{{embedder}}" \
        --persona-fields "{{persona_fields}}"

qa-search query="高齢者介護の経験豊富なマネージャー" limit="3" format="table" embedder="mini-lm" persona_fields="all":
    uv run --frozen python -m search_ja_persona.cli search \
        --query "{{query}}" \
        --limit "{{limit}}" \
        --format "{{format}}" \
        --embedder "{{embedder}}" \
        --persona-fields "{{persona_fields}}" \
        --verbose

qa: qa-index qa-search

# Search-quality benchmark: golden-query precision@k + self-retrieval
# recall@k against the live index (report lands in outputs/). Extra flags
# pass through, e.g. `just eval --check-thresholds --rrf-weights 2,1`
eval *flags:
    uv run --frozen python -m scripts.evaluate_search {{flags}}

# Golden-set diagnostic: score each predicate against fused / keyword-only /
# random rankings to verify the eval measures retrieval, not the predicate
diagnose:
    uv run --frozen python -m scripts.diagnose_golden

# One-off migration: create the prefecture keyword payload index on an
# existing Qdrant collection (idempotent; new collections get it at index time)
ensure-payload-index qdrant_host="127.0.0.1" qdrant_port="6333" qdrant_collection="personas":
    uv run --frozen python -m scripts.ensure_payload_index \
        --qdrant-host "{{qdrant_host}}" \
        --qdrant-port "{{qdrant_port}}" \
        --qdrant-collection "{{qdrant_collection}}"

# Index the full corpus one shard at a time. Each shard is a checkpoint:
# uuid-keyed upserts make reruns idempotent, so on failure rerun from the
# failed shard only (or rerun the whole recipe; completed shards just
# overwrite themselves).
full-index embedder="ruri-v3-310m" batch_size="512":
    for shard in datasets/Nemotron-Personas-Japan/data/train-*.parquet; do \
        echo "=== indexing ${shard}"; \
        uv run --frozen python -m search_ja_persona.cli index \
            --dataset "${shard}" \
            --batch-size "{{batch_size}}" \
            --embedder "{{embedder}}" \
            --persona-fields all || exit 1; \
    done

# Open the feature-catalog marimo notebook (pulls marimo[sql] on demand)
notebook:
    uv run --frozen --with "marimo[sql]" marimo edit marimo/catalog.py

# Open the persona-panel notebook (top-M personas answer a multimodal input
# via local Ollama; answers land in outputs/*.jsonl)
panel:
    uv run --frozen --with "marimo[sql]" marimo edit marimo/persona_panel.py

# Persona panel as a read-mode app (no code shown, forms only)
panel-app:
    uv run --frozen --with "marimo[sql]" marimo run marimo/persona_panel.py

# Headless-run the catalog notebook and export static HTML (validation/preview)
notebook-export:
    uv run --frozen --with "marimo[sql]" marimo export html marimo/catalog.py -o marimo/catalog.html
