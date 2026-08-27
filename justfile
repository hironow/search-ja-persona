# https://just.systems

# Use bash for consistent behavior
set shell := ["bash", "-cu"]

default: help

help:
    @just --list --unsorted

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

# Open the feature-catalog marimo notebook (pulls marimo[sql] on demand)
notebook:
    uv run --frozen --with "marimo[sql]" marimo edit marimo/catalog.py

# Headless-run the catalog notebook and export static HTML (validation/preview)
notebook-export:
    uv run --frozen --with "marimo[sql]" marimo export html marimo/catalog.py -o marimo/catalog.html
