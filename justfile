set shell := ["bash", "-lc"]

default: help

help:
    @just --list --unsorted

format:
    uv run ruff format .

lint:
    uv run ruff check .

test:
    UV_CACHE_DIR=.uv-cache uv run pytest

integration:
    UV_CACHE_DIR=.uv-cache uv run pytest tests/test_integration_emulators.py -m integration

qa-index:
    uv run python -m search_ja_persona.cli index \
        --dataset qa_samples/qa_sample.parquet \
        --batch-size 64 \
        --limit 100

qa-search query="高齢者介護の経験豊富なマネージャー" limit="3" format="table":
    uv run python -m search_ja_persona.cli search \
        --query {{query}} \
        --limit {{limit}} \
        --format {{format}}

qa: qa-index qa-search
