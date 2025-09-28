# https://just.systems

# Use bash for consistent behavior
set shell := ["bash", "-cu"]

default: help

help:
    @just --list --unsorted

format:
    uv run ruff format .

lint:
    uv run ruff check --fix .

test:
    UV_CACHE_DIR=.uv-cache uv run pytest

integration:
    UV_CACHE_DIR=.uv-cache uv run pytest tests/test_integration_emulators.py -m integration

qa-clear:
    uv run python -m search_ja_persona.cli clear-emulators

qa-sample limit="1000":
    uv run python -m scripts.generate_qa_sample --limit {{limit}}

qa-index embedder="mini-lm" persona_fields="all":
    uv run python -m search_ja_persona.cli index \
        --dataset qa_samples/qa_sample.parquet \
        --batch-size 64 \
        --limit 1000 \
        --embedder {{embedder}} \
        --persona-fields {{persona_fields}}

qa-search query="高齢者介護の経験豊富なマネージャー" limit="3" format="table" embedder="mini-lm" persona_fields="all":
    uv run python -m search_ja_persona.cli search \
        --query {{query}} \
        --limit {{limit}} \
        --format {{format}} \
        --embedder {{embedder}} \
        --persona-fields {{persona_fields}} \
        --verbose

qa: qa-index qa-search
