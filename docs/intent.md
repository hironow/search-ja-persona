# Intent

**Last updated:** 2026-08-27
**Requester:** hironow
**Status:** DRAFT — AI が README / git 履歴から起草。2026-08-27 に decision-queue の2件（emulator 非目標・sample 表現）を requester 指示で改訂
**Work unit:** search-ja-persona — CLI tooling for indexing and searching the Nemotron Personas Japan dataset with local emulators

## Goal
Provide self-contained tooling for indexing and searching the [Nemotron Personas Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) dataset (~1M rows) against local emulators — Qdrant (vector), Elasticsearch (keyword), Neo4j (persona graph context) — so developers can explore personas without reading the source first (per README).

## Success Criteria
- Unit test suite under `tests/` passes (`just test`).
- Integration tests against running emulators pass (`just integration`, `tests/test_integration_emulators.py`).
- The QA flow works end-to-end on the locally generated 1k sample (`just qa-sample`; git-ignored, not checked in): `just qa` (= `qa-index` + `qa-search`) returns results for the default Japanese query.

## Scope
### In scope
- CLI for download, index, search, and clearing emulators (`search_ja_persona/cli.py`).
- Indexing pipeline (`PersonaRepository` → `PersonaIndexer`) and search orchestration with hit fusion (`PersonaSearchService`).
- Embedding backends: hashed n-gram, SentenceTransformers, fastembed (`search_ja_persona/embeddings.py`).
- Architecture documentation in `docs/architecture.md` and ADRs in `docs/adr/`.

### Out of scope (Non-goals)
- The full emulator-set kit: this repo vendors only a minimal three-service subset (`emulator/compose.yaml`, ADR 0001 — Qdrant/Elasticsearch/Neo4j); the canonical multi-emulator kit lives upstream at `github.com/hironow/emulator-set`.

## Constraints
- Python 3.12+; dependency management via `uv`; tool versions via `mise.toml` (README / repo files).
- Local emulators (Qdrant, Elasticsearch, Neo4j) must be running before indexing/searching (README prerequisites).

## Open Questions
- [ ] requester による本ドラフトのレビュー
- [ ] Search relevance targets: no quality benchmarks or relevance criteria are defined in the repo — is "returns merged results" sufficient, or is there a quality bar?
- [x] CI: resolved — `.github/workflows/ci.yaml` runs prek hooks + unit tests on ubuntu, plus a windows-latest unit-test leg.
- [ ] Intended audience / downstream use of the indexed persona data (exploration only, or feeding another system?).
