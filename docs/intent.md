# Intent

**Last updated:** 2026-06-10
**Requester:** hironow
**Status:** DRAFT — AI が README / git 履歴から起草。requester 未確認
**Work unit:** search-ja-persona — CLI tooling for indexing and searching the Nemotron Personas Japan dataset with local emulators

## Goal
Provide self-contained tooling for indexing and searching the [Nemotron Personas Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) dataset (~1M rows) against local emulators — Qdrant (vector), Elasticsearch (keyword), Neo4j (persona graph context) — so developers can explore personas without reading the source first (per README).

## Success Criteria
- Unit test suite under `tests/` passes (`just test`).
- Integration tests against running emulators pass (`just integration`, `tests/test_integration_emulators.py`).
- The QA flow works end-to-end on the bundled 1k sample: `just qa` (= `qa-index` + `qa-search`) returns results for the default Japanese query.

## Scope
### In scope
- CLI for download, index, search, and clearing emulators (`search_ja_persona/cli.py`).
- Indexing pipeline (`PersonaRepository` → `PersonaIndexer`) and search orchestration with hit fusion (`PersonaSearchService`).
- Embedding backends: hashed n-gram, SentenceTransformers, fastembed (`search_ja_persona/embeddings.py`).
- Architecture documentation in `docs/architecture.md` and ADRs in `docs/adr/`.

### Out of scope (Non-goals)
- The emulator stack itself: the emulator submodule was removed and moved to `sets/emulator-set` (commit dc48659); this repo only consumes running emulators.

## Constraints
- Python 3.12+; dependency management via `uv`; tool versions via `mise.toml` (README / repo files).
- Local emulators (Qdrant, Elasticsearch, Neo4j) must be running before indexing/searching (README prerequisites).

## Open Questions
- [ ] requester による本ドラフトのレビュー
- [ ] Search relevance targets: no quality benchmarks or relevance criteria are defined in the repo — is "returns merged results" sufficient, or is there a quality bar?
- [ ] CI: there is no `.github/workflows/` in this repo — should tests run in CI?
- [ ] Intended audience / downstream use of the indexed persona data (exploration only, or feeding another system?).
