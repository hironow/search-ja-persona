# Intent

**Last updated:** 2026-08-27
**Requester:** hironow
**Status:** Approved — 2026-08-27 に requester がセッション内で承認（初稿は AI が README / git 履歴から起草）
**Work unit:** search-ja-persona — CLI tooling for indexing and searching the Nemotron Personas Japan dataset with local emulators

## Goal
[Nemotron Personas Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan) の全量（1,000,000 件）をローカル emulator 群 — Qdrant (vector), Elasticsearch (keyword), Neo4j (persona graph context) — にインデックスし、日本語で高品質に検索・活用（ペルソナパネル調査を含む）できる自己完結環境を維持する。

## Success Criteria
- Unit test suite under `tests/` passes (`just test`).
- Integration tests against running emulators pass (`just integration`, `tests/test_integration_emulators.py`).
- The QA flow works end-to-end on the locally generated 1k sample (`just qa-sample`; git-ignored, not checked in): `just qa` (= `qa-index` + `qa-search`) returns results for the default Japanese query.
- 検索品質バー（`just eval --check-thresholds` で機械強制）: golden mean precision@5（basic tier）**≥ 0.85** かつ self-retrieval recall@1 **≥ 0.90** かつ recall@10 **≥ 0.99** を維持する。recall@1 バーは 2026-08-27 に requester 裁定で 0.99 から改定 — 人名除外埋め込み（匿名化ベクトルは同質ペルソナを意図的に等価へ寄せるため、名前入り本人要約での 1 位は保証対象外、top-10 内は保証）。ベースライン（2026-08-27 golden 保守後）: basic 0.917 / recall 0.92 / 1.00 — `docs/research/2026-08-27-golden-maintenance.md` 参照。

## Scope
### In scope
- CLI for download, index, search, and clearing emulators (`search_ja_persona/cli.py`).
- Indexing pipeline (`PersonaRepository` → `PersonaIndexer`) and search orchestration with hit fusion (`PersonaSearchService`).
- Embedding backends: hashed n-gram, SentenceTransformers, fastembed (`search_ja_persona/embeddings.py`) — 推奨プリセットは `ruri-v3-310m`（query/document プレフィックスの非対称埋め込み機構を含む）。埋め込み入力からは人名を除去する（strip-person-names-v1、保存テキストは原文のまま）。
- Full-corpus operation: `just full-index`（シャード単位・冪等）による全量インデックスの再現。
- Search-quality benchmark: `just eval`（golden precision@k + self-retrieval recall@k）。
- marimo notebooks: feature catalog（`just notebook`）と persona panel（`just panel` / `panel-app` — 上位 M 件へのマルチモーダル質問と JSONL 出力）。
- Architecture documentation in `docs/architecture.md` and ADRs in `docs/adr/`.

### Out of scope (Non-goals)
- The full emulator-set kit: this repo vendors only a minimal three-service subset (`emulator/compose.yaml`, ADR 0001 — Qdrant/Elasticsearch/Neo4j); the canonical multi-emulator kit lives upstream at `github.com/hironow/emulator-set`.

## Constraints
- Python 3.12+; dependency management via `uv`; tool versions via `mise.toml` (README / repo files).
- Local emulators (Qdrant, Elasticsearch, Neo4j) must be running before indexing/searching (README prerequisites).

## Open Questions
- [x] requester による本ドラフトのレビュー — 2026-08-27 承認済み。
- [x] Search relevance targets — `just eval` のしきい値（precision@5 ≥ 0.85 / recall@1 ≥ 0.99）として Success Criteria に確定。
- [x] CI: resolved — `.github/workflows/ci.yaml` runs prek hooks + unit tests on ubuntu, plus a windows-latest unit-test leg.
- [x] Intended audience / downstream use — 第一用途はローカルでのペルソナ探索と、ペルソナなりきりパネル調査（JSONL 出力を下流分析に供する）。
