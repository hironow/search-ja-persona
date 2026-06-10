# Handover

**Last updated:** 2026-06-10 (JST)
**Updated by:** claude (AI draft from git history — review before trusting)

## Current State
The full pipeline (parquet streaming → indexing into Qdrant/Elasticsearch/Neo4j → fused search) is implemented and documented in the README and `docs/architecture.md`, with unit tests and emulator integration tests under `tests/`. The last meaningful commit is `5af804a` "docs: add decision queue for human-review items (#4)" on 2026-06-10; shortly before that, the emulator submodule was removed (moved to `sets/emulator-set`, commit dc48659) and `mise` was added (a182caf).

## In Progress
不明 (git 履歴からは判別できず) — recent commits are docs/tooling housekeeping, no feature work in flight.

## Next Actions
1. requester による docs/intent.md ドラフトのレビューと確定
2. Decide whether to add CI (no `.github/workflows/` exists; tests currently run only locally).
3. Watch `docs/decision-queue.md` for human-review items (currently "(none yet)").

## Known Risks / Blockers
- Indexing and search require local emulators to be running; without them `just integration` and the QA flow fail (README prerequisites).
- Full-corpus indexing handles ~1,000,000 rows; batch size must match available memory (README).

## Context the Next Actor Needs
- Tooling: `uv` for Python deps, `mise` for tool versions, `just` for tasks.
- Emulators now live in a separate checkout (`sets/emulator-set` per commit dc48659); start them there before using this CLI.
- A 1k-row QA sample ships at `qa_samples/qa_sample.parquet`; regenerate with `uv run python -m scripts.generate_qa_sample --limit 1000`.
- Score semantics differ by backend: Qdrant cosine similarity vs. mapped Elasticsearch `_score` (README "Score Semantics").

## Relevant Files and Commands
- `search_ja_persona/cli.py` — CLI entry point (index / search / download-dataset / clear)
- `docs/architecture.md` — current system architecture
- `docs/decision-queue.md` — queue of items needing human decisions
- `just test` / `just integration` — unit and emulator integration tests
- `just qa` — index + search the bundled 1k sample (quick smoke check)
