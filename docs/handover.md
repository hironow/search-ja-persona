# Handover

**Last updated:** 2026-07-27 (JST)
**Updated by:** claude (AI session)

## Current State
The full pipeline (parquet streaming → indexing into Qdrant/Elasticsearch/Neo4j → fused search) is implemented, documented, and now runnable end to end. PR #6 (`f8a972d`) vendored a minimal, standalone `emulator/compose.yaml` (Qdrant/Elasticsearch/Neo4j) into the repo and was verified by a full `just qa` smoke test — 1,000 personas indexed into all three backends, combined search returns results. `main` is at `f8a972d`.

## In Progress
None in flight. The emulator-vendoring + docs-refresh work just merged (PR #6). `docs/adr/0001` records the vendoring decision; `docs/storage-footprint.md` documents disk requirements.

## Next Actions
1. Human to resolve the `docs/decision-queue.md` item (2026-07-27): the `docs/intent.md` emulator non-goal is now contradicted by the vendored stack — revise intent.md accordingly (agents must not edit intent.md).
2. Decide whether to add CI (still no `.github/workflows/`; tests and the QA smoke test run only locally).
3. Restore the full corpus only when needed (see Context) — the repo currently runs on the bundled QA sample.

## Known Risks / Blockers
- Indexing/search require the emulators running: `cd emulator && docker compose up -d`.
- Full-corpus indexing needs the dataset submodule checked out (~1M rows); batch size must match available memory.

## Context the Next Actor Needs
- **Emulators are now vendored** at `emulator/compose.yaml` (`docker compose up -d`); ports/auth match `ApplicationConfig`. The canonical full kit lives at `~/dotfiles/emulator` (upstream `github.com/hironow/emulator-set`); the earlier "moved to `sets/emulator-set`" note (commit `dc48659`) is inaccurate — that path does not exist.
- The repo runs **QA-sample-only locally**: the full parquet shards under `datasets/Nemotron-Personas-Japan/data/` were removed to save disk, so the submodule shows dirty and full-corpus `index` fails until restored via `git -C datasets/Nemotron-Personas-Japan lfs pull` (or `download-dataset`).
- Tooling: `uv` (Python deps), `mise` (tool versions), `just` (tasks).
- Score semantics differ by backend: Qdrant cosine similarity vs. mapped Elasticsearch `_score`.

## Relevant Files and Commands
- `search_ja_persona/cli.py` — CLI entry point (index / search / download-dataset / clear)
- `emulator/compose.yaml` — standalone local emulator stack (`docker compose up -d`)
- `docs/architecture.md` — current system architecture; `docs/adr/0001-vendor-emulator-compose.md` — emulator vendoring decision
- `docs/storage-footprint.md` — disk requirements and space-reclaim guide
- `docs/decision-queue.md` — items needing human decisions
- `just qa` — index + search the bundled 1k sample (quick smoke check)
- `just test` / `just integration` — unit and emulator integration tests
