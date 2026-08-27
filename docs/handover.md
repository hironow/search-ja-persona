# Handover

**Last updated:** 2026-08-27 (JST)
**Updated by:** claude (AI session, Windows workstation)

## Current State
The full corpus is indexed and verified. PRs #35–#38 are squash-merged on
`main`:

- **#35 Windows enablement + security**: Windows 11 (Docker Desktop/WSL2) is a
  first-class dev environment; IPv4 loopback defaults (localhost stalled
  ~250ms/request on Windows); UTF-8 I/O; all 5 Dependabot alerts cleared
  (transformers 5.16.1 / protobuf 7.36.0 / pygments 2.21.0); multi-platform
  lock guarantees (`required-environments`, `exclude-newer`); CUDA torch
  (2.13.0+cu130) on win32 via the explicit pytorch.org index; batched
  embedding (`embed_many`) and batched Neo4j UNWIND merges; backend
  body-level errors now raise instead of silently dropping documents;
  `windows-latest` CI leg.
- **#36 full-corpus prerequisites**: Neo4j uniqueness constraints ensured at
  ingest start (unindexed MERGE is O(n²)); emulator heaps sized for 1M rows;
  asymmetric retrieval API (`embed_query`/`embed_documents`) with per-preset
  prefixes; **`ruri-v3-310m` adopted** per the 2026-08 benchmark
  (multilingual-e5 truncates 22–23% of this corpus at 512 tokens).
- **#37 execution kit**: batched Neo4j reset (`delete_all_personas`),
  metadata persisted at reset time, `just full-index` (shard-by-shard,
  idempotent uuid-keyed upserts = per-shard checkpoints), the benchmark
  record in `docs/research/2026-08-27-embedding-model-comparison.md`, and
  emulator volume-persistence notes in the README.
- **#38 marimo catalog**: rebuilt around the real index — pipeline-checks
  section (health + three-store count agreement + metadata), a committed
  snapshot of real full-corpus results (`marimo/catalog_snapshot.json`),
  and button-triggered live search using the metadata-recorded embedder.

**Index state (verified 2026-08-27):** Qdrant / Elasticsearch / Neo4j all
hold exactly **1,000,000** personas (ground truth: 1,000,000 distinct uuids,
0 nulls); 768-dim ruri-v3-310m vectors; random spot-checks consistent across
parquet and all three stores; fused search answers in ~45–76ms. Unit suite:
51 passed / 1 skipped; CI (ubuntu + windows) green; open Dependabot alerts: 0.

## In Progress
Nothing in flight.

## Next Actions
1. Search relevance quality bar is still undefined (intent.md open
   question) — only eyeball checks exist. A golden-query set or
   JMTEB-style eval over the indexed corpus would make quality regressions
   visible.
2. intent.md remains DRAFT: requester review and the "intended audience /
   downstream use" open question are still unanswered.
3. Housekeeping (optional): regenerate `marimo/catalog_snapshot.json` after
   any reindex; Docker Desktop's WSL2 vhdx grows with the ~15GB of volume
   data and does not shrink automatically.

## Known Risks / Blockers
- On this machine `uv run`/`uv lock` **without** `--frozen`/`UV_NO_CONFIG=1`
  rewrites `uv.lock` (machine-local `~/.config/uv/uv.toml` sets a
  flatt-mirror index + `exclude-newer`). All justfile recipes now pass
  `--frozen`; keep new uv invocations frozen too, and use `UV_NO_CONFIG=1`
  for lock changes meant to be committed.
- Emulators must be running (`cd emulator && docker compose up -d`); Docker
  Desktop's engine needs to be started first on Windows. Volumes persist
  across stop/down/reboot — only `down -v`, volume prune, or a Docker
  Desktop purge destroy them (see README "Emulator Data Persistence").
- `scripts/generate_qa_sample.py` still requires the HF cache
  (`download-dataset`); the current `qa_samples/qa_sample.parquet` was cut
  directly from shard 0 of the submodule (all 8 LFS shards are pulled,
  ~1.7GB).

## Context the Next Actor Needs
- Emulators are vendored at `emulator/compose.yaml`; ports/auth match
  `ApplicationConfig`. The canonical full kit lives at `~/dotfiles/emulator`
  (upstream `github.com/hironow/emulator-set`).
- The **full corpus is indexed and persistent** — day-to-day work does not
  need reindexing. A full rebuild is `just full-index` (~50min on the RTX
  4090); a failed run resumes by rerunning the failed shard only.
- Switching embedder presets triggers a confirmed reset (dimensions change);
  the reset deletes Neo4j personas in batched transactions and persists the
  new metadata immediately.
- Tooling: `uv` (Python deps), `mise` (tool versions), `just` (tasks; recipes
  run under bash — on Windows that means Git Bash on PATH).
- Score semantics differ by backend: Qdrant cosine similarity vs. mapped
  Elasticsearch `_score`.

## Relevant Files and Commands
- `search_ja_persona/cli.py` — CLI entry point (index / search /
  download-dataset / clear-emulators)
- `docs/research/2026-08-27-embedding-model-comparison.md` — why
  ruri-v3-310m; measured throughput/tokenizer/quality data
- `docs/architecture.md` — system architecture (embedder presets + prefixes)
- `just full-index` — shard-by-shard full-corpus ingest
- `just qa` — 1k smoke (mini-lm); `just test` / `just integration`
- `just notebook` — marimo catalog (`marimo/catalog.py` +
  `marimo/catalog_snapshot.json`)
