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
- **#40 persona panel**: read-mode survey app (`just panel` / `panel-app`) —
  top-M personas from the fused search answer a text/image input in
  character via local Ollama (installed-model dropdown, default
  huihui_ai/Qwen3.8-abliterated), JSONL outputs, run-history preview, and
  a committed all-huihui M=30 sample (`marimo/panel_example.jsonl`).
- **#41 uuid fusion fix**: Qdrant's hyphenated point ids never matched the
  dataset's hyphen-less uuids, so keyword-dedup was dead and Neo4j context
  for vector hits had been silently empty since the feature shipped; ids
  are now normalized at the fusion boundary (graph enrichment verified
  live).
- **#42/#43 search-quality benchmark + approved intent**: `just eval`
  (golden precision@5 + self-retrieval recall@k) landed and the requester
  ratified the quality bar in intent.md (basic golden mean ≥ 0.85,
  recall@1 ≥ 0.99) — see
  `docs/research/2026-08-27-search-quality-baseline.md`.
- **#44 golden eval hardening**: the golden
  set graded too kindly (one predicate matched 98% of random personas).
  Now tiered — 12 frozen "basic" queries (the bar metric) + 12 "hard"
  (multi-aspect `text_all`, geo+theme, paraphrase), runtime-validated
  (`load_golden_queries`), with `just diagnose` scoring every predicate
  against fused / keyword-only / random rankings. Baseline: basic 0.900
  (unchanged) | hard 0.433 | random base 0.047 | recall 1.00/1.00.
  Headline finding: **person-name pollution** — vector search chases
  names (福岡姓 → 福岡県クエリ, 「温泉 正次」氏 → 温泉クエリ 1 位) in
  6 of 12 hard queries. Full analysis:
  `docs/research/2026-08-27-golden-set-hardening.md`.

**Index state (verified 2026-08-27):** Qdrant / Elasticsearch / Neo4j all
hold exactly **1,000,000** personas (ground truth: 1,000,000 distinct uuids,
0 nulls); 768-dim ruri-v3-310m vectors; random spot-checks consistent across
parquet and all three stores; fused search answers in ~45–76ms. Unit suite:
77 passed / 1 skipped; CI (ubuntu + windows) green; open Dependabot alerts: 0.

- **#45 prefecture filter**: residency is
  now an explicit filter — `--prefecture` (validated against the 47
  official names) drives a Qdrant payload filter + ES term filter; the
  keyword payload index is created at index time for new collections and
  was backfilled onto the live 1M via `just ensure-payload-index`
  (idempotent migration; search path stays read-only). Benchmark schema
  v3 adds a paired filtered section: **filtered geo mean 1.000 (n=4)**
  vs 0.65 unfiltered, with tier means untouched (basic 0.900 / hard
  0.433). See `docs/research/2026-08-27-prefecture-filter-results.md`.

## In Progress
- PR for `fix/missing-index-metadata` (this branch) — merge on green per
  the session's per-PR agreement. `.cache/index_metadata.json` had gone
  missing (cause untraceable; the file is CWD-relative and unlinked by
  reset/clear paths). Search without `--embedder` used to fall back to
  hashed-256 and die with an opaque Qdrant dimension error; worse, an
  index rerun reconstructed partial metadata that always failed
  `_should_reset`'s schema check, putting a destructive reset prompt in
  front of the healthy 1M collection. Both paths now fail closed, and a
  read-only `repair-metadata` subcommand re-records metadata after
  verifying collection dimension + stored persona fields. Live metadata
  restored via the new command (1,000,000 points, dim 768); bare CLI
  search works again.
- **Neo4j collateral found and healed**: the pre-existing integration
  test `test_index_and_search_with_emulators` indexes the first 5 REAL
  dataset rows into isolated Qdrant/ES resources but the SHARED Neo4j
  database, and its cleanup DETACH-DELETEs those uuids — every
  `just integration` run silently removed 5 real personas from the live
  graph (found at 999,995). Restored to exactly 1,000,000 via idempotent
  reindex of shard-0 rows 0-4. The test still needs fixing (synthetic
  uuids, or skip the Neo4j cleanup for real-uuid rows).

## Next Actions
1. Human decisions queued by the hardening research note: (a) new hard-tier
   bar (proposed: hard mean precision@5 ≥ 0.35, baseline 0.433) as an
   intent.md addition; (b) whether to fix the 3 degenerate basic predicates
   (和食 random 0.980 / 茶道 0.495 / 登山 0.380) at the cost of re-basing
   the bar; (c) ordering of the improvement candidates below.
2. Improvement candidates (each its own work unit, evidence in
   `docs/research/2026-08-27-golden-set-hardening.md`): ~~prefecture
   payload filter~~ (done — geo class solved, filtered mean 1.000) →
   fusion redesign (keyword leg is effectively unused; BM25 alone beats
   fused on the predicate metrics) → name-token exclusion at embed time
   (needs ~50min reindex) → pooled human qrels.
3. Fix the integration test's Neo4j collateral (see In Progress) — one
   small `fix` PR.
4. Metadata robustness candidates (structural, separate work unit, from
   the plan review): identity-keyed metadata (record/verify Qdrant
   endpoint+collection, per-identity files instead of one global
   CWD-relative file), atomic writes, and clear-emulators unlinking the
   global metadata even when pointed at a non-default collection.
5. Tooling candidate (from the plan review): this repo has no `just check`
   aggregate gate and no mypy/semgrep; the enforced gate today is
   `just pre-commit` + `just test` (mirrors CI). Adding the full AGENTS.md
   gate is a separate structural PR.
6. Housekeeping (optional): regenerate `marimo/catalog_snapshot.json` after
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
- `.cache/index_metadata.json` is CWD-relative and can go missing; since
  the fail-closed fix, `search`/`index` refuse to guess and the recovery
  is `repair-metadata --embedder <preset>` (read-only, verifies the
  collection dimension and stored persona fields). The live file was
  restored on 2026-08-27.
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
