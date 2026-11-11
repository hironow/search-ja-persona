# Handover

**Last updated:** 2026-11-11 (JST)
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

- **RRF fusion (branch `feat/rrf-fusion`)**: results are now ranked by
  weighted Reciprocal Rank Fusion (K=60, production 1:1) instead of
  vector-fills-then-truncate; fetch depth max(limit, min(3*limit, 30));
  Neo4j context only for the returned top-limit. Pre-registered A/B on
  the live 1M chose unweighted: basic 0.900→**0.950**, hard
  0.433→**0.450**, overall 0.700, recall 1.00/1.00, eval wall time
  11.3s→**9.0s**. Filtered geo reads 0.950 vs 1.000 — the changed hit is
  a documented predicate false negative (Okinawa sea-leisure persona
  phrased as 泳ぎ/潮風), disclosed as a deviation from the pre-registered
  constraint. Side products: `--check-thresholds` (machine-enforced
  intent bars) and `just check`. See
  `docs/research/2026-08-27-rrf-fusion-results.md`.

- **#46 metadata fail-closed + repair**: `.cache/index_metadata.json`
  went missing (cause untraceable); search silently fell back to
  hashed-256 and an index rerun walked into a destructive reset prompt.
  Both paths now fail closed and a read-only `repair-metadata`
  subcommand re-records metadata after verifying the collection
  dimension and stored persona fields. Live file restored.
- **#47 Neo4j test isolation**: the integration test used to index the
  first 5 REAL dataset rows and DETACH-DELETE them from the shared live
  graph on every run (found at 999,995; restored to 1,000,000 via
  idempotent reindex). Rows are now re-keyed with synthetic uuids.

- **#48 RRF fusion**: weighted Reciprocal Rank Fusion (K=60, 1:1)
  replaced vector-fills-then-truncate; `--check-thresholds` and
  `just check` landed alongside.
- **name exclusion (branch `feat/name-exclusion`)**: embedding inputs
  now strip person names (strip-person-names-v1; stored text unchanged);
  full 1M reindexed in place and proven migrated content-wise (40/40
  stratified cosine checks). basic 0.950→0.983, hard 0.450→**0.633**,
  vector-only leg 1.000/0.650 (first time above BM25), 6-query
  name-pollution class +0.30 mean. Self-retrieval recall@1 landed at
  0.92 (recall@10 1.00) — the requester ratified an amended bar
  (recall@1 ≥ 0.90 ∧ recall@10 ≥ 0.99, intent.md updated). Includes the
  keyword-leg tie-break, the metadata-vanishing culprit fix (the reset
  test deleted the real metadata file on every pytest run), and a
  regenerated catalog snapshot. See
  `docs/research/2026-08-27-name-exclusion-results.md`.

- **#49 name exclusion**: merged with the requester-ratified bar
  amendment (recall@1 ≥ 0.90 ∧ recall@10 ≥ 0.99).
- **golden maintenance (branch `fix/golden-maintenance`)**: Okinawa
  predicate gains 泳ぎ/潮風/浜 (filtered FN healed → 1.00), the
  degenerate washoku predicate rebuilt (random base 0.980 → 0.275,
  exposing a genuinely weak query: fused 0.20 / vector 0.80), sadou
  drops 着物. New baseline: basic 0.917 / hard 0.650 / filtered 0.950,
  `--check-thresholds` green. See
  `docs/research/2026-08-27-golden-maintenance.md`.

- **#51 ratified bars**: hard ≥ 0.55 and filtered geo ≥ 0.90 mechanized.
- **canary metrics (branch `feat/canary-metrics`)**: three silent-death
  canaries join the machine-enforced gate — 3-store count agreement,
  graph-context coverage ≥ 0.99, keyword-leg contribution > 0 (each
  motivated by a real incident found this session: the 999,995 Neo4j
  drift, the months-dead graph context of #41, and the pre-RRF dead
  keyword leg). Live baseline: counts 1M×3, coverage 1.00, kw-rate 1.00.

- **#53 name-lookup baseline**: `just eval-names` on a fixed stratified
  40-name set — recall@1 **0.725** / recall@10 0.900; misses are
  common-surname + short-given-name cases (CJK unigram BM25 limit).
- **#54 panel example regenerated**: the bundled
  `marimo/panel_example.jsonl` now reflects the current backend (RRF +
  prefecture filter + name-stripped vectors): text きのこ 23-7 and
  image きのこ 23-7 with brand-name verdicts — the old mode flip
  (name-based kinoko vs photo-based takenoko) disappeared with the new
  panel selection. `marimo/catalog_snapshot.json` was already
  regenerated in #49.

- **#101 finance + pii datasets**: `nvidia/Nemotron-Personas-Finance-Japan` (c73cc86) and
  `nvidia/Nemotron-PII-Japan` (a074898) are Git LFS submodules under
  `datasets/`, and both are **fully indexed in their own stores**:
  finance 1,000,000×3 (`finance_personas` / `FinancePersona`), pii
  100,000×3 (`pii_documents` / `PiiDocument`). Every store request goes
  through a dataset profile (`--dataset-profile`, `just profile=...`;
  ADR 0004), with guards against mixing datasets. intent.md was updated
  with the requester's approval (personal research). Baselines: finance
  golden basic 0.457 / hard 0.425 / recall 0.91 / 1.00; pii basic 0.900 /
  hard 0.880 / value lookup 0.875 / 1.00; canaries OK for all three. The
  japan index is proven unchanged (store fingerprints and eval identical
  before and after). Details:
  `docs/research/2026-11-11-finance-pii-baseline.md`.

## In Progress
Nothing in flight. The finance / pii work landed as PR #101 (squash
merge approved by the owner). The pre-indexing volume backup was deleted
after japan was verified unchanged. The emulators are stopped; their
volumes hold all three indexes (`cd emulator && docker compose up -d`).

## Next Actions
1. ~~Bar ratification~~ — done 2026-08-27: hard ≥ 0.55 and filtered geo
   ≥ 0.90 ratified and mechanized in check_thresholds + intent.md. All
   five quality bars are now machine-enforced.
2. Improvement candidates (each its own work unit, evidence in
   `docs/research/2026-08-27-golden-set-hardening.md`): ~~prefecture
   payload filter~~ (done) → ~~fusion redesign~~ (done — RRF adopted) →
   ~~name-token exclusion~~ (done — adopted with the amended bar) →
   pooled human qrels. Also queued: golden-set maintenance (add
   泳ぎ/潮風-class vocabulary to the Okinawa predicate, autopsy the two
   filtered-geo 0.80 rows, fix the 3 degenerate basic predicates,
   re-baseline) as one PR; name-lookup strengthening (name-only rank-1
   is 19/24 under CJK unigram BM25 — e.g. an extracted-name keyword
   field).
4. Metadata robustness candidates (structural, separate work unit, from
   the plan review): atomic writes and recording the Qdrant endpoint.
   (Per-dataset and per-collection metadata files, and clear-emulators
   no longer unlinking the default metadata for a custom collection, are
   done.)
5. Tooling candidate (from the plan review): this repo has no `just check`
   aggregate gate and no mypy/semgrep; the enforced gate today is
   `just pre-commit` + `just test` (mirrors CI). Adding the full AGENTS.md
   gate is a separate structural PR.
6. Housekeeping (optional): regenerate `marimo/catalog_snapshot*.json`
   (`just profile=<name> catalog-snapshot`) after any reindex; Docker
   Desktop's WSL2 vhdx grows with the ~27 GB of volume data and does not
   shrink automatically.
7. Owner decisions (open, in `docs/decision-queue.md`): finance / pii
   quality bars, and whether pii should mask PII spans in the embedding
   input.
8. finance quality candidates (each its own work unit, evidence in the
   2026-11-11 research doc): a vector-leaning RRF weight A/B for finance
   (vector-only beats fused on several queries), and moving person names
   out of the BM25 text (the keyword leg matched 奨太 for 奨学金).
9. Containers: the global rules require Podman 5+; this machine has only
   Docker Desktop, which the owner allowed for this session only.
   Migrating the emulator stack (and the ~27 GB of volumes) to Podman is
   an open work unit.

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
- GPU sharing: a vLLM server (Qwen3.8-27B) in WSL Ubuntu holds most of
  the RTX 4090. Embedding speed swung between 35 and 130 docs/s, and the
  finance full index took 151 min (japan took ~50 min alone).
- Disk: C: is at 91% (173 GB free) with all three indexes in the
  emulator volumes (~27 GB).

## Context the Next Actor Needs
- Emulators are vendored at `emulator/compose.yaml`; ports/auth match
  `ApplicationConfig`. The canonical full kit lives at `~/dotfiles/emulator`
  (upstream `github.com/hironow/emulator-set`).
- **All three datasets are indexed and persistent** — day-to-day work
  does not need reindexing. A full rebuild is `just full-index` (~50 min
  on the RTX 4090) or `just profile=<finance|pii> full-index`; a failed
  run resumes by rerunning the failed shard only.
- Recipe arguments are positional (`just qa-index ruri-v3-310m`); only the
  top-level `profile` variable takes `name=value`.
- Before touching one dataset's stores for real, take
  `just profile=<other> store-fingerprint` and compare it afterwards.
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
  `marimo/catalog_snapshot*.json`, dataset selector)
- `search_ja_persona/dataset_profiles.py` — per-dataset fields and stores
- `docs/research/2026-11-11-finance-pii-baseline.md` — finance / pii
  baselines and the japan-unchanged evidence
- `just store-fingerprint` — read-only before/after check of one dataset
