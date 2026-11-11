# Handover

**Last updated:** 2026-11-11 (JST)
**Updated by:** claude (AI session, Windows workstation)

## Current State

All three datasets are fully indexed with `ruri-v3-310m` (768d) in their own
stores, and the counts agree across Qdrant, Elasticsearch, and Neo4j:
japan 1,000,000 (`personas` / `Persona`), finance 1,000,000
(`finance_personas` / `FinancePersona`), and pii 100,000 (`pii_documents` /
`PiiDocument`). japan meets every ratified bar (golden basic 0.917, hard
0.650, filtered geo 0.950, self-retrieval recall@1 0.92 / @10 1.00, all
canaries). finance (basic 0.457 / hard 0.425) and pii (basic 0.900 / hard
0.880) have baselines but no bars yet. `main` is at the squash merge of #101;
CI (ubuntu + windows) is green. The emulators are stopped; their volumes
(~27 GB) keep all three indexes. Measurements:
`docs/research/2026-11-11-finance-pii-baseline.md` (finance / pii and the
proof that japan stayed unchanged) and the 2026-08-27 research docs (japan).

## In Progress

Branch `docs/refresh-markdown`: brings README, architecture,
storage-footprint, emulator/README, and this file up to date after #101.

## Next Actions

1. Owner decisions (`docs/decision-queue.md`, Open Items): finance / pii
   quality bars, whether pii masks PII spans in the embedding input, and the
   proposed japan name-lookup floor (recall@1 >= 0.70).
2. Search-quality candidates, each its own work unit:
   - finance: a vector-leaning RRF weight A/B (vector-only beats fused on
     several queries).
   - japan and finance: move person names out of the BM25 text into a
     keyword field. Exact-name lookup is japan recall@1 0.725, and the
     finance keyword leg matched 奨太 for 奨学金.
   - japan: pooled human relevance judgments (qrels) to replace keyword
     predicates.
3. Metadata robustness (structural): atomic writes, and recording the
   Qdrant endpoint in the metadata.
4. Gate gap: `just check` runs prek (ruff) and pytest only. AGENTS.md also
   requires `ty` (installed, not run) and semgrep (no `.semgrep/` yet). The
   justfile comment above `check` still names mypy.
5. Containers: the global rules require Podman 5+, but this machine has only
   Docker Desktop, which the owner allowed for the #101 session. Migrating
   the emulator stack and its ~27 GB of volumes to Podman is open.
6. Housekeeping: rerun `just profile=<name> catalog-snapshot` after any
   reindex. Docker Desktop's WSL2 virtual disk grows with the volume data and
   does not shrink on its own.

## Known Risks / Blockers

- On this machine `uv run` / `uv lock` without `--frozen` or `UV_NO_CONFIG=1`
  rewrites `uv.lock` (the machine-local `~/.config/uv/uv.toml` sets a mirror
  index and `exclude-newer`). Every justfile recipe passes `--frozen`; keep new
  uv calls frozen, and use `UV_NO_CONFIG=1` for lock changes meant to be
  committed.
- The emulators must be running for index, search, and eval
  (`cd emulator && docker compose up -d`; on Windows start Docker Desktop
  first). Only `down -v`, a volume prune, or a Docker Desktop purge destroys
  the volumes.
- Metadata files under `.cache/` are CWD-relative and can go missing.
  `search` and `index` then refuse to guess; recover with
  `repair-metadata --embedder <preset> [--dataset-profile <name>]`.
- The japan QA sample (`just qa-sample`) needs the Hugging Face cache
  (`download-dataset`); finance and pii cut theirs from the submodules.
- GPU sharing: a vLLM server (Qwen3.8-27B) in WSL Ubuntu holds most of the
  RTX 4090. Embedding speed swung between 35 and 130 docs/s, and the finance
  full index took 151 min (japan alone takes about 50 min).
- Disk: C: is at 91% (173 GB free).

## Context the Next Actor Needs

- Each dataset is selected by a profile: `--dataset-profile` on the CLI,
  `just profile=<name> <recipe>` in the justfile (default `japan`). Guards
  refuse another dataset's files, store names, and metadata (ADR 0004).
- Before touching one dataset's stores for real, take
  `just profile=<other> store-fingerprint` and compare it afterwards.
- Recipe arguments are positional (`just qa-index ruri-v3-310m`); only the
  top-level `profile` variable takes `name=value`.
- A full rebuild is `just profile=<name> full-index`; a failed run resumes
  from the failed shard (uuid-keyed upserts are idempotent).
- Switching embedder presets asks before resetting the profile's stores; the
  reset deletes Neo4j nodes in batches and saves the new metadata at once.
- `emulator/compose.yaml` is a vendored subset; the full kit lives in
  `~/dotfiles/emulator` (upstream `github.com/hironow/emulator-set`).
- Tooling: `uv` (Python deps), `mise` (tool versions), `just` (tasks; recipes
  run under bash, which on Windows means Git Bash on `PATH`).

## Relevant Files and Commands

- `search_ja_persona/dataset_profiles.py`: per-dataset fields and stores.
- `search_ja_persona/cli.py`: CLI (index / search / download-dataset /
  clear-emulators / repair-metadata).
- `docs/architecture.md`: system architecture; `docs/storage-footprint.md`:
  disk needs.
- `just check`: local gate. `just integration`: emulator tests.
- `just profile=<name> eval --check-thresholds`: quality gate per dataset.
- `just notebook` / `just panel`: marimo catalog / persona panel.
