# Handover

**Last updated:** 2026-08-27 (JST)
**Updated by:** claude (AI session, Windows workstation)

## Current State
Windows (11 + Docker Desktop/WSL2) is now a verified dev environment: `uv sync
--frozen`, the unit suite (31 passed / 1 skipped), `just integration` against
live emulators, and the full `just qa` flow (1k personas indexed into
Qdrant/Elasticsearch/Neo4j, fused Japanese search returning results) all pass
locally. The lock is multi-platform by declaration (`required-environments`)
and Windows installs CUDA torch (2.13.0+cu130, verified on RTX 4090). All work
sits on branch `fix/windows-portability`, pushed as PR #35 at hironow's
direction; both queued decision-queue items were resolved in-session per the
requester's instruction, and `ci.yaml` gained a `windows-latest` unit-test
leg.

## In Progress
Branch `fix/windows-portability` (8 commits) awaits human review and push/PR:

1. `fix(cli)` UTF-8 index-metadata I/O (was cp932 on Japanese Windows).
2. `test(cli)` platform-agnostic fastembed cache-dir assertion.
3. `perf(cli)` emulator hosts default to `127.0.0.1` — Windows resolves
   `localhost` to `::1` first and stalled ~250ms per request (measured
   264ms → 3ms; `just qa` wall time ~8min → 41s).
4. `fix(cli)` force UTF-8 stdout/stderr; tolerate undecodable metadata files.
5. `chore(dev)` quote just interpolations; add `.gitattributes` (LF).
6. `docs(readme)` Windows bash-on-PATH prerequisite.
7. `fix(deps)` lock upgrade clearing all 5 open Dependabot alerts:
   transformers 4.56.2→5.16.1, protobuf 6.32.1→7.36.0, pygments
   2.19.2→2.21.0; sentence-transformers 5.7.0→6.0.0 rides along.
8. `build(uv)` resolution policy in pyproject: `required-environments`
   (Linux x86_64/aarch64, macOS arm64, Windows AMD64 wheel coverage
   enforced at lock time) + `exclude-newer = "7 days"` cooldown
   (mandated by the repo semgrep rule for any `[tool.uv]` block).
9. `build(uv)` Windows pulls CUDA torch (2.13.0+cu130) from the explicit
   pytorch.org index; Linux/macOS keep PyPI wheels. torch declared as a
   direct dependency so `tool.uv.sources` applies.
10. `feat(embeddings)` + `perf(indexer)` batch embedding: `embed_many` on
    the Embedder protocol (all three backends) and one batched encode per
    ingest batch. Measured: 9.7ms → 0.24ms per text on the RTX 4090.
11. `feat(services)` + `perf(indexer)` batched Neo4j ingest: UNWIND-based
    `merge_personas`, one transaction per batch. `just qa` overall:
    40s → 26s; verified against the live emulator (`just integration`,
    1000/1000 persona nodes).
12. `fix(services)` backend response errors surfaced: Neo4j tx/commit
    `errors` and Elasticsearch `_bulk` per-item errors (both hidden
    behind HTTP 200) now raise RuntimeError with details. This is what
    silently dropped 5/1000 personas in the old per-item merge path;
    live-verified with a null-uuid merge.
13. `feat(embeddings)` + `feat(search)` asymmetric retrieval API:
    embed_query/embed_documents with per-preset prefixes (e5:
    query:/passage:, Ruri: 検索クエリ:/検索文書:) and encode batch caps.
    New recommended preset `ruri-v3-310m` (768 dims, Apache-2.0) per the
    2026-08 model research: Ruri's tokenizer avoids the ~23% document
    truncation multilingual-e5 suffers on this corpus. Live-verified:
    1k reindex in 9s on the RTX 4090, eyeball relevance strong on 4/5
    Japanese queries (the 5th lacks matching personas in the 1k sample).

## Next Actions
1. After PR #35 merges: confirm the 5 Dependabot alerts auto-closed and the
   new `windows-latest` CI leg is green (first run pays the CUDA-torch
   download; later runs hit the uv cache).
2. Search relevance quality bar is still undefined (intent.md open
   question) — no benchmarks exist beyond "returns merged results".
3. Full-corpus indexing (1M rows) remains restore-on-demand: pull the
   remaining LFS shards, then `index --dataset datasets/.../data`.

## Known Risks / Blockers
- On this machine `uv run`/`uv lock` **without** `--frozen`/`UV_NO_CONFIG=1`
  rewrites `uv.lock`: the machine-local `~/.config/uv/uv.toml` (harden_env.sh)
  sets a flatt-mirror default index and `exclude-newer`. Lock operations meant
  for commit were run with `UV_NO_CONFIG=1` so the committed lock stays
  pypi.org-portable — note this bypasses the machine's package firewall for
  those resolutions.
- Emulators must be running (`cd emulator && docker compose up -d`); Docker
  Desktop's engine needs to be started first on Windows.
- The dataset submodule is checked out with LFS pointers only, except
  `data/train-00000-of-00008.parquet` (216MB pulled). The current
  `qa_samples/qa_sample.parquet` was cut from that shard directly;
  `scripts/generate_qa_sample.py` itself still requires the HF cache
  (`download-dataset`, full corpus).

## Context the Next Actor Needs
- Emulators are vendored at `emulator/compose.yaml`; ports/auth match
  `ApplicationConfig`. The canonical full kit lives at `~/dotfiles/emulator`
  (upstream `github.com/hironow/emulator-set`).
- The repo runs QA-sample-only day to day; full-corpus `index` needs the
  submodule's LFS blobs (`git -C datasets/Nemotron-Personas-Japan lfs pull`,
  ~1.7GB total).
- Tooling: `uv` (Python deps), `mise` (tool versions), `just` (tasks; recipes
  run under bash — on Windows that means Git Bash on PATH).
- Score semantics differ by backend: Qdrant cosine similarity vs. mapped
  Elasticsearch `_score`.

## Relevant Files and Commands
- `search_ja_persona/cli.py` — CLI entry point (index / search /
  download-dataset / clear-emulators)
- `emulator/compose.yaml` — standalone local emulator stack
- `docs/architecture.md` — system architecture; `docs/adr/0001` — emulator
  vendoring decision; `docs/decision-queue.md` — items needing human decisions
- `just qa` — index + search the 1k sample; `just test` / `just integration`
- `just notebook` — marimo feature-catalog notebook (`marimo/catalog.py`)
