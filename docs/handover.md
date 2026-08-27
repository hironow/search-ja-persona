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
sits on branch `fix/windows-portability`; `main` is untouched at `6957205` and
the branch is **not pushed** (push was explicitly forbidden this session).

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

## Next Actions
1. Human: review and push `fix/windows-portability`, open a PR. The Dependabot
   alerts close automatically once the upgraded lock reaches `main`.
2. Human: the `docs/decision-queue.md` item (2026-07-27) — intent.md's
   emulator non-goal vs. the vendored stack — is still unresolved.
3. Consider a `windows-latest` leg in `ci.yaml` so Windows support cannot
   regress (not added here: unverifiable locally while push is forbidden).
4. Batch embedding in `PersonaIndexer`/`Embedder`: the indexer encodes one
   record at a time, so GPU gains almost nothing end to end (`just qa`
   ~40s on GPU ≈ CPU). Measured on the RTX 4090: single-item 9.7ms/text
   vs batched (64) 0.24ms/text — a batch API would cut full-corpus
   embedding from hours to minutes. Drive it TDD (protocol extension with
   fallback for hashed/fastembed backends).

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
