# 0001. Vendor a minimal emulator compose stack instead of a submodule

**Date:** 2026-07-27
**Status:** Accepted

## Context

The local emulators (Qdrant, Elasticsearch, Neo4j) were originally pulled in as a
git submodule at `emulator/`, pointing to `git@github.com:hironow/emulator-set.git`.
That submodule was later removed (commit `dc48659`) with the stated intent of
relocating it to a sibling checkout, but the referenced destination was not
present in working checkouts, and the docs still described a `just start` entry
point that no longer existed. The net effect was no reliable way to stand up the
three backends the CLI needs.

Two forces made a plain submodule the wrong fit:

- A submodule can only be initialized while its upstream remains reachable — the
  opposite of what we want if the upstream ever disappears.
- `emulator-set` is a large multi-emulator kit (Bigtable, Spanner, Firebase,
  MLflow, and more); this project only ever talks HTTP to three of its services.

## Decision

Vendor a minimal, standalone `emulator/compose.yaml` into this repository,
containing only the Qdrant, Elasticsearch, and Neo4j service definitions this
codebase uses, with the external network, Compose profiles, and companion CLI
containers removed. Ports and the Neo4j credential pair match the
`ApplicationConfig` defaults, so `docker compose up -d` is sufficient.

## Consequences

### Positive

- The stack runs standalone; the codebase keeps working even if the upstream
  emulator-set becomes unavailable.
- No large submodule checkout — only the three services this project needs.
- Setup is a single `docker compose up -d`, and the docs match reality again.

### Negative

- The vendored service definitions can drift from the canonical emulator-set and
  must be updated by hand when image versions or settings change upstream.

### Neutral

- `github.com/hironow/emulator-set` remains the source of truth for the full
  multi-emulator kit; this repo intentionally carries only a subset.
