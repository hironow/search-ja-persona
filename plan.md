# Persona Search Plan

## Goals
- Index personas from `datasets/Nemotron-Personas-Japan` into the local emulator services (Qdrant for vector similarity, Elasticsearch for keyword search, Neo4j for relationship exploration).
- Provide a Python API that supports free-form queries by fusing vector and keyword signals and enriches the top persona with graph context.
- Follow TDD: drive behavior with failing tests, implement minimal code, then consider refactoring.

## Completed Foundations
- Streaming repository, hashed n-gram vectorizer, and service clients are implemented with unit coverage.
- Batch indexer coordinates embeddings + emulator sync, and search orchestration is available through `PersonaApplication`.
- CLI and application layers demonstrate dependency injection patterns for local usage.

## Next Focus: Large-Scale Indexing Workflow
1. **Dataset Manifest & Sampling**
   - Enumerate parquet shards under `datasets/Nemotron-Personas-Japan` into a deterministic manifest (sorted list with row counts when available).
   - Provide a helper to build a small sampled subset (e.g., first N shards) for smoke testing before committing to full indexing.
   - Drive with tests that manifest ordering is stable and sampling respects limits.
2. **Resumable Progress Tracking**
   - Introduce a progress store (e.g., JSON state file) that records the last successfully processed shard and intra-shard row offset.
   - Update `PersonaIndexer` (or a wrapper) to persist progress after each batch and resume by skipping completed work.
   - Cover with tests simulating interruption and verifying resume continues from the recorded point without duplicating entries.
3. **CLI Support for Manifest + Resume**
   - Extend the `index` command to accept `--manifest` and `--state-file` options (default locations under `datasets/` and `.cache/`).
   - Ensure CLI validates the manifest exists, initializes progress state when absent, and surfaces resume instructions in help text.
   - Add integration-style tests using fakes to assert CLI passes state paths through to the indexer.
4. **Operational Guardrails**
   - Implement chunked commit sizes (tunable batch size) and optional throttling to avoid overwhelming emulators.
   - Add logging or metrics hooks to report processed shard counts and estimated completion time (can be stubbed in tests).
5. **Documentation + Recipes**
   - Document end-to-end indexing flow in the README: manifest creation, smoke run on sample shards, full run with resume.
   - Provide troubleshooting notes for restarting after failure and cleaning up partial emulator state if needed.

## Testing Strategy
- Continue using fake transports and state-file fixtures to keep tests independent of running emulators.
- Add resume-path tests that write to temporary directories to mimic abrupt termination scenarios.
- Keep dataset-manifest logic pure (no network or emulator dependencies) so it can be validated quickly in CI.

## Open Questions / Assumptions
- Assume emulators are reachable at the default ports listed in `emulator/README.md` when running outside tests.
- Manifest generation should tolerate missing or newly added shards by re-scanning directories when requested.
- Progress tracking will focus on local filesystem state; distributed coordination (e.g., cloud storage) can follow later if needed.
