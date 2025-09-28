# Persona Search Plan

## Goals
- Index personas from `datasets/Nemotron-Personas-Japan` into the local emulator services (Qdrant for vector similarity, Elasticsearch for keyword search, Neo4j for relationship exploration).
- Provide a Python API that supports free-form queries by fusing vector and keyword signals and enriches the top persona with graph context.
- Follow TDD: drive behavior with failing tests, implement minimal code, then consider refactoring.

## Approach
1. Build a lightweight `PersonaRepository` to stream persona records from parquet files without loading the entire dataset into memory.
2. Implement a deterministic hashed character n-gram embedder to generate fixed-size vectors without downloading external models.
3. Create service-specific clients (`QdrantService`, `ElasticsearchService`, `Neo4jService`) that wrap HTTP interactions with the local emulators, keeping transport swappable for testing.
4. Develop a `PersonaIndexer` that reads personas from the repository, embeds them, and synchronizes each service (collection/index creation + batched upserts).
5. Provide a `PersonaSearchService` that embeds the query, queries Qdrant for candidate UUIDs, refines via Elasticsearch, and fetches relationship annotations for the best match from Neo4j.
6. Expose a cohesive orchestration entry point (e.g., `main.py` or module function) illustrating indexing and search usage with dependency injection for paths and hosts.

## Testing Strategy
- Write unit tests with fake HTTP transports to assert request payloads and orchestration logic (no large mocks, but small fakes capturing method calls).
- Ensure the embedder and repository behavior is covered by direct tests (deterministic vectors, chunked iteration).
- Keep tests independent from running emulators while mirroring real payloads so they double as usage documentation.

## Open Questions / Assumptions
- Assume emulators are reachable at the default ports listed in `emulator/README.md` when running outside tests.
- Initial implementation will index a configurable sample size to keep iteration tractable; scaling strategies can follow once validated.
