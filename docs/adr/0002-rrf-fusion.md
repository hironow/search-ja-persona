# 0002. Fuse retrieval legs with weighted Reciprocal Rank Fusion

**Date:** 2026-08-27
**Status:** Accepted

## Context

The original fusion filled the top-`limit` with Qdrant vector hits and
appended Elasticsearch-only hits afterwards, truncating them away. On a
fully indexed corpus Qdrant always returns `limit` candidates, so the
keyword leg contributed nothing (fused/keyword top-5 uuid overlap ~0)
while BM25 alone outscored the fused ranking on the hardened golden
benchmark (hard tier 0.583 vs 0.433). The two legs are complementary:
keyword wins on names, rare terms, and literal conjunctions; vector wins
on paraphrase. Their scores (cosine vs BM25) are not comparable, which
rules out score-based mixing.

## Decision

Rank results by weighted Reciprocal Rank Fusion:
`rrf_score(d) = Σ_leg w_leg / (60 + rank_leg(d))`, K=60, production
weights 1:1 — chosen by a pre-registered A/B on the live 1M index against
a 2:1 vector-weighted variant (which degenerated to the old behavior).
Both legs fetch `max(limit, min(3 * limit, 30))` candidates; the ordering
is fully specified (rrf desc → source count desc → best rank asc →
keyword-leg presence → uuid asc), and Neo4j context is fetched only for
the returned top-`limit`. The exact-tie preference for the keyword leg
exists because a BM25 rank-1 is a strong identity signal (names, rare
terms) while a vector rank-1 among a million near-duplicates is not.

## Consequences

- The keyword leg genuinely contributes again; a canary
  (keyword-contribution > 0) guards against silent regression.
- `score` keeps its historical per-leg meaning; ranking moved to the new
  `rrf_score`, and `sources` records provenance.
- Fetching deeper lists costs more per leg but restricting context
  lookups to the final top-`limit` made the benchmark ~20% faster net.
- Measurements: `docs/research/2026-08-27-rrf-fusion-results.md`.
