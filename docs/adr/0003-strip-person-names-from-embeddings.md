# 0003. Strip person names from embedding inputs

**Date:** 2026-08-27
**Status:** Accepted

## Context

Every persona text opens with (and repeats) the persona's name, and the
embedding model attends to it heavily: 福岡-surnamed personas surfaced
for 福岡県 residence queries, a Mr. 温泉 topped hot-spring queries, and
Mr./Ms. パン dominated baking queries — 6 of 12 hard golden queries
showed the pattern. The dataset has no separate name column, and the
name signal is never what a semantic query is asking for; exact-name
lookup is a lexical problem the keyword leg already handles.

## Decision

Strip the persona's name from the text that gets embedded
(`strip-person-names-v1`, `search_ja_persona/name_stripping.py`), while
stored text keeps names everywhere (payload, Elasticsearch, display).
Names are detected from the field openings and accepted only when at
least two fields agree, with pronoun stopwords, a conflicting-candidate
no-op, and particle/punctuation boundaries on removal (a surname that
doubles as a place name survives inside real place references). The full
corpus was re-embedded in place via the idempotent indexer; migration
completeness was proven content-wise (recomputed embeddings vs live
vectors, 40/40 stratified cosine = 1.0).

## Consequences

- Vector-leg quality jumped (basic 0.900 -> 1.000, hard 0.433 -> 0.650,
  first time above BM25); the name-pollution class disappeared.
- A persona's own named summary no longer guarantees rank-1 (recall@1
  0.92, recall@10 1.00): anonymized vectors intentionally treat
  same-vibe personas as equivalent. The requester amended the intent bar
  accordingly (recall@1 >= 0.90 ∧ recall@10 >= 0.99).
- Exact-name lookup rests on the keyword leg (tracked by
  `just eval-names`; strengthening it is a recorded candidate).
- Measurements: `docs/research/2026-08-27-name-exclusion-results.md`.
