# 0004. Index each dataset into its own stores through a dataset profile

**Date:** 2026-11-11
**Status:** Accepted

## Context

The requester added two NVIDIA datasets next to Nemotron-Personas-Japan:
Nemotron-Personas-Finance-Japan (1M personas, 9 persona texts and
household finance attributes) and Nemotron-PII-Japan (100k synthetic
documents with PII span annotations). They asked for everything the japan
corpus supports (index, search, filters, graph context, evaluation,
notebooks) for both, for personal research, with each dataset handled
independently.

The code hard-wired one dataset: a single `personas` collection and
index, one `Persona` label, one `.cache/index_metadata.json`, and the six
japan text fields in the mapping, indexer, search results, and golden
validator. Finance shards carry every japan column and the same file
names, so the indexer would have mixed them into the live japan index
without an error; PII files have no `uuid` and streamed as all-null
records. The live 1M japan index had to stay untouched.

## Decision

Describe each dataset with a `DatasetProfile`
(`search_ja_persona/dataset_profiles.py`) that owns its columns, text and
payload fields, filter fields, graph shape, embedding-text policy, and its
own Qdrant collection, Elasticsearch index, Neo4j label, and metadata file.
Every store request goes through the profile; `--dataset-profile` and the
justfile's `profile` variable select it, defaulting to japan, whose
profile reproduces the previous requests byte for byte (pinned by
`tests/test_japan_wire_contract.py`).

Guards keep the datasets apart: the repository detects a parquet file's
dataset from its columns and refuses a mismatch; a profile refuses another
profile's store names; metadata records its dataset and a mismatch fails
closed; Neo4j constraints, merges, counts, and deletes are scoped to the
profile's label. `Prefecture` / `Region` nodes stay shared as reference
data.

## Consequences

### Positive
- finance and pii get the full feature set (QA flow, full index, filters,
  graph context, eval with canaries, diagnose, lookup benchmark, catalog,
  and the finance persona panel) without touching japan.
- Clearing a custom collection no longer deletes the default index's
  metadata (a known bug).

### Negative
- Search filters changed from a `prefecture=` argument to a
  `filters` mapping; callers outside the repo would need updating.
- The justfile repeats each profile's source path and QA-sample path; a
  dry-run test keeps them in step.

### Neutral
- pii embeds raw document text (`raw-text-v1`); masking PII spans in the
  embedding input is an untested alternative.
- Only japan has ratified quality bars; finance and pii gate on the
  canaries until the requester rules on theirs.
