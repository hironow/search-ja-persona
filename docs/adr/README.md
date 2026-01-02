# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the search-ja-persona project.

## What is an ADR?

An ADR captures the context and reasoning behind significant architectural decisions. While `docs/` describes the current state (the "What"), ADRs preserve the history of decisions (the "Why").

## When to Create an ADR

Create an ADR when:

- Introducing new technology or framework
- Changing established patterns or conventions
- Making non-obvious tradeoffs with significant consequences
- Deprecating or replacing existing approaches
- Making decisions that future developers might question

## File Naming

Use the pattern: `NNNN-short-title.md`

Examples:
- `0001-use-qdrant-for-vector-search.md`
- `0002-adopt-sentence-transformers.md`

## Template

```markdown
# {NNNN}. {Title}

**Date:** YYYY-MM-DD
**Status:** Proposed / Accepted / Deprecated / Superseded by [NNNN]

## Context

{The problem and constraints at the time of this decision.
What forces are at play? What are we trying to achieve?}

## Decision

{What we are doing. State the decision clearly and concisely.}

## Consequences

### Positive
- {Benefit 1}
- {Benefit 2}

### Negative
- {Tradeoff or downside 1}
- {Tradeoff or downside 2}

### Neutral
- {Side effect or implication that is neither clearly positive nor negative}
```

## Immutability

ADRs are immutable after acceptance. To change a decision, create a new ADR that supersedes the old one and update the old ADR's status to "Superseded by [NNNN]".

## Index

No ADRs have been created yet.
