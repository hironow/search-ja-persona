# Local Emulators

Self-contained Docker Compose stack for the three backends search-ja-persona
uses: **Qdrant** (vector), **Elasticsearch** (keyword), and **Neo4j** (graph).

This is a vendored, minimal subset derived from
[github.com/hironow/emulator-set](https://github.com/hironow/emulator-set),
trimmed to only these three services and stripped of the external network,
profiles, and companion CLI containers so it runs standalone — the codebase
keeps working even if the upstream set becomes unavailable.

## Start / Stop

```bash
cd emulator
docker compose up -d          # start Qdrant, Elasticsearch, Neo4j
docker compose ps             # check health
docker compose down           # stop (keeps named volumes / data)
docker compose down -v        # stop and delete all indexed data
```

## Endpoints

| Service | Port | URL / auth |
|---------|------|------------|
| Qdrant | 6333 (REST), 6334 (gRPC) | http://localhost:6333 |
| Elasticsearch | 9200 (REST), 9300 (transport) | http://localhost:9200 |
| Neo4j | 7474 (HTTP), 7687 (Bolt) | http://localhost:7474 — `neo4j` / `password` |

These match the defaults in `search_ja_persona/application.py`
(`ApplicationConfig`), so the CLI works without extra flags once the stack is up.

## Overriding Ports / Credentials

Every port and the Neo4j auth pair read from environment variables with the
defaults above, e.g. `NEO4J_HTTP_PORT`, `QDRANT_REST_PORT`, `NEO4J_AUTH`. Set
them in the shell or a `.env` file next to `compose.yaml` to change them, then
pass the matching `--*-host/--*-port` flags to the CLI.

## Data Volumes

Indexed data lives in named Docker volumes (`qdrant_data`,
`elasticsearch_data`, `neo4j_data`, ...). See
[`../docs/storage-footprint.md`](../docs/storage-footprint.md) for how large
these grow and how to reclaim the space.
