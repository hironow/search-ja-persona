import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", app_title="search-ja-persona feature catalog")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 🗾 search-ja-persona — Feature Catalog

        A runnable tour of everything this system does. Pick a dataset
        below (each has its own stores), then read top to bottom:

        1. **Data model** — the dataset's schema and text fields.
        2. **Corpus exploration** — DuckDB SQL over the generated QA parquet.
        3. **Embedders** — the pluggable vectorization backends and their
           retrieval prefixes.
        4. **Pipeline checks** — emulator health, store counts, and the
           recorded index metadata.
        5. **Recorded results** — a committed snapshot of real full-corpus
           queries, visible even with the emulators down.
        6. **Live search** — fused vector + keyword + graph search against
           the indexed corpus (button-triggered; loads the embedding model).
        """
    )


@app.cell
def _(PROFILES, mo):
    dataset_choice = mo.ui.dropdown(
        options={
            "Nemotron-Personas-Japan (japan)": "japan",
            "Nemotron-Personas-Finance-Japan (finance)": "finance",
            "Nemotron-PII-Japan (pii)": "pii",
        },
        value="Nemotron-Personas-Japan (japan)",
        label="Dataset",
    )
    assert set(dataset_choice.options.values()) == set(PROFILES)
    dataset_choice
    return (dataset_choice,)


@app.cell
def _(dataset_choice, get_profile):
    profile = get_profile(dataset_choice.value)
    return (profile,)


@app.cell
def _():
    import concurrent.futures
    import json
    import sys
    import time
    from pathlib import Path

    # Make the project importable and all paths stable regardless of how the
    # notebook is launched (marimo does not put the repo root on sys.path or
    # guarantee the working directory the way `python -m` does).
    ROOT = next(
        (
            base
            for base in [Path.cwd(), *Path.cwd().parents]
            if (base / "search_ja_persona").is_dir()
        ),
        Path.cwd(),
    )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.evaluate_search import collect_store_counts
    from scripts.generate_qa_sample import qa_sample_path
    from search_ja_persona.application import ApplicationConfig, PersonaApplication
    from search_ja_persona.dataset_profiles import PROFILES, get_profile
    from search_ja_persona.embeddings import EMBEDDER_PRESETS, HashedNgramEmbedder
    from search_ja_persona.repository import PersonaRepository
    from search_ja_persona.services import RequestDescriptor, SimpleHttpTransport

    def run_io(fn, *args, **kwargs):
        # The emulator HTTP transport spins up its own event loop, which
        # clashes with marimo's already-running loop; run these calls in a
        # worker thread (a fresh thread has no running loop).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            return _ex.submit(fn, *args, **kwargs).result()

    CACHE_DIR = ROOT / ".cache"
    app_cache: dict = {}
    return (
        ApplicationConfig,
        CACHE_DIR,
        EMBEDDER_PRESETS,
        HashedNgramEmbedder,
        PROFILES,
        PersonaApplication,
        PersonaRepository,
        ROOT,
        RequestDescriptor,
        SimpleHttpTransport,
        app_cache,
        collect_store_counts,
        get_profile,
        json,
        qa_sample_path,
        run_io,
        time,
    )


@app.cell
def _(CACHE_DIR, ROOT, profile, qa_sample_path):
    QA_PARQUET = ROOT / qa_sample_path(profile.name)
    _suffix = "" if profile.name == "japan" else f".{profile.name}"
    SNAPSHOT_PATH = ROOT / "marimo" / f"catalog_snapshot{_suffix}.json"
    INDEX_METADATA_PATH = CACHE_DIR / profile.metadata_filename
    return INDEX_METADATA_PATH, QA_PARQUET, SNAPSHOT_PATH


@app.cell
def _(QA_PARQUET, mo, profile):
    _metadata = [c for c in profile.columns if c not in profile.text_fields]
    mo.md(
        f"""
        ## 1. Data model — `{profile.name}`

        Records stream from parquet via `PersonaRepository` (source
        `{profile.source_dir}`). Each record carries metadata
        ({", ".join(f"`{c}`" for c in _metadata)}) plus
        **{len(profile.text_fields)} free-text fields** that get aggregated
        and embedded (`{profile.embedding_text_policy}`):

        {"".join(f"- `{f}`\n" for f in profile.text_fields)}

        Stores: Qdrant `{profile.qdrant_collection}`, Elasticsearch
        `{profile.es_index}`, Neo4j `:{profile.node_label}`. Filters:
        {", ".join(f"`{f}`" for f in profile.filter_fields)}.

        QA sample: `{QA_PARQUET.name}` (1k rows, generated locally with
        `just profile={profile.name} qa-sample`).
        """
    )


@app.cell
def _(PersonaRepository, QA_PARQUET, mo, profile):
    if QA_PARQUET.exists():
        _repo = PersonaRepository([QA_PARQUET], profile=profile)
        _records = list(_repo.iter_personas(limit=200))
        _preview = [
            {
                "uuid": str(r.get("uuid", ""))[:8],
                **{f: r.get(f) for f in profile.payload_fields},
                profile.summary_field: (r.get(profile.summary_field) or "")[:120],
            }
            for r in _records
        ]
        sample_view = mo.ui.table(
            _preview,
            show_search=True,
            pagination=True,
            page_size=8,
            selection=None,
            column_widths={"uuid": 90, profile.summary_field: 420},
            label=f"First {len(_records)} records from the QA sample",
        )
    else:
        sample_view = mo.callout(
            mo.md(
                f"QA sample not found — generate it with "
                f"`just profile={profile.name} qa-sample` (needs the dataset "
                "submodule, or the HF cache for japan)."
            ),
            kind="warn",
        )
    sample_view


@app.cell
def _(QA_PARQUET, mo, profile):
    mo.stop(
        not QA_PARQUET.exists(),
        mo.md("_Corpus SQL skipped — QA sample not generated._"),
    )
    # Payload fields are source columns in every dataset's parquet.
    _first, _second = profile.payload_fields
    mo.md(
        f"""
        ## 2. Corpus exploration (DuckDB SQL)

        marimo SQL cells run DuckDB directly over the parquet — no ingestion,
        no server. Here is the `{_first}` distribution of the sample.
        """
    )
    payload_distribution = mo.sql(
        f"""
        SELECT {_first},
               {_second},
               count(*) AS records
        FROM read_parquet('{QA_PARQUET.as_posix()}')
        WHERE {_first} IS NOT NULL
        GROUP BY {_first}, {_second}
        ORDER BY records DESC
        LIMIT 15
        """
    )


@app.cell
def _(EMBEDDER_PRESETS, mo):
    mo.md(
        """
        ## 3. Embedder backends

        The `Embedder` protocol embeds **queries and documents asymmetrically**
        (`embed_query` / `embed_documents`): retrieval models like Ruri and e5
        expect a side-specific prefix, declared per preset and applied
        automatically by the pipeline. `ruri-v3-310m` is the production preset
        for the Japanese corpus.
        """
    )
    embedder_choice = mo.ui.dropdown(
        options=list(EMBEDDER_PRESETS.keys()),
        value="ruri-v3-310m",
        label="Embedder preset",
    )
    embed_text = mo.ui.text(
        value="高齢者介護の経験豊富なケアマネージャー",
        label="Text to embed (hashed preset only)",
        full_width=True,
    )
    mo.vstack([embedder_choice, embed_text])
    return embed_text, embedder_choice


@app.cell
def _(EMBEDDER_PRESETS, HashedNgramEmbedder, embed_text, embedder_choice, mo):
    _preset = EMBEDDER_PRESETS.get(embedder_choice.value, {})
    _type = _preset.get("type", embedder_choice.value)

    _stats = [
        mo.stat(embedder_choice.value, label="preset"),
        mo.stat(_type, label="backend"),
        mo.stat(_preset.get("model", "—"), label="model"),
        mo.stat(_preset.get("query_prefix", "—") or "—", label="query prefix"),
        mo.stat(_preset.get("document_prefix", "—") or "—", label="document prefix"),
        mo.stat(_preset.get("encode_batch_size", "—"), label="encode batch"),
    ]

    if _type == "hashed":
        _emb = HashedNgramEmbedder(dimension=256, ngram_sizes=(2, 3))
        _vector = _emb.embed_query(embed_text.value)
        _preview = ", ".join(f"{v:.3f}" for v in _vector[:8])
        _detail = mo.md(
            f"Live demo (offline, {_emb.dimension} dims) — first 8 components: "
            f"`[{_preview}, ...]`"
        )
    else:
        _detail = mo.callout(
            mo.md(
                "Model-backed preset — weights load on first use. Try it via "
                "the **Live search** section below, or from a shell with "
                f"`just qa-index {embedder_choice.value}`."
            ),
            kind="neutral",
        )
    mo.vstack([mo.hstack(_stats, wrap=True), _detail])


@app.cell
def _(
    ApplicationConfig,
    INDEX_METADATA_PATH,
    PersonaApplication,
    RequestDescriptor,
    SimpleHttpTransport,
    collect_store_counts,
    json,
    mo,
    profile,
    run_io,
):
    def _probe(host, port, path, auth=None):
        try:
            _t = SimpleHttpTransport(host, port, timeout=1.5, auth=auth)
            return run_io(_t.request, RequestDescriptor("GET", path))
        except Exception:
            return None

    _qdrant_health = _probe("127.0.0.1", 6333, "/healthz")
    _es_health = _probe("127.0.0.1", 9200, "/_cluster/health")
    _neo4j_health = _probe("127.0.0.1", 7474, "/")
    emulators_up = all(
        h is not None for h in (_qdrant_health, _es_health, _neo4j_health)
    )

    def _count_stores():
        # The hashed embedder loads no model; only the stores are read.
        _app = PersonaApplication.build(
            ApplicationConfig(dataset_profile=profile.name)
        )
        counts = run_io(collect_store_counts, _app) or {}
        try:
            _res = run_io(
                _app.qdrant.transport.request,
                RequestDescriptor("GET", f"/collections/{_app.qdrant.collection}"),
            )["result"]
            counts["dimension"] = _res["config"]["params"]["vectors"]["size"]
        except Exception:
            pass
        return counts

    store_counts = _count_stores() if emulators_up else {}
    index_metadata = (
        json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
        if INDEX_METADATA_PATH.exists()
        else None
    )

    def _badge(ok, name):
        return mo.stat(
            "up" if ok else "down",
            label=name,
            caption="✅" if ok else "⛔",
            bordered=True,
        )

    _rows = [
        mo.md("## 4. Pipeline checks"),
        mo.hstack(
            [
                _badge(_qdrant_health is not None, "Qdrant :6333"),
                _badge(_es_health is not None, "Elasticsearch :9200"),
                _badge(_neo4j_health is not None, "Neo4j :7474"),
            ]
        ),
    ]
    if emulators_up:
        _store_values = [
            store_counts.get("qdrant"),
            store_counts.get("elasticsearch"),
            store_counts.get("neo4j"),
        ]
        _agree = len({v for v in _store_values if v is not None}) == 1 and all(
            v is not None for v in _store_values
        )
        _rows.append(
            mo.hstack(
                [
                    mo.stat(
                        f"{store_counts.get('qdrant') or 0:,}",
                        label=f"Qdrant {profile.qdrant_collection}",
                    ),
                    mo.stat(
                        f"{store_counts.get('elasticsearch') or 0:,}",
                        label=f"Elasticsearch {profile.es_index}",
                    ),
                    mo.stat(
                        f"{store_counts.get('neo4j') or 0:,}",
                        label=f"Neo4j :{profile.node_label}",
                    ),
                    mo.stat(store_counts.get("dimension", "—"), label="vector dims"),
                ]
            )
        )
        _rows.append(
            mo.callout(
                mo.md(
                    "All three stores agree — the index is consistent."
                    if _agree
                    else "Store counts disagree (or the index is empty) — "
                    "rerun the failed shard(s) with "
                    f"`just profile={profile.name} full-index`."
                ),
                kind="success" if _agree else "warn",
            )
        )
    else:
        _rows.append(
            mo.callout(
                mo.md(
                    "Emulators are down. Start them with "
                    "`cd emulator && docker compose up -d`, then re-run this "
                    "notebook. The recorded snapshot below still works."
                ),
                kind="warn",
            )
        )
    if index_metadata:
        _rows.append(
            mo.accordion(
                {
                    f"🗂️ Recorded index metadata (.cache/{INDEX_METADATA_PATH.name})": mo.json(
                        index_metadata
                    )
                }
            )
        )
    mo.vstack(_rows)
    return emulators_up, index_metadata


@app.cell
def _(SNAPSHOT_PATH, json, mo, profile):
    if SNAPSHOT_PATH.exists():
        _snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        _sections = {}
        for _entry in _snapshot.get("queries", []):
            _sections[f"「{_entry['query']}」 — {_entry['latency_ms']}ms"] = (
                mo.ui.table(
                    _entry["results"],
                    selection=None,
                    show_search=False,
                    column_widths={"score": 80, "excerpt": 460},
                    label="",
                )
            )
        snapshot_view = mo.vstack(
            [
                mo.md(
                    f"""
                    ## 5. Recorded full-corpus results — `{profile.name}`

                    Real fused-search results captured against the complete
                    index and committed with the notebook, so they render even
                    with the emulators down.

                    - Captured: `{_snapshot.get("generated_at")}`
                    - Embedder: `{_snapshot.get("embedder")}`
                      ({_snapshot.get("dimension")} dims)
                    - Corpus: {_snapshot.get("corpus_size", 0):,} records
                      (Qdrant/Elasticsearch/Neo4j counts all matched at capture)
                    """
                ),
                mo.accordion(_sections),
            ]
        )
    else:
        snapshot_view = mo.callout(
            mo.md(
                f"No recorded snapshot (`marimo/{SNAPSHOT_PATH.name}`). "
                "Generate one after a full index with "
                f"`just profile={profile.name} catalog-snapshot`."
            ),
            kind="neutral",
        )
    snapshot_view


@app.cell
def _(mo):
    live_query = mo.ui.text(
        value="伝統的な和食を大切にする料理人",
        label="Search query",
        full_width=True,
    )
    live_limit = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Result limit")
    live_run = mo.ui.run_button(label="Search the live index")
    mo.vstack(
        [
            mo.md(
                """
                ## 6. Live search

                Runs the real pipeline (embed query → Qdrant vector search →
                Elasticsearch keyword fusion → Neo4j graph context) against
                whatever the emulators currently hold, using the embedder
                recorded in the index metadata. The first click loads the
                embedding model (a few seconds on GPU).
                """
            ),
            live_query,
            live_limit,
            live_run,
        ]
    )
    return live_limit, live_query, live_run


@app.cell
def _(
    ApplicationConfig,
    EMBEDDER_PRESETS,
    PersonaApplication,
    app_cache,
    emulators_up,
    index_metadata,
    live_limit,
    live_query,
    live_run,
    mo,
    profile,
    run_io,
    time,
):
    mo.stop(
        not live_run.value,
        mo.md("_Click **Search the live index** to run._"),
    )
    mo.stop(
        not emulators_up,
        mo.callout(mo.md("Emulators are down — live search unavailable."), kind="warn"),
    )
    _preset = ((index_metadata or {}).get("embedder") or {}).get("preset") or "hashed"
    if _preset not in EMBEDDER_PRESETS:
        _preset = "hashed"
    if (profile.name, _preset) not in app_cache:
        app_cache[(profile.name, _preset)] = PersonaApplication.build(
            ApplicationConfig(embedder=_preset, dataset_profile=profile.name)
        )
    _app = app_cache[(profile.name, _preset)]

    _t0 = time.perf_counter()
    _results, _stats = run_io(
        _app.search_service.search,
        live_query.value,
        limit=live_limit.value,
        return_stats=True,
    )
    _elapsed_ms = (time.perf_counter() - _t0) * 1000

    _rows = [
        {
            "score": round(float(r.get("score", 0.0)), 4),
            **{f: r.get(f) for f in profile.payload_fields},
            "text": (r.get("text") or "")[:160],
        }
        for r in _results
    ]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(profile.name, label="dataset"),
                    mo.stat(_preset, label="embedder"),
                    mo.stat(f"{_elapsed_ms:.0f}ms", label="latency"),
                    mo.stat(_stats["vector_hits"], label="vector hits"),
                    mo.stat(_stats["keyword_hits"], label="keyword hits"),
                    mo.stat(_stats["context_calls"], label="graph lookups"),
                ]
            ),
            mo.ui.table(
                _rows,
                selection=None,
                show_search=False,
                column_widths={"score": 80, "text": 460},
                label="Fused results",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.accordion(
        {
            "📐 Score semantics": mo.md(
                """
                - Ranking uses `rrf_score` — weighted Reciprocal Rank Fusion
                  over the vector and keyword legs (K=60, production 1:1);
                  `sources` lists the legs that returned each hit.
                - `score` keeps the per-leg meaning: Qdrant cosine similarity
                  when the vector leg saw the hit, otherwise the Elasticsearch
                  `_score`.
                - `return_stats=True` exposes `vector_hits`, `keyword_hits`,
                  `context_calls`, and `results`.
                """
            ),
            "🧩 Pipeline": mo.md(
                """
                `PersonaRepository` → `PersonaIndexer` (compose text → strip
                person names for the embedding input, persona datasets only →
                `embed_documents` in one batched call → the dataset's own
                Qdrant collection / Elasticsearch index / Neo4j label) →
                `PersonaSearchService` (`embed_query` → both legs at fetch
                depth → RRF fusion → graph context for the top hits).
                See `docs/architecture.md`.
                """
            ),
            "🚀 Run it for real": mo.md(
                """
                - `cd emulator && docker compose up -d` — start the backends.
                - `just qa` — index + search the 1k sample from the CLI
                  (`just profile=finance qa` / `just profile=pii qa` for the
                  other datasets).
                - `just full-index` — index the full 1M corpus shard by shard
                  with `ruri-v3-310m` (~1h on a GPU machine);
                  `just profile=<finance|pii> full-index` for the others.
                """
            ),
        }
    )


if __name__ == "__main__":
    app.run()
