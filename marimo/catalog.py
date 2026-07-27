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

        A runnable tour of everything this system does. Execute top to bottom:

        1. **Data model** — the persona schema and text fields.
        2. **Corpus exploration** — DuckDB SQL over the bundled QA parquet.
        3. **Embedders** — the pluggable vectorization backends.
        4. **Live pipeline** — index → fused vector + keyword + graph search
           against the local emulators.

        The offline sections always run. The live pipeline lights up when the
        emulators are running (`cd emulator && docker compose up -d`); otherwise
        it degrades gracefully with a note.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Make the project importable regardless of how the notebook is launched
    # (marimo does not put the repo root on sys.path the way `python -m` does).
    _root = next(
        (
            base
            for base in [Path.cwd(), *Path.cwd().parents]
            if (base / "search_ja_persona").is_dir()
        ),
        Path.cwd(),
    )
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from search_ja_persona.application import ApplicationConfig, PersonaApplication
    from search_ja_persona.embeddings import EMBEDDER_PRESETS, HashedNgramEmbedder
    from search_ja_persona.persona_fields import PERSONA_TEXT_FIELDS
    from search_ja_persona.repository import PersonaRepository
    from search_ja_persona.services import RequestDescriptor, SimpleHttpTransport

    import concurrent.futures

    def run_io(fn, *args, **kwargs):
        # The emulator HTTP transport spins up its own event loop, which clashes
        # with marimo's already-running loop; run these calls in a worker thread
        # (a fresh thread has no running loop, so run_until_complete works).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            return _ex.submit(fn, *args, **kwargs).result()

    QA_PARQUET = Path("qa_samples/qa_sample.parquet")
    return (
        ApplicationConfig,
        EMBEDDER_PRESETS,
        HashedNgramEmbedder,
        PERSONA_TEXT_FIELDS,
        PersonaApplication,
        PersonaRepository,
        QA_PARQUET,
        RequestDescriptor,
        SimpleHttpTransport,
        run_io,
    )


@app.cell
def _(PERSONA_TEXT_FIELDS, QA_PARQUET, mo):
    mo.md(
        f"""
        ## 1. Persona data model

        Records stream from parquet via `PersonaRepository`. Each persona carries
        metadata (`uuid`, `prefecture`, `region`, `occupation`, `age`, `sex`) plus
        **{len(PERSONA_TEXT_FIELDS)} free-text fields** that get aggregated and
        embedded:

        {"".join(f"- `{f}`\n" for f in PERSONA_TEXT_FIELDS)}

        Bundled sample: `{QA_PARQUET}` (1k rows, no download required).
        """
    )
    return


@app.cell
def _(PersonaRepository, QA_PARQUET, mo):
    _repo = PersonaRepository([QA_PARQUET])
    personas = list(_repo.iter_personas(limit=200))

    _preview = [
        {
            "uuid": str(p.get("uuid", ""))[:8],
            "prefecture": p.get("prefecture"),
            "region": p.get("region"),
            "occupation": p.get("occupation"),
            "age": p.get("age"),
            "sex": p.get("sex"),
            "persona": (p.get("persona") or "")[:120],
        }
        for p in personas
    ]
    persona_table = mo.ui.table(
        _preview,
        show_search=True,
        pagination=True,
        page_size=8,
        selection=None,
        column_widths={"uuid": 90, "persona": 420},
        label=f"First {len(personas)} personas from the QA sample",
    )
    persona_table
    return (personas,)


@app.cell
def _(QA_PARQUET, mo):
    mo.md(
        """
        ## 2. Corpus exploration (DuckDB SQL)

        marimo SQL cells run DuckDB directly over the parquet — no ingestion,
        no server. Here is the prefecture distribution of the sample.
        """
    )
    prefecture_distribution = mo.sql(
        f"""
        SELECT prefecture,
               region,
               count(*) AS personas
        FROM read_parquet('{QA_PARQUET}')
        WHERE prefecture IS NOT NULL
        GROUP BY prefecture, region
        ORDER BY personas DESC
        LIMIT 15
        """
    )
    return


@app.cell
def _(EMBEDDER_PRESETS, mo):
    mo.md(
        """
        ## 3. Embedder backends

        The `Embedder` protocol (`dimension` + `embed(text)`) is satisfied by a
        hashed n-gram encoder (offline), SentenceTransformers, and FastEmbed
        (ONNX). Pick a preset and embed some text below. Only `hashed` runs with
        zero downloads; the model-backed presets fetch weights on first use.
        """
    )
    embedder_choice = mo.ui.dropdown(
        options=list(EMBEDDER_PRESETS.keys()),
        value="hashed",
        label="Embedder preset",
    )
    embed_text = mo.ui.text(
        value="高齢者介護の経験豊富なケアマネージャー",
        label="Text to embed",
        full_width=True,
    )
    mo.vstack([embedder_choice, embed_text])
    return embed_text, embedder_choice


@app.cell
def _(EMBEDDER_PRESETS, HashedNgramEmbedder, embed_text, embedder_choice, mo):
    _preset = EMBEDDER_PRESETS.get(embedder_choice.value, {})
    _type = _preset.get("type", embedder_choice.value)

    if _type == "hashed":
        _emb = HashedNgramEmbedder(dimension=256, ngram_sizes=(2, 3))
        _vector = _emb.embed(embed_text.value)
        _dim = _emb.dimension
        _preview = ", ".join(f"{v:.3f}" for v in _vector[:8])
        embed_view = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(embedder_choice.value, label="preset"),
                        mo.stat(_dim, label="dimension"),
                        mo.stat(
                            f"{sum(1 for v in _vector if v):d}",
                            label="non-zero buckets",
                        ),
                    ]
                ),
                mo.md(f"First 8 components: `[{_preview}, ...]`"),
            ]
        )
    else:
        _model = _preset.get("model", "?")
        embed_view = mo.callout(
            mo.md(
                f"Preset **{embedder_choice.value}** uses `{_type}` backend "
                f"(`{_model}`). It downloads model weights on first use — run it "
                f'from a shell (`just qa embedder="{embedder_choice.value}"`) '
                f"rather than inline here."
            ),
            kind="neutral",
        )
    embed_view
    return


@app.cell
def _(RequestDescriptor, SimpleHttpTransport, mo, run_io):
    def _probe(host, port, path, auth=None):
        try:
            transport = SimpleHttpTransport(host, port, timeout=1.5, auth=auth)
            run_io(transport.request, RequestDescriptor("GET", path))
            return True
        except Exception:
            return False

    qdrant_ok = _probe("localhost", 6333, "/healthz")
    es_ok = _probe("localhost", 9200, "/_cluster/health")
    neo4j_ok = _probe("localhost", 7474, "/")
    emulators_up = qdrant_ok and es_ok and neo4j_ok

    def _badge(ok, name):
        return mo.stat(
            "up" if ok else "down",
            label=name,
            caption="✅" if ok else "⛔",
            bordered=True,
        )

    mo.vstack(
        [
            mo.md("## 4. Live pipeline — emulator status"),
            mo.hstack(
                [
                    _badge(qdrant_ok, "Qdrant :6333"),
                    _badge(es_ok, "Elasticsearch :9200"),
                    _badge(neo4j_ok, "Neo4j :7474"),
                ]
            ),
            mo.callout(
                mo.md(
                    "All three emulators are up — the live pipeline below is active."
                ),
                kind="success",
            )
            if emulators_up
            else mo.callout(
                mo.md(
                    "Emulators are down. Start them with "
                    "`cd emulator && docker compose up -d`, then re-run this "
                    "notebook to see live indexing and search."
                ),
                kind="warn",
            ),
        ]
    )
    return (emulators_up,)


@app.cell
def _(
    ApplicationConfig,
    PersonaApplication,
    QA_PARQUET,
    emulators_up,
    mo,
    personas,
    run_io,
):
    if emulators_up:
        _config = ApplicationConfig(
            embedder="hashed",
            vector_dimension=256,
            qdrant_collection="catalog_personas",
            es_index="catalog_personas",
        )
        catalog_app = PersonaApplication.build(_config)
        run_io(catalog_app.index, [QA_PARQUET], batch_size=64, limit=40)
        index_view = mo.callout(
            mo.md(
                f"Indexed **40** personas (of {len(personas)} loaded) into the "
                "`catalog_personas` collection/index with the offline `hashed` "
                "embedder. Neo4j nodes are merged by `uuid` (idempotent)."
            ),
            kind="success",
        )
    else:
        catalog_app = None
        index_view = mo.callout(
            mo.md("Live indexing skipped — emulators are down."), kind="warn"
        )
    index_view
    return (catalog_app,)


@app.cell
def _(mo):
    search_query = mo.ui.text(
        value="介護 経験 マネージャー",
        label="Search query",
        full_width=True,
    )
    search_limit = mo.ui.slider(start=1, stop=10, step=1, value=5, label="Result limit")
    mo.vstack([search_query, search_limit])
    return search_limit, search_query


@app.cell
def _(catalog_app, mo, run_io, search_limit, search_query):
    if catalog_app is None:
        search_view = mo.md("_Start the emulators and re-run to search._")
    else:
        _results, _stats = run_io(
            catalog_app.search_service.search,
            search_query.value,
            limit=search_limit.value,
            return_stats=True,
        )
        _rows = [
            {
                "uuid": str(r.get("uuid", ""))[:8],
                "score": round(float(r.get("score", 0.0)), 4),
                "prefecture": r.get("prefecture"),
                "region": r.get("region"),
                "text": (r.get("text") or "")[:160],
            }
            for r in _results
        ]
        search_view = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(_stats["vector_hits"], label="Qdrant vector hits"),
                        mo.stat(_stats["keyword_hits"], label="Elasticsearch hits"),
                        mo.stat(_stats["context_calls"], label="Neo4j context lookups"),
                        mo.stat(len(_results), label="fused results"),
                    ]
                ),
                mo.ui.table(
                    _rows,
                    selection=None,
                    show_search=False,
                    column_widths={"uuid": 90, "text": 460},
                    label="Fused results (vector shortlist + keyword fallback + graph context)",
                ),
            ]
        )
    search_view
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📐 Score semantics": mo.md(
                """
                - **Vector hits** (Qdrant): `score` = cosine similarity (0–1).
                - **Keyword fallback** (Elasticsearch): `score` = mapped `_score`
                  for personas not in the vector shortlist.
                - `return_stats=True` exposes `vector_hits`, `keyword_hits`,
                  `context_calls`, and `results`.
                """
            ),
            "🧩 Pipeline": mo.md(
                """
                `PersonaRepository` → `PersonaIndexer` (compose text → embed →
                Qdrant / Elasticsearch / Neo4j) → `PersonaSearchService`
                (embed query → vector search → keyword fusion → graph context).
                See `docs/architecture.md`.
                """
            ),
            "🚀 Run it for real": mo.md(
                """
                - `cd emulator && docker compose up -d` — start the backends.
                - `just qa` — index + search the 1k sample from the CLI.
                - `just qa-index embedder="mini-lm"` — swap in a real model.
                """
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
