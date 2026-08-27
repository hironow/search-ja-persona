import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="search-ja-persona persona panel")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 🗳️ Persona Panel — 100万人の街頭アンケート

    Fused search で **上位 M 件のペルソナ**を選び、テキスト（＋任意で画像）の
    入力について、各ペルソナに **なりきり回答** させます。推論はローカルの
    Ollama、結果は **JSONL** として `outputs/` に書き出されます。

    1. パネル選定 — 検索クエリで回答者 M 人を選ぶ
    2. 入力 — 質問テキストと、任意の画像（画像には vision 対応モデルが必要）
    3. 実行 — 1 人ずつ Ollama で回答を生成し、JSONL に記録
    """)
    return


@app.cell
def _():
    import base64
    import concurrent.futures
    import json
    import sys
    import time
    from datetime import UTC, datetime
    from pathlib import Path

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

    from search_ja_persona.application import ApplicationConfig, PersonaApplication
    from search_ja_persona.embeddings import EMBEDDER_PRESETS
    from search_ja_persona.services import RequestDescriptor, SimpleHttpTransport

    def run_io(fn, *args, **kwargs):
        # The HTTP transport spins up its own event loop, which clashes with
        # marimo's running loop; run these calls in a worker thread.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            return _ex.submit(fn, *args, **kwargs).result()

    OUTPUT_DIR = ROOT / "outputs"
    INDEX_METADATA_PATH = ROOT / ".cache" / "index_metadata.json"
    OLLAMA_HOST = "127.0.0.1"
    OLLAMA_PORT = 11434
    DEFAULT_MODEL = "huihui_ai/Qwen3.8-abliterated:latest"
    panel_app_cache: dict = {}
    ANSWER_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "verdict": {"type": "string"},
        },
        "required": ["answer", "verdict"],
    }
    return (
        ANSWER_SCHEMA,
        ApplicationConfig,
        DEFAULT_MODEL,
        EMBEDDER_PRESETS,
        INDEX_METADATA_PATH,
        OLLAMA_HOST,
        OLLAMA_PORT,
        OUTPUT_DIR,
        PersonaApplication,
        RequestDescriptor,
        SimpleHttpTransport,
        UTC,
        base64,
        datetime,
        json,
        panel_app_cache,
        run_io,
        time,
    )


@app.cell
def _(
    DEFAULT_MODEL,
    OLLAMA_HOST,
    OLLAMA_PORT,
    RequestDescriptor,
    SimpleHttpTransport,
    mo,
    run_io,
):
    try:
        _tags = run_io(
            SimpleHttpTransport(OLLAMA_HOST, OLLAMA_PORT, timeout=3.0).request,
            RequestDescriptor("GET", "/api/tags"),
        )
        ollama_models = [m["name"] for m in _tags.get("models", [])]
    except Exception:
        ollama_models = []
    ollama_up = bool(ollama_models)

    if ollama_up:
        model_choice = mo.ui.dropdown(
            options=ollama_models,
            value=DEFAULT_MODEL if DEFAULT_MODEL in ollama_models else ollama_models[0],
            label="Ollama モデル（インストール済みから選択）",
        )
        _view = mo.vstack(
            [
                model_choice,
                mo.md(
                    "_画像を入力に使う場合は vision 対応モデル"
                    "（例: `qwen3-vl:*`）を選んでください。_"
                ),
            ]
        )
    else:
        model_choice = None
        _view = mo.callout(
            mo.md(
                "Ollama に接続できません（`127.0.0.1:11434`）。"
                "Ollama を起動してからノートブックを再実行してください。"
            ),
            kind="warn",
        )
    _view
    return model_choice, ollama_up


@app.cell
def _(mo):
    panel_query = mo.ui.text(
        value="お菓子やチョコレートが好きな人",
        label="パネル選定クエリ（この検索の上位 M 人が回答者になる）",
        full_width=True,
    )
    panel_m = mo.ui.slider(start=1, stop=20, step=1, value=3, label="M（回答者数）")
    input_text = mo.ui.text_area(
        value="きのこの山とたけのこの里、どちらが好きですか？理由も添えて答えてください。",
        label="質問（テキスト入力）",
        full_width=True,
    )
    input_image = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        kind="area",
        label="画像入力（任意・マルチモーダル）",
    )
    run_panel = mo.ui.run_button(label="パネルに聞く")
    mo.vstack([panel_query, panel_m, input_text, input_image, run_panel])
    return input_image, input_text, panel_m, panel_query, run_panel


@app.cell
def _(input_image, mo):
    if input_image.value:
        _f = input_image.value[0]
        image_preview = mo.vstack(
            [
                mo.md(f"添付画像: `{_f.name}` ({len(_f.contents):,} bytes)"),
                mo.image(_f.contents, width=320),
            ]
        )
    else:
        image_preview = mo.md("_画像なし（テキストのみで質問します）_")
    image_preview
    return


@app.cell
def _(
    ANSWER_SCHEMA,
    ApplicationConfig,
    EMBEDDER_PRESETS,
    INDEX_METADATA_PATH,
    OLLAMA_HOST,
    OLLAMA_PORT,
    OUTPUT_DIR,
    PersonaApplication,
    RequestDescriptor,
    SimpleHttpTransport,
    UTC,
    base64,
    datetime,
    input_image,
    input_text,
    json,
    mo,
    model_choice,
    ollama_up,
    panel_app_cache: dict,
    panel_m,
    panel_query,
    run_io,
    run_panel,
    time,
):
    mo.stop(
        not run_panel.value,
        mo.md("_「パネルに聞く」を押すと実行します（M×数十秒かかります）。_"),
    )
    mo.stop(
        not ollama_up,
        mo.callout(mo.md("Ollama が起動していません。"), kind="warn"),
    )

    _preset = "ruri-v3-310m"
    if INDEX_METADATA_PATH.exists():
        _meta = json.loads(INDEX_METADATA_PATH.read_text(encoding="utf-8"))
        _recorded = (_meta.get("embedder") or {}).get("preset")
        if _recorded in EMBEDDER_PRESETS:
            _preset = _recorded
    if _preset not in panel_app_cache:
        panel_app_cache[_preset] = PersonaApplication.build(
            ApplicationConfig(embedder=_preset)
        )
    _search_app = panel_app_cache[_preset]

    _panelists = run_io(_search_app.search, panel_query.value, limit=panel_m.value)
    mo.stop(
        not _panelists,
        mo.callout(
            mo.md("検索結果が 0 件でした。クエリを変えてください。"), kind="warn"
        ),
    )

    _images_b64 = (
        [base64.b64encode(input_image.value[0].contents).decode("ascii")]
        if input_image.value
        else None
    )
    _image_name = input_image.value[0].name if input_image.value else None
    _model = model_choice.value
    _ollama = SimpleHttpTransport(OLLAMA_HOST, OLLAMA_PORT, timeout=600.0)

    _records = []
    _started_all = time.perf_counter()
    for _rank, _persona in enumerate(
        mo.status.progress_bar(_panelists, title="パネル回答を生成中", show_eta=True),
        start=1,
    ):
        _system = (
            "あなたは以下のペルソナの人物です。この人物になりきって、一人称で、"
            "その人らしい視点・語彙で回答してください。回答(answer)は日本語で"
            "200字以内。verdict にはあなたの結論をごく短く（例: 商品名や"
            "賛成/反対など）記入してください。\n\n"
            f"{_persona.get('text', '')}"
        )
        _user_message = {"role": "user", "content": input_text.value}
        if _images_b64:
            _user_message["images"] = _images_b64
        _t0 = time.perf_counter()
        _body = {
            "model": _model,
            "messages": [
                {"role": "system", "content": _system},
                _user_message,
            ],
            "stream": False,
            "format": ANSWER_SCHEMA,
            # Thinking models (Qwen3 系) spend the token budget on hidden
            # reasoning and return empty content; ask for direct answers
            # first and fall back for models that reject the think flag.
            "think": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }
        try:
            try:
                _resp = run_io(
                    _ollama.request, RequestDescriptor("POST", "/api/chat", body=_body)
                )
            except RuntimeError:
                _body_no_think = {k: v for k, v in _body.items() if k != "think"}
                _resp = run_io(
                    _ollama.request,
                    RequestDescriptor("POST", "/api/chat", body=_body_no_think),
                )
            _raw = _resp.get("message", {}).get("content", "")
            try:
                _parsed = json.loads(_raw)
            except json.JSONDecodeError:
                _parsed = {"answer": _raw, "verdict": "(parse-error)"}
        except Exception as _exc:  # noqa: BLE001 - keep the panel going
            _parsed = {"answer": f"(error: {_exc})", "verdict": "(error)"}
        _records.append(
            {
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": _model,
                "panel_query": panel_query.value,
                "input_text": input_text.value,
                "image": _image_name,
                "rank": _rank,
                "score": round(float(_persona.get("score", 0.0)), 4),
                "uuid": _persona.get("uuid"),
                "prefecture": _persona.get("prefecture"),
                "region": _persona.get("region"),
                "verdict": _parsed.get("verdict", ""),
                "answer": _parsed.get("answer", ""),
                "latency_ms": round((time.perf_counter() - _t0) * 1000),
            }
        )
    _total_s = time.perf_counter() - _started_all

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    _jsonl_path = OUTPUT_DIR / f"persona_panel-{_stamp}.jsonl"
    _jsonl_text = "\n".join(
        json.dumps(record, ensure_ascii=False) for record in _records
    )
    _jsonl_path.write_text(_jsonl_text + "\n", encoding="utf-8")

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(_model, label="model"),
                    mo.stat(len(_records), label="panelists"),
                    mo.stat(f"{_total_s:.1f}s", label="total"),
                    mo.stat("あり" if _image_name else "なし", label="画像"),
                ]
            ),
            mo.ui.table(
                [
                    {
                        "rank": record["rank"],
                        "prefecture": record["prefecture"],
                        "verdict": record["verdict"],
                        "answer": record["answer"][:160],
                        "ms": record["latency_ms"],
                    }
                    for record in _records
                ],
                selection=None,
                show_search=False,
                column_widths={"verdict": 130, "answer": 420},
                label=f"パネル回答（JSONL: {_jsonl_path.relative_to(_jsonl_path.parents[1])}）",
            ),
            mo.download(
                data=(_jsonl_text + "\n").encode("utf-8"),
                filename=_jsonl_path.name,
                label="JSONL をダウンロード",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📄 JSONL フォーマット": mo.md(
                """
                1 行 = 1 ペルソナの回答。フィールド:
                `timestamp, model, panel_query, input_text, image, rank, score,
                uuid, prefecture, region, verdict, answer, latency_ms`。
                出力先は `outputs/persona_panel-<UTC時刻>.jsonl`（git 管理外）。
                """
            ),
            "⏱️ レイテンシの目安": mo.md(
                """
                1 回答あたり数秒〜数十秒（モデルサイズ依存。初回はモデルロードで
                +数十秒）。M を大きくする前に小さな M で試してください。
                """
            ),
            "🖼️ 画像入力": mo.md(
                """
                画像を添付した場合は vision 対応モデル（`qwen3-vl:*` など）を
                選択してください。テキスト専用モデルは画像を無視するか、
                エラーになります。
                """
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
