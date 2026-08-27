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

    100万ペルソナの検索インデックスから **上位 M 人** を回答者に選び、
    テキスト（＋任意で画像）の問いかけに **なりきり回答** してもらいます。
    推論はローカル Ollama、結果は画面のテーブルと **JSONL**
    （`outputs/persona_panel-*.jsonl`）に記録されます。

    使い方: ① モデルを選ぶ → ② 回答者を決める → ③ 問いかけを作る
    （例題「きのこの山 vs たけのこの里」をそのまま選べます）→
    ④ **パネルに聞く**
    """)


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

    QUESTION_EXAMPLES = {
        "例題: きのこの山 vs たけのこの里": (
            "きのこの山とたけのこの里、どちらが好きですか？理由も添えて答えてください。"
        ),
        "自由入力": "",
    }
    EXAMPLE_IMAGE_PATH = ROOT / "marimo" / "assets" / "kinoko-vs-takenoko.jpg"
    IMAGE_NONE = "なし（テキストのみ）"
    IMAGE_EXAMPLE = "例題画像: きのこ vs たけのこ実物写真（CC0）"
    IMAGE_UPLOAD = "アップロードした画像を使う"

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
        EXAMPLE_IMAGE_PATH,
        IMAGE_EXAMPLE,
        IMAGE_NONE,
        IMAGE_UPLOAD,
        INDEX_METADATA_PATH,
        OLLAMA_HOST,
        OLLAMA_PORT,
        OUTPUT_DIR,
        PersonaApplication,
        QUESTION_EXAMPLES,
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
                mo.md("## ① モデル"),
                model_choice,
                mo.md(
                    "_画像を使う場合は vision 対応モデル（例: `qwen3-vl:*`）を"
                    "選んでください。_"
                ),
            ]
        )
    else:
        model_choice = None
        _view = mo.callout(
            mo.md(
                "Ollama に接続できません（`127.0.0.1:11434`）。"
                "Ollama を起動してからページを再読み込みしてください。"
            ),
            kind="warn",
        )
    _view
    return model_choice, ollama_up


@app.cell
def _(mo):
    panel_query = mo.ui.text(
        value="お菓子やチョコレートが好きな人",
        label="どんなペルソナを集める？（検索クエリの上位が回答者になる）",
        full_width=True,
    )
    panel_m = mo.ui.slider(start=1, stop=20, step=1, value=3, label="何人に聞く？（M）")
    mo.vstack([mo.md("## ② 回答者（パネル）"), panel_query, panel_m])
    return panel_m, panel_query


@app.cell
def _(QUESTION_EXAMPLES, mo):
    question_example = mo.ui.dropdown(
        options=list(QUESTION_EXAMPLES.keys()),
        value=next(iter(QUESTION_EXAMPLES.keys())),
        label="質問プリセット",
    )
    mo.vstack([mo.md("## ③ 問いかけ"), question_example])
    return (question_example,)


@app.cell
def _(QUESTION_EXAMPLES, mo, question_example):
    input_text = mo.ui.text_area(
        value=QUESTION_EXAMPLES.get(question_example.value, ""),
        label="質問テキスト（プリセットを選ぶと自動入力。自由に編集可）",
        full_width=True,
    )
    input_text
    return (input_text,)


@app.cell
def _(IMAGE_EXAMPLE, IMAGE_NONE, IMAGE_UPLOAD, mo):
    image_source = mo.ui.radio(
        options=[IMAGE_NONE, IMAGE_EXAMPLE, IMAGE_UPLOAD],
        value=IMAGE_NONE,
        label="画像入力",
    )
    input_image = mo.ui.file(
        filetypes=[".png", ".jpg", ".jpeg", ".webp"],
        kind="area",
        label="（「アップロードした画像を使う」を選んだ場合はここに投入）",
    )
    mo.vstack([image_source, input_image])
    return image_source, input_image


@app.cell
def _(
    EXAMPLE_IMAGE_PATH,
    IMAGE_EXAMPLE,
    IMAGE_UPLOAD,
    image_source,
    input_image,
    mo,
):
    resolved_image = None
    if image_source.value == IMAGE_EXAMPLE and EXAMPLE_IMAGE_PATH.exists():
        resolved_image = (EXAMPLE_IMAGE_PATH.name, EXAMPLE_IMAGE_PATH.read_bytes())
    elif image_source.value == IMAGE_UPLOAD and input_image.value:
        _f = input_image.value[0]
        resolved_image = (_f.name, _f.contents)

    if resolved_image:
        _preview = mo.vstack(
            [
                mo.md(
                    f"この画像を見せます: `{resolved_image[0]}` "
                    f"({len(resolved_image[1]):,} bytes)"
                ),
                mo.image(resolved_image[1], width=360),
            ]
        )
    else:
        _preview = mo.md("_画像なし（テキストのみで質問します）_")
    _preview
    return (resolved_image,)


@app.cell
def _(mo):
    run_panel = mo.ui.run_button(label="🗳️ パネルに聞く")
    mo.vstack([mo.md("## ④ 実行"), run_panel])
    return (run_panel,)


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
    input_text,
    json,
    mo,
    model_choice,
    ollama_up,
    panel_app_cache: dict,
    panel_m,
    panel_query,
    resolved_image,
    run_io,
    run_panel,
    time,
):
    mo.stop(
        not run_panel.value,
        mo.md("_「パネルに聞く」を押すと実行します（1 人あたり数秒〜数十秒）。_"),
    )
    mo.stop(
        not ollama_up,
        mo.callout(mo.md("Ollama が起動していません。"), kind="warn"),
    )
    mo.stop(
        not input_text.value.strip(),
        mo.callout(mo.md("質問テキストを入力してください。"), kind="warn"),
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
        [base64.b64encode(resolved_image[1]).decode("ascii")]
        if resolved_image
        else None
    )
    _image_name = resolved_image[0] if resolved_image else None
    _model = model_choice.value
    _ollama = SimpleHttpTransport(OLLAMA_HOST, OLLAMA_PORT, timeout=600.0)

    _records = []
    _started_all = time.perf_counter()
    for _rank, _persona in enumerate(
        mo.status.progress_bar(_panelists, title="パネル回答を生成中", show_eta=True),
        start=1,
    ):
        _fields = _persona.get("persona_fields") or {}
        _profile = (_fields.get("persona") or _persona.get("text") or "").strip()
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
                "persona": _profile,
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
            mo.md("## 結果"),
            mo.hstack(
                [
                    mo.stat(_model, label="model"),
                    mo.stat(len(_records), label="回答者"),
                    mo.stat(f"{_total_s:.1f}s", label="所要時間"),
                    mo.stat("あり" if _image_name else "なし", label="画像"),
                ]
            ),
            mo.ui.table(
                [
                    {
                        "#": record["rank"],
                        "地域": f"{record['prefecture']}（{record['region']}）",
                        "ペルソナ": record["persona"],
                        "結論": record["verdict"],
                        "回答": record["answer"],
                        "ms": record["latency_ms"],
                    }
                    for record in _records
                ],
                selection=None,
                show_search=False,
                wrapped_columns=["ペルソナ", "回答"],
                column_widths={
                    "#": 40,
                    "地域": 130,
                    "ペルソナ": 300,
                    "結論": 110,
                    "回答": 340,
                    "ms": 70,
                },
                page_size=20,
                label=f"パネル回答（JSONL: outputs/{_jsonl_path.name}）",
            ),
            mo.download(
                data=(_jsonl_text + "\n").encode("utf-8"),
                filename=_jsonl_path.name,
                label="JSONL をダウンロード",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.accordion(
        {
            "📄 JSONL フォーマット": mo.md(
                """
                1 行 = 1 ペルソナの回答。フィールド:
                `timestamp, model, panel_query, input_text, image, rank, score,
                uuid, prefecture, region, persona, verdict, answer, latency_ms`。
                出力先は `outputs/persona_panel-<UTC時刻>.jsonl`（git 管理外）。
                """
            ),
            "⏱️ レイテンシの目安": mo.md(
                """
                1 回答あたり数秒〜数十秒（モデルサイズ依存。初回はモデルロードで
                +数十秒）。M を大きくする前に小さな M で試してください。
                """
            ),
            "🖼️ 画像とライセンス": mo.md(
                """
                画像を使う場合は vision 対応モデル（`qwen3-vl:*` など）を選択して
                ください。例題画像は Wikimedia Commons の CC0 写真です
                （出典・注意事項は `marimo/assets/CREDITS.md`）。
                """
            ),
        }
    )


if __name__ == "__main__":
    app.run()
