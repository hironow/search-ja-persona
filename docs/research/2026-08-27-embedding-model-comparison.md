# 日本語 embedding モデル比較実測 (2026-08-27)

**Snapshot date:** 2026-08-27
**Environment:** Windows 11 / RTX 4090 24GB / fp16 + SDPA (flash-attn は
Windows でビルド不可) / sentence-transformers 6.0.0 / torch 2.13.0+cu130
**Corpus sample:** `qa_samples/qa_sample.parquet` 由来の実データ 1,024 件
(Nemotron-Personas-Japan、6 フィールド連結テキスト。mean 681 chars /
median 650 / p95 1,004)
**Provenance:** 本セッションの first-party 実測。独立した 2 系統の計測が
±6% 以内で一致。JMTEB 等の公表スコアはローカル再検証していない。

## 結論

**採用: `cl-nagoya/ruri-v3-310m`**(768 次元、Apache-2.0、`検索クエリ: `/
`検索文書: ` プレフィックス、encode batch 16)。実験イテレーション向けの
高速代替として `ruri-v3-130m`(512 次元、約 2.1 倍速)。
multilingual-e5 系は本コーパスで **22–23% の文書が 512 トークン切り捨て**
となり、速度同等でも品質面で失格級のハンデを負う。

## スループット実測 (1,024 実文書、length-sorted batching)

| Model | Params | Dim | max_seq | tok/doc | Best bs | docs/s | hrs/1M | VRAM |
|---|---|---|---|---|---|---|---|---|
| ruri-v3-130m | 132M | 512 | 8192 | 328 | 32 | 803 | 0.35 | 0.8 GB |
| **ruri-v3-310m** | 315M | 768 | 8192 | 328 | 16 | 384 | 0.72 | 1.0 GB |
| multilingual-e5-large | 560M | 1024 | 512 | 427 (22% 切捨て) | 32 | 366 | 0.76 | 1.5 GB |
| embeddinggemma-300m | 308M | 768 | 2048 | 426 | 16 | 319 | 0.87 | 1.2 GB |
| pplx-embed-v1-0.6b | 596M | 1024 | 32768 | 530 | 8 | 118 | 2.36 | 1.5 GB |
| sarashina-embedding-v2-1b | 1.2B | 1792 | 8192 | 684 | 8 | 64.6 | 4.30 | 4.0 GB |

I/O とベクトル書き込みで +10–20% を見込む。パイプライン実測: 1,000 件の
index (embed + Qdrant/ES/Neo4j 書き込み、モデルロード込み) が 9.0 秒。

## トークナイザ効率 (実測 2,000 件)

| Tokenizer | chars/token | >512 トークン率 |
|---|---|---|
| Ruri v3 (102k vocab) | 2.08 | 2.9% |
| multilingual-e5 / XLM-R (250k) | 1.51 | 23.0% |
| embeddinggemma (262k) | 1.58 | 19.3% |
| pplx-embed / Qwen3 (152k) | 1.27 | 49.9% |
| sarashina-v2-1b (102k) | 1.00 | 87.1% |

Ruri のトークナイザは XLM-R 比で 38% 効率が高い。Ruri の優位は
「ModernBERT だから速い」ではなく (512 トークン域では ModernBERT-base は
BERT より遅い — arXiv:2412.13663)、トークナイザ効率と無切り捨てに由来する。
e5-large が 560M params で 310M の Ruri と同速なのは、e5 のパラメータの
約半分 (250k×1024≈256M) が計算を伴わない埋め込み表だから。

## 検索品質の目視評価 (同一 1k コーパス、5 クエリ × top-3)

| クエリ | mini-lm (旧既定) | e5-small+prefix | ruri-v3-310m (実パイプライン) |
|---|---|---|---|
| 高齢者介護の経験豊富なマネージャー | 1/3 | 3/3 | 3/3 (ケアマネ昇格志望まで一致) |
| 登山やアウトドアが好きなエンジニア | 0/3 | 2.5/3 | 3/3 全員エンジニア職 |
| 伝統的な和食を大切にする料理人 | 0/3 | 0.5/3 | 0.5/3 (母集団に料理人不在の疑い) |
| 海外旅行が趣味で語学に関心がある人 | 0/3 | 0.5/3 | 2/3 (多言語調整の旅行 PM がヒット) |
| 子育てと仕事を両立している看護師 | 0/3 | 3/3 | 3/3 (子育て軸まで一致) |

mini-lm (all-MiniLM-L6-v2、英語モデル) は日本語の意味をほぼ捉えられず
表層語彙の一致に退化する。プレフィックス (`query:`/`passage:`、Ruri は
`検索クエリ: `/`検索文書: `) は e5/Ruri の品質前提であり、パイプラインは
preset 宣言で自動適用する (`search_ja_persona/embeddings.py` の
`EMBEDDER_PRESETS`)。

## 運用上の知見

- **バッチサイズは大きいほど遅い。** 全モデル bs=8–32 が最速。
  ruri-310m は bs=16→128 で約 20% 劣化。sentence-transformers は長さ順
  ソートで padding を最小化済みのため、大バッチはメモリ圧迫しか生まない。
  preset に encode_batch_size (310m=16 / 130m=32) として固定済み。
- **Windows WDDM は VRAM 超過で OOM 例外を出さずホスト RAM に静かに
  スピルしてハングする** (1.2B モデル bs=128 で発生、強制終了が必要)。
  OOM 捕捉に頼らずバッチ上限を明示すること。
- **sarashina-embedding-v2-1b は transformers 5.x 系で日本語 1 文字 =
  1 トークンに退化する** (fast/slow 両トークナイザで再現。語彙自体は
  54,973 個の複数文字日本語トークンを含み正常)。公称ベンチの再現には
  transformers の版固定検証が必要。非商用ライセンスでもあり除外。
- **pplx-embed は fp16 で出力が全て NaN** (bf16 必須)。日本語トークナイザ
  効率も低く (1.27 chars/token)、総合 4.4 倍遅。
- torch の CUDA 版は Windows では PyPI に無く、`pytorch-cu130` の explicit
  index 経由 (pyproject の `tool.uv.sources`)。

## 生データ

計測スクリプトと raw JSON はセッション scratchpad (リポジトリ外)。
トークン統計の要点: Ruri v3 で mean 327.8 tok / median 313 / p95 478 /
max 771 (vocab 102,400、model_max_length 8,192)。
