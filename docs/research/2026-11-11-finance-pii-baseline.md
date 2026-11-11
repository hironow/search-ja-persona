# finance / pii 全量索引の baseline と japan 不変の検証 (2026-11-11)

**Snapshot date:** 2026-11-11
**Environment:** Windows 11 / RTX 4090 24GB / Docker Desktop の emulator（Qdrant 1.20.0、Elasticsearch 9.4.3、Neo4j 2025-community）。埋め込みは `ruri-v3-310m`（768d）。
**Harness:** `just profile=<finance|pii> full-index` / `eval --check-thresholds` / `diagnose` / `eval-names`、`just store-fingerprint`

## 結論

- 3 データセットの 3 ストア件数がすべて一致した: japan 1,000,000×3、finance 1,000,000×3、pii 100,000×3。カナリア 3 本も全データセットで OK。
- japan（V1）は不変だった。作業前（05:34）と作業後（08:36）の `store_fingerprint` が完全一致した（JSON ファイルの sha256 も同一）。ES `personas` の `index_total` は前後とも 0（emulator 起動後、V1 への書き込みは 1 件もない）。eval も全指標・24 クエリ単位で前後一致した。
- pii の検索品質は高い（golden basic 0.900 / hard 0.880）。finance は japan より大きく低い（basic 0.457 / hard 0.425）。

## 索引

| データセット | 件数 | 所要時間 | 備考 |
|---|---|---|---|
| pii | 100,000 | 13 分 54 秒 | `data.parquet` 1 ファイル |
| finance | 1,000,000 | 150 分 54 秒 | 8 シャード。WSL 上の vLLM（Qwen3.8-27B）と GPU を共有し、埋め込み速度は 35〜130 件/秒で揺れた |

finance の所要時間は japan の実績（約 50 分）の 3 倍。テキストが 9 フィールド（japan は 6）で長いことと、GPU の共有が理由。

## 評価

| 指標 | japan（参考） | finance | pii |
|---|---|---|---|
| golden basic precision@5 | 0.917 | 0.457 | 0.900 |
| golden hard precision@5 | 0.650 | 0.425 | 0.880 |
| filtered（japan/finance: 都道府県、pii: domain） | 0.950 | 0.750 | 1.000 |
| self-retrieval recall@1 / @10 | 0.92 / 1.00 | 0.91 / 1.00 | 対象外 |
| lookup recall@1 / @10（n=40） | 0.725 / 0.900（名前、2026-08-27 の記録。今回は未測定） | 0.825 / 0.950（名前） | 0.875 / 1.000（PII 値） |
| 述語のランダム基準率（basic / hard） | — | 0.030 / 0.026 | 0.024 / 0.018 |
| context coverage / keyword 寄与 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

- pii は self-retrieval を測らない。文書の説明文（`document_description`）は約 10 文書で共有されるので、自分の文書を一意に引けない。代わりに、1 文書にしか現れない PII 値 40 件（domain ごとに 5 件）の lookup で測る。
- finance と pii の品質バーは未裁定（`docs/decision-queue.md`）。今は報告のみで、gate はカナリアだけ。

### finance の述語修正（測定後）

初回の basic は 0.414 だった。「毎月コツコツ積み立てて資産を増やしている人」の述語が漢字の「積立」だけを見ていて、本文の動詞「積み立てる」を取りこぼしていた（上位 5 件はどれも毎月の積み立てを書いていた）。同じ語の表記ゆれなので「積み立」を足した。1 シャード全体での基準率は 0.5% から 6.7% になり、ほかの述語と同じ水準。この 1 件の修正で basic は 0.414 から 0.457 になった。ほかの述語は変えていない。

### finance が低い理由（diagnose の観察）

- 0 点のクエリ: 保険の見直し、奨学金の返済、共働き＋住宅ローン＋教育費、退職金の運用、相続。上位は「保険や資産を持つ人」で、見直しや相続の記述はない。検索の取りこぼしで、述語の問題ではない。
- 奨学金クエリでは keyword レグが名前の「奨太」に一致した（CJK を 1 文字単位で扱う BM25 の限界）。japan の名前汚染と同じ型が、今度は keyword 側で起きている。
- vector 単独は fused より高い場合がある（basic: vector 0.471 / fused 0.457。「大阪で自営業」は vector 1.00 / fused 0.40、「福岡県で年金生活」は vector 1.00 / fused 0.40）。keyword レグが fused の順位を下げている。
- finance のペルソナは家計の記述が多く似通っている。語彙で区別できる特徴（相続、奨学金など）の出現率は 0.02〜0.4%（1 シャード 125,000 件中 24〜501 件）と小さい。

## 改善候補（別の作業単位）

- finance の RRF 重みの A/B（vector 寄り）。japan の重み 1:1 は requester の裁定なので、データセット別の重みは新しい判断になる。
- 名前を keyword フィールドに分けて、BM25 の本文から外す（japan の name lookup 改善候補と同じ）。
- pii の埋め込み入力で PII span をマスクするかの A/B（decision-queue に記録）。

## 再現

```bash
just profile=pii full-index && just profile=finance full-index
just profile=finance eval --check-thresholds && just profile=finance diagnose && just profile=finance eval-names
just profile=pii eval --check-thresholds && just profile=pii diagnose && just profile=pii eval-names
just store-fingerprint   # 作業の前後で 1 回ずつ取り、JSON を比べる
```
