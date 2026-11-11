# finance / pii 全量索引の baseline と japan 不変の検証 (2026-11-11)

**Snapshot date:** 2026-11-11
**Environment:** Windows 11 / RTX 4090 24GB / Docker Desktop の emulator（Qdrant 1.20.0、Elasticsearch 9.4.3、Neo4j 2025-community）、埋め込みは `ruri-v3-310m`（768d）
**Harness:** `just profile=<finance|pii> full-index` / `eval --check-thresholds` / `diagnose` / `eval-names`、`just store-fingerprint`

## 結論

- 3 データセットとも、3 ストアの件数が一致した（japan 1,000,000×3、finance 1,000,000×3、pii 100,000×3）。
  カナリア 3 本も全データセットで通った。
- japan（V1）は変わっていない。
  作業前（05:34）と作業後（08:36）の `store_fingerprint` が完全に一致し、JSON ファイルの sha256 も同じだった。
  ES `personas` の `index_total` は前後とも 0 で、emulator の起動後に V1 への書き込みは 1 件もない。
  eval も、全指標と 24 クエリそれぞれの値が前後で一致した。
- pii の検索品質は高い（golden basic 0.900 / hard 0.880）。
  finance は japan よりかなり低い（basic 0.457 / hard 0.425）。

## 索引にかかった時間

| データセット | 件数 | 所要時間 | 備考 |
|---|---|---|---|
| pii | 100,000 | 13 分 54 秒 | `data.parquet` 1 ファイル |
| finance | 1,000,000 | 150 分 54 秒 | 8 シャード |

finance の所要時間は、japan の実績（約 50 分）の 3 倍だった。
索引中は WSL 上の vLLM（Qwen3.8-27B）と GPU を共有しており、埋め込み速度は 35〜130 件/秒の間で揺れた。
finance のテキストは 9 フィールドで、japan（6 フィールド）より長い。
二つの要因がそれぞれどれだけ効いたかは切り分けていない。

## 検索品質の測定結果

| 指標 | japan（参考） | finance | pii |
|---|---|---|---|
| golden basic precision@5 | 0.917 | 0.457 | 0.900 |
| golden hard precision@5 | 0.650 | 0.425 | 0.880 |
| filtered（japan と finance は都道府県、pii は domain） | 0.950 | 0.750 | 1.000 |
| self-retrieval recall@1 / @10 | 0.92 / 1.00 | 0.91 / 1.00 | 対象外 |
| lookup recall@1 / @10（n=40） | 0.725 / 0.900（名前、2026-08-27 の記録で今回は未測定） | 0.825 / 0.950（名前） | 0.875 / 1.000（PII 値） |
| 述語のランダム基準率（basic / hard） | 今回は未測定 | 0.030 / 0.026 | 0.024 / 0.018 |
| context coverage / keyword 寄与 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

pii では self-retrieval を測らない。
文書の説明文（`document_description`）は約 10 文書で共有されるので、説明文をクエリにしても自分の文書を一意に引けないからだ。
代わりに、1 文書にしか現れない PII 値 40 件（domain ごとに 5 件）の lookup で、文書を特定できるかを測る。

finance と pii の品質バーは未裁定で、`docs/decision-queue.md` に載せた。
今は品質指標を報告するだけで、gate はカナリアだけにかけている。

### finance の述語の修正

初回の測定では、finance の basic は 0.414 だった。
「毎月コツコツ積み立てて資産を増やしている人」の述語が漢字の「積立」だけを見ていて、本文の動詞「積み立てる」を取りこぼしていた。
実際、このクエリの上位 5 件は、どれも毎月の積み立てを書いていた。
同じ語の表記ゆれなので、述語に「積み立」を足した。
1 シャード全体で見た基準率は 0.5% から 6.7% になり、ほかの述語と同じ水準に収まった。
この 1 件の修正で、basic は 0.414 から 0.457 になった。
ほかの述語は変えていない。

### finance の精度が低い理由

diagnose の結果から、次の 4 点が読み取れる。

- 0 点のクエリは、保険の見直し、奨学金の返済、共働きで住宅ローンと教育費を抱える世帯、退職金の運用、相続の 5 件だった。
  上位は「保険や資産を持つ人」で、見直しや相続の記述はない。
  述語の問題ではなく、検索の取りこぼしである。
- 奨学金のクエリでは、keyword レグが名前の「奨太」に一致した。
  BM25 が CJK を 1 文字単位で扱うので、「奨」の 1 文字が名前と本文の区別なく一致する。
  japan ではベクトル側で起きた名前汚染が、ここでは keyword 側で起きている。
- vector 単独が fused より高い場合がある（basic で vector 0.471 / fused 0.457）。
  「大阪で自営業を営む人」と「福岡県で年金生活を送る人」は、どちらも vector 1.00 / keyword 0.00 / fused 0.40 だった。
  RRF は二つのレグを 1:1 で混ぜるので、述語に合わない keyword レグの上位が fused の上位 5 件に入り込む。
- finance のペルソナは家計の記述が多く、互いに似ている。
  語彙で区別できる特徴の出現率は小さく、1 シャード 125,000 件のうち奨学金は 24 件（0.02%）、相続は 501 件（0.4%）だった。

## 改善候補（それぞれ別の作業単位）

- finance の RRF 重みの A/B（vector 寄り）。
  japan の重み 1:1 は requester の裁定なので、データセット別の重みは新しい判断になる。
- 名前を keyword フィールドに分けて、BM25 の本文から外す（japan の name lookup の改善候補と同じ）。
- pii の埋め込み入力で PII span をマスクするかの A/B（decision-queue に記録した）。

## 再現

```bash
just profile=pii full-index && just profile=finance full-index
just profile=finance eval --check-thresholds && just profile=finance diagnose && just profile=finance eval-names
just profile=pii eval --check-thresholds && just profile=pii diagnose && just profile=pii eval-names
just store-fingerprint   # 作業の前後で 1 回ずつ取り、JSON を比べる
```
