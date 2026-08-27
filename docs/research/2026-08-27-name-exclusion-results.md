# 埋め込みからの人名除外 (2026-08-27)

**Snapshot date:** 2026-08-27
**Index:** full corpus 1,000,000 personas / ruri-v3-310m (768 dims) /
embedding_text_policy = strip-person-names-v1
**Harness:** `just eval --check-thresholds` + `just diagnose`（vector 単独列
新設）+ 層化コホート / 内容ベース移行検証（scratch）
**前提:** 改善候補③。人名汚染は hard 12 中 6 クエリで確認済み
（`docs/research/2026-08-27-golden-set-hardening.md`）。

## 設計

- dataset に name カラムは無く、名前は各 persona フィールド先頭に埋め込み。
  `name_stripping.detect_person_name` はフィールド先頭パターン
  （スペース入り「姓 名は、」/ スペース無し「西口甲一は」）を抽出し、
  **2 フィールド以上の一致**で認定（代名詞 stopword・候補競合は no-op）。
  除去は主語形（名前+年齢括弧+係助詞+読点）と、助詞/句読点境界つきの
  完全一致のみ — 「福岡」姓と福岡県の衝突は起きない。
- **埋め込み入力のみ**除去。保存テキスト（payload/ES/表示）は原文 —
  BM25 の名前検索は温存。metadata に policy を forensic 記録。
- Metamorphic 保証（単体テスト）: 名前だけ異なる同文ペルソナは
  同一の埋め込み入力になる。

## 事前検証（全コーパス dry-run）

- 検出 **991,473 / 1,000,000（99.15%）**、スペース型 982,703 /
  無スペース型 8,770、distinct 945,744（最頻 8 件）。
- 頻度上位 15 は全て実名 — 誤爆観測ゼロ。no-op 0.85% は半角括弧・
  前置ラベル・スペース位置変種などの**安全側の取りこぼし**。

## 移行と完走証明

- `just full-index` の in-place 再実行（8 シャード冪等 upsert、~88 分）。
- **内容ベース検証**: 層化コホート 40 uuid（8 シャード × クラス別）の
  埋め込みを手元で再計算し Qdrant 実ベクトルと照合 —
  **40/40 cosine = 1.000000**（件数ではなく内容で完走を証明）。
  3 ストア 1,000,000 一致。

## 途中で必要になった融合修正（同 PR）

名前入り本人要約クエリで本人が vector top-k から漏れ、keyword rank1
（1/61）が別人の vector rank1 と**完全同点** → uuid 順の運任せになる
症例を確認（probe 3/3 一致）。**同点時は keyword レグ優先**の
tie-break を追加（BM25 rank1 は名前・希少語の強い字面一致で、100 万の
近傍雑音の vector rank1 より特異）。recall@1 0.68 → 0.92 に回復し、
golden も改善（hard 0.600 → 0.633）。

## 結果（ライブ 1M、tie-break 込み）

| 指標 | ③前 | ③後 |
|---|---|---|
| basic precision@5 | 0.950 | **0.983** |
| hard precision@5 | 0.450 | **0.633** |
| overall | 0.700 | **0.808** |
| **vector 単独** basic / hard | 0.900 / 0.433 | **1.000 / 0.650** |
| filtered geo (n=4) | 0.950 | 0.900 |
| self-retrieval recall@1 / @10 (n=100) | 1.00 / 1.00 | **0.92 / 1.00** |
| eval elapsed（warm） | 9.0s | 9.3s |

対象 6 クエリ（人名汚染クラス）の vector 単独: 平均 delta **+0.30**、
5/6 改善（温泉 0.40→1.00 / 福岡 0.60→1.00 / ミステリー 0.80→1.00 /
沖縄 0.40→0.80 / パン 0.00→0.20）。楽器 0.00 のみ据え置き —
これは人名でなく意味混同（カラオケ業務者）で、別クラスの課題。
**vector レグは初めて keyword 単独（hard 0.583）を上回った。**

## 事前登録制約からの逸脱と requester 裁定

- recall@1 0.92（< 事前登録 0.99）: 敗因は全数検死で 2 型 —
  同点 tie（修正済み）と、**そっくりな別人の両レグ合議による正当勝ち**。
  後者は匿名化ベクトルの設計意図（名前抜きで意味等価なら等価）と
  旧バーの衝突。recall@10 = 1.00（配線健全）。
- **Requester 裁定（2026-08-27）: 採用 + バー改定** —
  self-retrieval バーを「recall@1 ≥ 0.90 ∧ recall@10 ≥ 0.99」に改定
  （intent.md 更新済み、check_thresholds で機械強制）。
- filtered geo 0.900（北海道 0.80 / 沖縄 0.80、各 -1 ヒット）は未検死 —
  沖縄述語の語彙不足（泳ぎ/潮風）と同型の可能性が高く、
  **golden 保守 PR** で精査・再基準化する。
- 名前単独クエリの rank1 は 19/24（CJK unigram BM25 の精度限界。
  ③前は未計測）。名前検索強化（ES への name 抽出フィールド等）は
  追加 work unit 候補。

## 副産物

- **metadata 失踪事件の真犯人逮捕**: `test_reset_indexes_uses_batched_
  persona_delete` が METADATA_PATH を monkeypatch せず本物の
  `.cache/index_metadata.json` を**毎 pytest 実行で削除**していた
  （番兵ファイル + テスト単位バイセクトで特定、isolation 追加で封鎖）。
- diagnose に vector 単独列、catalog_snapshot を新インデックスで再生成
  （embedding_text_policy 注記つき）。
