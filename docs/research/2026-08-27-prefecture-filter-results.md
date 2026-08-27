# prefecture payload フィルタの効果測定 (2026-08-27)

**Snapshot date:** 2026-08-27
**Index:** full corpus 1,000,000 personas / ruri-v3-310m (768 dims)
**Harness:** `just eval`（レポート schema v3、filtered セクション新設）
**前提:** `docs/research/2026-08-27-golden-set-hardening.md` の改善候補①。
仮説 —「居住は埋め込みでなくフィルタで扱うべき」。

## 実装

- 居住条件を明示 API 化: `--prefecture`（CLI）/ `prefecture=`（application
  以下全層）。Qdrant は payload filter、Elasticsearch は `bool.filter` の
  `term`。47 都道府県の正式名バリデータで「沖縄」等の黙殺を入口で拒否。
- Qdrant の keyword payload index は新規コレクションでは index 時に自動
  作成、既存コレクションには `just ensure-payload-index`（冪等 migration、
  検索経路は read-only のまま）。ライブ 1M へは適用済み
  （payload_schema: prefecture keyword / points 1,000,000）。
- ベンチ設計（交絡防止）: tier 平均は従来どおり **unfiltered のみ**で算出
  し、`filters` 付きエントリは追加で filtered 実行して別セクションに
  paired delta を記録する。

## 結果（`outputs/search_eval-20260827-085520.json`）

| hard 地理クエリ | unfiltered | filtered | delta |
|---|---|---|---|
| 北海道×スキー | 0.60 | **1.00** | +0.40 |
| 沖縄県×海辺レジャー | 0.40 | **1.00** | +0.60 |
| 京都府×寺社歴史 | 1.00 | **1.00** | ±0 |
| 福岡県×ラーメン屋台 | 0.60 | **1.00** | +0.40 |
| **filtered geo mean** | 0.65 | **1.000** | +0.35 |

- unfiltered 系は不変: basic 0.900 / hard 0.433 / self-retrieval 1.00/1.00
  （payload index 追加による無フィルタ検索への回帰なし）。
- 実 emulator の integration テストで、両レグ・fused とも指定県のみを
  返すことと migration の冪等を検証済み
  （`tests/test_integration_emulators.py::test_prefecture_filtered_search_with_emulators`）。

## 知見

- 地理弱点クラスは**フィルタで完全解消**（4/4 満点）。人名汚染由来の
  誤ヒット（福岡姓・具志堅/与那嶺姓）もフィルタが機械的に落とす。
- basic の 沖縄 0.40 / 北海道 0.80 は素通し測定として意図的に残して
  ある（生の意味検索の地理弱点の回帰監視）。
- 残る hard の失点は複合条件（子育て×看護 0.20 等）と人名汚染
  （パン 0.00）— それぞれ改善候補②融合再設計・③人名除外の領域。

## Requester 裁定事項（追記）

- filtered geo mean のバー化（例: ≥ 0.9）を intent.md に入れるか。
- 既裁定待ち: hard バー（≥ 0.35 案）/ basic 退化述語 3 件 / 候補②③④の順。
