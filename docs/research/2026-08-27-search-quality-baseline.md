# 検索品質ベースライン (2026-08-27)

**Snapshot date:** 2026-08-27
**Index:** full corpus 1,000,000 personas / ruri-v3-310m (768 dims) /
Qdrant + Elasticsearch + Neo4j fused search（uuid 正規化修正 #41 適用後）
**Harness:** `just eval` (`scripts/evaluate_search.py` +
`search_ja_persona/evaluation.py`)。LLM 審判なしの機械採点のみ。

## 測定方法

- **Golden queries（precision@5）**: `scripts/golden_queries.json` の
  12 クエリ。各クエリは text キーワード群または prefecture/region の
  述語を持ち、top-5 のうち述語を満たす件数の割合を採点する。
- **Self-retrieval（recall@k）**: シャード 0 から固定シード
  (20260827) で 100 ペルソナを抽出し、その `persona` 要約文をクエリに
  して本人 uuid が top-k に返るかを測る。embedder とインデックスの
  整合性の直接的な健全性指標。

## ベースライン結果

| 指標 | 値 |
|---|---|
| Golden mean precision@5 | **0.900** |
| Self-retrieval recall@1 (n=100) | **1.00** |
| Self-retrieval recall@10 (n=100) | **1.00** |
| 実行時間（112 クエリ、モデルロード込み） | 32.8s |

クエリ別（満点 9/12、以下が非満点）:

| クエリ | precision@5 | 主因 |
|---|---|---|
| 北海道で暮らし雪や冬のスポーツを楽しむ人 | 0.80 | 地理制約の混入漏れ |
| 登山やハイキングが趣味の人 | 0.60 | 述語キーワードの網羅不足も混在 |
| 沖縄で暮らし地元の食文化を愛する人 | 0.40 | 「沖縄好きの他県民」を意味検索が返す |

## 知見

- **地理制約付きクエリが弱点クラス**。意味ベクトルは「土地への言及・
  嗜好」と「居住」を区別しないため、居住条件は検索側のフィルタ
  （Qdrant payload filter で prefecture を絞る等）で扱うのが筋。
  現状の CLI/API に居住フィルタは未実装（改善候補）。
- Self-retrieval が満点である間は、embedder・プレフィックス・
  インデックスの配線劣化を強く否定できる（回帰の即死検知に有効）。
- 述語はあくまで下界（真の適合をキーワードが取りこぼす）。
  precision の絶対値よりも経時変化を見ること。

## 提案する品質バー（requester 承認待ち）

- Golden mean precision@5 **≥ 0.85**
- Self-retrieval recall@1 **≥ 0.99**

embedder・前処理・融合ロジック変更時は `just eval` を回し、この線を
割ったら回帰として扱う（しきい値の正式化は intent.md 側の裁定事項）。
