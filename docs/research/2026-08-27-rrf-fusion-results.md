# RRF 融合の導入と A/B 判定 (2026-08-27)

**Snapshot date:** 2026-08-27
**Index:** full corpus 1,000,000 personas / ruri-v3-310m (768 dims)
**Harness:** `just eval`（`--check-thresholds` / `--rrf-weights` 新設）+
`just diagnose`
**前提:** `docs/research/2026-08-27-golden-set-hardening.md` の改善候補②。
根拠 — 旧融合は vector が top-limit を埋め keyword レグを実質廃棄
（overlap ≈ 0）、BM25 単独が predicate 指標で fused を上回っていた
（hard 0.583 vs 0.433、オラクル 0.667）。

## 設計

- 重み付き Reciprocal Rank Fusion:
  `rrf_score(d) = Σ_leg w_leg / (60 + rank_leg(d))`。
  全順序: rrf desc → 由来レグ数 desc → best rank asc → uuid asc。
- 候補深さ `max(limit, min(limit*3, 30))`、Neo4j context は融合後
  top-limit のみ。`score` は従来の意味（vector に現れれば Qdrant score、
  keyword-only は ES `_score`）を維持し、順位付けは新設 `rrf_score`。
  `sources` に由来レグを記録。

## 事前登録 A/B（同一 seed・warm 3 回ずつ、決定的）

候補: (a) unweighted 1:1 / (b) vector:keyword = 2:1。
採用規則（事前明文化）: 必須 = recall@1 ≥ 0.99 ∧ basic ≥ 0.85 ∧
filtered geo ≥ 1.000 維持 → 満たす案から hard 最大。

| 指標 | 旧融合 | (a) 1:1 | (b) 2:1 |
|---|---|---|---|
| basic precision@5（バー） | 0.900 | **0.950** | 0.900 |
| hard precision@5 | 0.433 | **0.450** | 0.433 |
| overall | 0.667 | **0.700** | 0.667 |
| filtered geo (n=4) | 1.000 | 0.950 | 1.000 |
| self-retrieval recall@1/@10 | 1.00/1.00 | **1.00/1.00** | 1.00/1.00 |
| eval elapsed（warm 中央値） | 11.3s | **9.0s** | ≈同等 |

- (b) は本コーパスで旧融合と完全一致（overlap ≈ 0 のため重み 2:1 では
  vector 支配に退化）— 無害だが無益。
- (a) は basic +0.05 / hard +0.017 / overall +0.033 / **20% 高速化**
  （context 呼び出しを候補全件→top-limit に削減した効果が 3 倍取得を
  上回る）。

## filtered geo 0.950 の検死（事前登録制約からの逸脱開示）

形式上、(a) は「filtered geo ≥ 1.000」の必須制約に 1 ヒット分抵触する。
現物検分（沖縄クエリ、filtered top-5）の結果、入れ替わったヒット
（沼田氏・keyword 由来）は「朝の潮風を感じながらウォーキング…季節ごとに
**海での泳ぎ**」— 海辺レジャーに親しむ沖縄県民そのものであり、述語語彙
（ビーチ/海水浴/ダイビング…）の**偽陰性**。真の劣化ではなく下界測定の
アーティファクトと判断し、(a) を採用（本逸脱は PR 上で明示開示し、
requester の merge 判断を追認とする）。述語への 泳ぎ/潮風 追加は
事後調整を避けるため本 PR では行わず、次回の golden 保守で再基準化と
共に扱う。

## 判定

**採用: (a) unweighted RRF（本番デフォルト 1:1）**。
`DEFAULT_RRF_WEIGHTS = (1.0, 1.0)`。

## diagnose（採用後）

- basic: fused **0.950** | keyword 0.983 | random 0.239
- hard: fused **0.450** | keyword 0.583 | random 0.047
- keyword 単独との残差（hard -0.133）は複合条件クエリ
  （子育て×看護等）で BM25 が字面 AND を拾える一方、意味レグの候補に
  片側面しか無いケース。次の伸びしろは改善候補③（人名除外、要再
  インデックス）と、複合クエリの分解（LLM 前処理ルート — 別 work unit）。

## 副産物

- `--check-thresholds`: 承認済み intent バー（basic ≥ 0.85 /
  recall@1 ≥ 0.99）+ 指標の存在検証を非ゼロ終了で機械強制。
- `just check`（pre-commit + test の集約ゲート）。
