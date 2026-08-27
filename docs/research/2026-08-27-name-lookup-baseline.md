# 名前検索 baseline (2026-08-27)

**Snapshot date:** 2026-08-27（人名除外採用後）
**Harness:** `just eval-names`（`scripts/name_lookup_queries.json` —
8 シャード層化・固定 40 名、`scripts/evaluate_name_lookup.py`）

## 背景

人名除外採用（#49）により、名前での本人検索は keyword レグ +
keyword 優先 tie-break が担う設計になった。その保証を固定セットで
定点観測する（採用時のアドホック計測は 19/24）。

## Baseline

| 指標 | 値 |
|---|---|
| recall@1 (n=40) | **0.725** |
| recall@10 | **0.900** |
| 実行時間 | 2.2s |

miss の型: 超頻出姓 + 短い名（佐藤 文晴 は top10 圏外、丸山 典 /
菅野 成 など一文字名）。CJK unigram BM25 では名前の文字が他人の
本文にも散在するため識別力が不足する。

## 改善候補（別 work unit）

- ES に抽出済み氏名の keyword フィールドを追加し名前完全一致を
  ブーストする（name_stripping の抽出を index 時に再利用）。
  再インデックス or ES 再構築が必要。

## バー提案（requester 裁定待ち）

- name lookup recall@1 **≥ 0.70** を回帰フロアとして `eval-names` に
  機械化（改善 work unit 完了後に引き上げ再基準化）。
