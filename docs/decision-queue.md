# Decision Queue

Items requiring human review, curated by AI agent sessions and the
automated tooling patrol (see hironow/dotfiles routine). Append new
entries under a dated section in Open Items; once the human has decided,
tick the checkbox and move the entry to Decided (or delete it).

Entry format:

```markdown
## YYYY-MM-DD

- [ ] **<topic>**: <decision needed> — background / options / recommendation
```

---

## Open Items

### 2026-11-11

- [ ] **finance / pii の品質バー**：japan と同じ形でバーを機械強制するか、値をいくつにするかを決める。
  対象の指標は golden basic / hard / filtered の precision@5、finance の self-retrieval recall@1 / @10、pii の値 lookup recall である。
  背景：両データセットは今、カナリア 3 本だけで gate し、品質指標は報告だけにしている（`search_ja_persona/evaluation.py:RATIFIED_QUALITY_PROFILES`）。
  ベースライン：`docs/research/2026-11-11-finance-pii-baseline.md`。
  推奨：japan と同じく、ベースラインから少し余裕を引いた値で裁定する。
  裁定後に intent.md と `RATIFIED_QUALITY_PROFILES` を更新する。
- [ ] **pii の埋め込み入力で PII span をマスクするか**：今は原文を埋め込んでいる（`raw-text-v1`）。
  選択肢は 3 つある。
  現状維持、直接識別子（氏名、電話番号、口座番号など）だけを `[label]` に置き換える、全 span を置き換える。
  推奨：採否は事前登録した A/B（golden と値 lookup）で決める。
  値の完全一致検索は keyword レグが担うので、マスクしても lookup は落ちない見込みである。
- [ ] **japan の名前検索の下限値**：`just eval-names` に recall@1 ≥ 0.70 の回帰フロアを機械化するかを決める。
  背景：2026-08-27 の baseline は recall@1 0.725 / recall@10 0.900 で、提案は `docs/research/2026-08-27-name-lookup-baseline.md` にある。
  推奨：名前を BM25 の本文から外す改善の後に、引き上げた値で再基準化する。

## Decided

### 2026-07-27

- [x] (2026-08-27 requester 指示により推奨どおり intent.md を改訂) **intent.md の emulator 非目標が実態と矛盾**: `docs/intent.md` の非目標「emulator stack は `sets/emulator-set` に移設・本 repo は稼働中 emulator を消費するだけ」を ADR 0001 / PR #6 が覆した（最小 `emulator/compose.yaml` を本 repo に vendor 保持）。加えて `sets/emulator-set` は不在で、実体は `~/dotfiles/emulator`（upstream `github.com/hironow/emulator-set`）。推奨: intent.md の非目標を「最小 emulator サブセットを vendor 保持／フルキットは upstream 側」に人間が改訂する。handover.md は本セッションで整合済み。
- [x] (2026-08-27 requester 指示により推奨どおり intent.md を改訂) **intent.md の「bundled 1k sample」表現が不正確**: `docs/intent.md` の Success Criteria は「the bundled 1k sample」と書くが、`qa_samples/qa_sample.parquet` は `.gitignore` の `qa_samples/` で除外され未コミット（`just qa-sample` で生成）。README / storage-footprint / handover は本セッションで「生成」表現に修正済み。推奨: intent.md の「bundled」を「locally-generated（`just qa-sample`）」に人間が改訂する。
