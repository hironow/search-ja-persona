# Decision Queue

Items requiring human review, curated by AI agent sessions and the
automated tooling patrol (see hironow/dotfiles routine). Append new
entries under a dated section; tick the checkbox (or delete the entry)
once the human has decided.

Entry format:

```markdown
## YYYY-MM-DD

- [ ] **<topic>**: <decision needed> — background / options / recommendation
```

---

## Open Items

## 2026-11-11

- [ ] **finance / pii の品質バー**: japan と同じ形（golden basic / hard / filtered の precision@5、finance は self-retrieval recall@1 / @10、pii は値 lookup recall）でバーを機械強制するか、値をいくつにするか。背景: 両データセットは現在カナリア 3 本だけで gate し、品質指標は報告のみ（`search_ja_persona/evaluation.py:RATIFIED_QUALITY_PROFILES`）。ベースライン: `docs/research/2026-11-11-finance-pii-baseline.md`。推奨: ベースラインから japan と同じく少し余裕を引いた値で裁定し、裁定後に intent.md と `RATIFIED_QUALITY_PROFILES` を更新する。
- [ ] **pii の埋め込み入力で PII span をマスクするか**: 現在は原文を埋め込む（`raw-text-v1`）。選択肢: 現状維持 / 直接識別子（氏名・電話・口座番号など）だけ `[label]` に置換 / 全 span を置換。推奨: 採否は事前登録した A/B（golden と値 lookup）で決める。値の完全一致検索は keyword レグが担うので、マスクしても lookup は落ちない見込み。

## 2026-07-27

- [x] (2026-08-27 requester 指示により推奨どおり intent.md を改訂) **intent.md の emulator 非目標が実態と矛盾**: `docs/intent.md` の非目標「emulator stack は `sets/emulator-set` に移設・本 repo は稼働中 emulator を消費するだけ」を ADR 0001 / PR #6 が覆した（最小 `emulator/compose.yaml` を本 repo に vendor 保持）。加えて `sets/emulator-set` は不在で、実体は `~/dotfiles/emulator`（upstream `github.com/hironow/emulator-set`）。推奨: intent.md の非目標を「最小 emulator サブセットを vendor 保持／フルキットは upstream 側」に人間が改訂する。handover.md は本セッションで整合済み。
- [x] (2026-08-27 requester 指示により推奨どおり intent.md を改訂) **intent.md の「bundled 1k sample」表現が不正確**: `docs/intent.md` の Success Criteria は「the bundled 1k sample」と書くが、`qa_samples/qa_sample.parquet` は `.gitignore` の `qa_samples/` で除外され未コミット（`just qa-sample` で生成）。README / storage-footprint / handover は本セッションで「生成」表現に修正済み。推奨: intent.md の「bundled」を「locally-generated（`just qa-sample`）」に人間が改訂する。
