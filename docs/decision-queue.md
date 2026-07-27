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

## 2026-07-27

- [ ] **intent.md の emulator 非目標が実態と矛盾**: `docs/intent.md` の非目標「emulator stack は `sets/emulator-set` に移設・本 repo は稼働中 emulator を消費するだけ」を ADR 0001 / PR #6 が覆した（最小 `emulator/compose.yaml` を本 repo に vendor 保持）。加えて `sets/emulator-set` は不在で、実体は `~/dotfiles/emulator`（upstream `github.com/hironow/emulator-set`）。推奨: intent.md の非目標を「最小 emulator サブセットを vendor 保持／フルキットは upstream 側」に人間が改訂する。handover.md は本セッションで整合済み。
- [ ] **intent.md の「bundled 1k sample」表現が不正確**: `docs/intent.md` の Success Criteria は「the bundled 1k sample」と書くが、`qa_samples/qa_sample.parquet` は `.gitignore` の `qa_samples/` で除外され未コミット（`just qa-sample` で生成）。README / storage-footprint / handover は本セッションで「生成」表現に修正済み。推奨: intent.md の「bundled」を「locally-generated（`just qa-sample`）」に人間が改訂する。
