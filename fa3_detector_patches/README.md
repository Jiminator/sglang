# FA3 read-side guard patches

`git am`-applicable patches for the FA3 paged-KV read-side guards, for forks that cannot
cherry-pick them directly. Apply onto a base that already has KV-Canary and the async-assert
machinery (see `FA3_ROOT_CAUSE_RUNBOOK.md`, Step 1, Path A):

```bash
curl -sL <raw-url>/0001-read-side-page-table-guard.patch | git am
curl -sL <raw-url>/0002-seqlen-fail-fast.patch          | git am
curl -sL <raw-url>/0003-extend-guard-spec-encoder-local.patch | git am
```

| File | Adds |
|---|---|
| `0001-read-side-page-table-guard.patch` | `maybe_assert_page_table_in_range` + call sites (read-side complement to #27459) |
| `0002-seqlen-fail-fast.patch` | `maybe_assert_seqlens_within_context` (length overshoot fail-fast) |
| `0003-extend-guard-spec-encoder-local.patch` | extends the read-side guard to the spec-decode-expand, encoder, and local-attention page tables |
