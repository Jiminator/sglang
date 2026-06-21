# Round 2 Contract

## Mainline Objective (ONE)
Make the (already-correct, comparator-accepted) M-B verdict GENUINELY publishable by satisfying the two
fail-closed gates Codex showed are still open: **AC-5** (every published DS SLO trial must carry non-null
`dense_fallback_total == 0` AND `selected_tokens_mean < total_tokens_mean`; `trial_evidence.py` currently
REFUSES all 6) and **AC-8** (raw verdict evidence preserved in committed artifacts + push resolved). The
fix for AC-5 must be a real backend data-path change, not a documentation workaround.

## Target ACs
- **AC-5** (no-op refusal): wire the GLM/`dsa_backend` DS per-request summary so `meta_info["double_sparsity"]`
  carries real `sparsity_rate/selected_tokens/dense_fallback`; every DS SLO trial's `.evidence.json` shows
  `verdict: "PASS"`.
- **AC-8** (evidence/ledger/push): commit raw verdict inputs (losslessly), regenerate `queue.md`+`results.md`
  to one clean state, re-run the comparator from committed artifacts, resolve push (owner remote or waiver).
- (AC-9 depends on AC-5's embedded per-trial no-op proof.)

## Blocking issues (truly block the objective)
- **B1 — GLM/dsa-backend DS summary publisher is missing.** `_publish_ds_request_summary` lives on
  DeepseekV2's attention (`deepseek_v2.py:2074`); GLM uses `Glm4MoeAttention` + the `dsa` backend, which
  never reaches it, so the DS aggregate fields are null and `trial_evidence.py` REFUSES. Fix in the backend
  data path (Codex plan):
  1. Helper near the DS decode path in `dsa_backend.py`: return unless `self.enable_double_sparsity`;
     `selected = (page_table_1 >= 0).sum(dim=1)` (the same validity used at the lifted-budget decode path);
     `total = forward_batch.seq_lens`; `dense_fallback = 0`; populate
     `forward_batch.ds_per_request_summary["double_sparsity"]` mirroring the DeepseekV2 record shape
     (`metrics.meta_info_for_request`).
  2. Call it in `forward_decode` after `page_table_1` is finalized, before the DS decode returns; do NOT
     touch native-DSA / non-DS decode paths.
  3. Existing transport (`model_runner.py:3232` copies `ds_per_request_summary` onto logits_output) +
     bench aggregation (`bench_serving.py`) then emit non-null `dense_fallback_total/selected_tokens_mean/
     total_tokens_mean`.
  4. Focused smoke check: a short GLM DS `bench_serving --output-details` run whose JSONL has the DS fields
     and on which `trial_evidence.py` exits 0.
- **B2 — raw evidence not committed + push unresolved.** AC-8 requires the reviewer to reproduce the verdict
  from committed artifacts. Commit lossless raw inputs (compressed `.jsonl.zst`/`.log.zst` with decompress
  command + hashes, or force-add) — per-trial JSONLs, per-boot serve logs, server_info, tax JSONLs/logs,
  comparator inputs/outputs, run-order/command ledger. Re-run the comparator from committed artifacts; update
  `EVIDENCE_SHA256.txt`. Push to an owner-approved remote/branch, or record an explicit owner waiver.

## Queued (out of scope this round; documented, not blocking)
- Plan-workflow terminology in implementation comments (`batch_result_processor.py:184/329/745`,
  AC/DEC references in `benchmark_compare.py`). Real drift from the plan's no-terminology rule, but does not
  block AC-5/AC-8 publication. Clean in a focused pass after the verdict is published.

## Success criteria
1. `dsa_backend.py` publishes `ds_per_request_summary["double_sparsity"]` for GLM DS decode, gated on
   `enable_double_sparsity`; native DSA + non-DS paths unaffected (verified by reading the gate + a DSA smoke).
2. A GLM DS `bench_serving --output-details` JSONL carries non-null `dense_fallback_total=0`,
   `selected_tokens_mean`, `total_tokens_mean`, and `trial_evidence.py` exits 0 on it.
3. DS verdict sweep re-run with the fix → all 6 trials `.evidence.json` `verdict: "PASS"`; DSA re-run at the
   SAME new HEAD → both comparators re-ACCEPT (rc=3) from the committed/replayable inputs.
4. Raw evidence committed losslessly (compressed + hashes + decompress command); comparator re-run from those.
5. `queue.md` + `results.md` regenerated to one current state (no stale RUNNING/PENDING rows).
6. Push to an owner-approved destination, or an explicit owner waiver recorded in `results.md`.

## Notes
- Re-running DS changes HEAD → DSA must be re-run at the same HEAD for the comparator's commit_sha gate (the
  backend change is DS-gated so DSA behavior is identical, but the gate requires matching sha). Full sweep
  re-run (~3.8h) is expected and acceptable (autonomous; honest verdict unchanged).
- The verdict itself (DS PASS@16/32, FAIL@64) is established + reproduced; Round 2 makes its EVIDENCE pass the
  fail-closed gates, it does not change the numbers.
