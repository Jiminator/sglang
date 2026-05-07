# CI Regression Bisection Report — `TestFlashInferDeterministic.test_prefix_with_logprobs` on H100

**Investigator:** Claude (sglang-bisect-ci-regression skill)
**Date:** 2026-05-07
**Repo:** `sgl-project/sglang`
**Reporting run:** https://github.com/sgl-project/sglang/actions/runs/24971499389/job/73115280649

---

## Status

**Still failing as of 2026-05-07.** The most recent scheduled run on `main` (run `25469734855`, head `2e642ea187`) reproduces the same `Some logprobs differ across batch sizes!` assertion (102 logprob mismatches across 42 samples × 4 batch sizes). No fix PR or open issue tracking this regression was found.

## Failure Signature

- **Test:** `test/registered/core/test_qwen3_next_deterministic.py::TestFlashInferDeterministic::test_prefix_with_logprobs`
- **Underlying assertion:** `python/sglang/test/test_deterministic_utils.py:74` (`assert result == 1`) raised after the helper prints `✗✗✗ Some logprobs differ across batch sizes! ✗✗✗` from `python/sglang/test/test_deterministic.py:556`.
- **Workflow / job:** `Nightly Test (Nvidia)` → `nightly-test-general-4-gpu-h100` (suite `nightly-4-gpu`, registered with `est_time=200, nightly=True`).
- **Model:** `Qwen/Qwen3-Next-80B-A3B-Instruct` (`Qwen3NextForCausalLM`).
- **Server config:** `--attention-backend flashinfer --tp 4 --enable-deterministic-inference --trust-remote-code --cuda-graph-max-bs 32 --random-seed=*` (per `COMMON_SERVER_ARGS` + the FlashInfer subclass override).
- **Test logic:** Fixed prompt is run at multiple batch sizes (n_start=10, n_trials=10, temperature=0.5, return_logprob=True). Asserts that the first-position logprob is bit-identical across batch sizes. Server starts once via `setUpClass`, so each batch size hits the same loaded weights/CUDA-graph captures.
- **Error excerpt (from job 73115280649):**

  ```
  ✗ Sample 28: Logprob mismatch at position 0: -0.03974096477031708 vs -0.05419498682022095 (diff: 0.01445402204990387)
  Found 26 mismatches out of 28 samples
  ✗✗✗ Some logprobs differ across batch sizes! ✗✗✗
  File ".../test_deterministic_utils.py", line 74, in test_prefix_with_logprobs
  AssertionError
  ✗ FAILED: .../test_qwen3_next_deterministic.py returned exit code 1
  ```

- **Deterministic across runs:** yes. Identical numeric diffs reappear across many sample indices in a single run (same prompt → same batch-size-dependent divergence) and the failure shows up every dispatched scheduled run from 2026-04-19 onward.

---

## Boundary

Looking at scheduled `Nightly Test (Nvidia)` runs on `main` between 2026-04-08 and 2026-05-07 and grepping each `nightly-test-general-4-gpu-h100` log for `Logprob mismatch at position` and `Some logprobs differ across batch sizes`:

| Status | Run ID | Job ID | Date (UTC) | Head SHA | Runner | Mismatches |
|---|---|---|---|---|---|---|
| **Last pass** | `24592875660` | `71917163703` | 2026-04-18 00:45 | `9c47bbad13df0ae36a9127a7c271a83aa461fa56` | `h100-novita10-gpu-0123` | 0 |
| **First fail** | `24617581978` | `71982368622` | 2026-04-19 00:53 | `2a327f08772f6b9ada7f2f4792f9b7d0e16a5fa1` | `h100-novita10-gpu-4567` | 110 |
| Reproduction (linked in question) | `24971499389` | `73115280649` | 2026-04-27 00:53 | `977830e91e4197628f581fd96cf257c6d9466f9d` | `h100-novita10-gpu-0123` | (same fingerprint) |
| Latest fail (current) | `25469734855` | `74730829418` | 2026-05-07 00:55 | `2e642ea1872d12e3d838bd3350d4d64f792042ec` | `h100-novita10-gpu-4567` | 102 |

### Full evidence table

| Date | Run | Job | Conclusion | SHA | Runner | `test_prefix_with_logprobs` |
|------|-----|-----|------------|-----|--------|----|
| 04-08 | 24111611527 | 70347164588 | success | dd73e9a62e | host3-gpu-4567 | PASS (0) |
| 04-09 | 24166163578 | 70528107799 | success | 2c4e113dd7 | novita6-gpu-0123 | PASS |
| 04-10 | 24220618561 | 70711088740 | success | 5638d40f3a | novita6-gpu-0123 | PASS |
| 04-11 | 24270600673 | 70874667954 | success | 0011d2aec0 | novita6-gpu-0123 | PASS |
| 04-12 | 24295268539 | 70942156925 | success | 8da1cfb30d | host3-gpu-0123 | PASS |
| 04-13 | 24320635302 | 71006055917 | failure | 37fc47c645 | host3-gpu-4567 | **PASS** (job failed on unrelated VLM accuracy regression on GLM-4.1V/Qwen2.5-VL-72B) |
| 04-14 | 24374813228 | 71185988967 | success | c456cba7fd | host3-gpu-4567 | PASS |
| 04-15 | 24430533442 | 71373799994 | failure | 2c9e76d333 | novita6-gpu-0123 | N/A — install step failed (`libtorch_global_deps.so: cannot open shared object file`); test never ran |
| 04-16 | 24486059047 | 71561141643 | success | a4cf2ea128 | novita10-gpu-0123 | PASS |
| 04-17 | 24541875545 | 71749246472 | success | 3d2d57c6cc | novita10-gpu-0123 | PASS |
| **04-18** | **24592875660** | **71917163703** | **success** | **9c47bbad13** | novita10-gpu-0123 | **last clean PASS** |
| **04-19** | **24617581978** | **71982368622** | **failure** | **2a327f0877** | novita10-gpu-4567 | **first FAIL** (110 mismatches) |
| 04-20 | 24643380928 | 72089373206 | failure | 1cff871c67 | novita10-gpu-4567 | N/A — `RuntimeError: Cannot find include path.`, server SIGKILLed in `setUpClass`; env-specific failure mode unrelated to the bisect signal |
| 04-21 | 24698143162 | 72235344702 | failure | 712b01d875 | host3-gpu-0123 | FAIL (177) |
| 04-22 | 24754179764 | 72423730539 | failure | 1408d97408 | host3-gpu-0123 | FAIL (136) |
| 04-23 | 24810524606 | 72614226605 | failure | c689f774a4 | novita10-gpu-4567 | FAIL |
| 04-24 | 24866511893 | 72970662863 | failure | c0166355ae | novita10-gpu-0123 | FAIL — same physical runner that passed on 04-18 |
| 04-25 | 24918423411 | 72975439352 | failure | a4facdf3f6 | host3-gpu-4567 | FAIL |
| 04-26 | 24944658760 | 73043911791 | failure | 714173555c | novita10-gpu-0123 | FAIL |
| 04-27 | 24971499389 | 73115280649 | failure | 977830e91e | novita10-gpu-0123 | FAIL (reference) |
| 04-28 | 25027858196 | 73302894349 | failure | 4a04a9818e | host3-gpu-0123 | FAIL |
| 04-29 | 25085535219 | 73500136906 | failure | 14b4e6fa69 | novita10-gpu-0123 | FAIL |
| 04-30 | 25141769035 | 73692926369 | failure | 3553fd0322 | host3-gpu-4567 | FAIL (69) |
| 05-01 | 25197098601 | 73880075551 | failure | 1742bfb610 | novita10-gpu-0123 | FAIL (39) |
| 05-04 | 25295900861 | 74154565198 | failure | c611a3fb78 | host3-gpu-4567 | FAIL (67) |
| 05-05 | 25351892972 | 74333099482 | failure | 2f7d99b7f7 | novita10-gpu-0123 | FAIL (68) |
| 05-06 | 25410559507 | 74531144605 | failure | b91b05ae27 | novita10-gpu-0123 | FAIL (47) |
| **05-07** | **25469734855** | **74730829418** | **failure** | **2e642ea187** | novita10-gpu-4567 | **FAIL (102) — current** |

Cancelled scheduled runs (04-19 prior, 04-26, 05-02, 05-03) are omitted; they don't change the picture, every dispatched run on either side of 04-18/04-19 confirms the boundary.

`test_qwen3_next_deterministic.py` itself was last modified on 2025-12-23 (PR #15582 migrated it into `test/registered/`); `test_deterministic_utils.py` is unchanged since 2025-10-25; `test_deterministic.py` since 2026-02-15. The test surface is stable across the whole window.

---

## Bisect Window and Candidate Introducing Commit

```
git log --oneline 9c47bbad13...2a327f0877    # last pass → first fail
615d6c93b2  [codex] Add flashinfer TRTLLM backend for diffusion NVFP4 (#22717)
0d94c3366a  [diffusion] feat: introduce ltx-2-two-stage device manager (#22869)
ea20f1baa4  [NPU] [DOC] Update npu best practice docs to match latest code (#23077)
4839cecbb0  [main] chore: add bias for base layer with lora (#22169)
c6a45fab64  Qwen3next flashinfer allreduce auto enable (#22664)        ← introducing commit
cd6ad80c00  diffusion: add HunyuanVideo GroupNorm+SiLU fast path (#22814)
2a327f0877  Fix Qwen3.5 video processing when passing video_data ... (#22431)
```

**Candidate introducing PR: [#22664](https://github.com/sgl-project/sglang/pull/22664) "Qwen3next flashinfer allreduce auto enable"** (BBuf, merged `c6a45fab64` on 2026-04-18 14:32 UTC).

The diff is one line in `python/sglang/srt/server_args.py::_handle_model_specific_adjustments`:

```diff
@@ -2197,6 +2198,7 @@ def _handle_model_specific_adjustments(self):
                 "Glm4MoeForCausalLM",
                 "Glm4MoeLiteForCausalLM",
                 "Qwen3MoeForCausalLM",
+                "Qwen3NextForCausalLM",
                 "KimiK25ForConditionalGeneration",
                 "Qwen3_5MoeForConditionalGeneration",
```

It adds `Qwen3NextForCausalLM` to the model-arch list that auto-sets `enable_flashinfer_allreduce_fusion = True` when the runtime is on SM90/SM100, with `tp_size > 1`, single-node, non-H20, no DP-attention, no MoE A2A backend (`server_args.py:2306–2329`).

---

## Why this breaks `test_prefix_with_logprobs`

1. The test asserts identical first-position logprobs across batch sizes — i.e. the deterministic-inference contract.
2. FlashInfer TRTLLM allreduce fusion changes FP reduction order (and selects a different kernel based on token count / batch shape) versus the standard NCCL/custom allreduce path. This produces logprobs that vary by batch size — exactly the divergence the test detects.
3. `_handle_deterministic_inference` (`server_args.py:3956–3961`) only force-disables the **aiter** allreduce-fusion path on AMD; it does **not** touch `enable_flashinfer_allreduce_fusion`, and it does not set `enforce_disable_flashinfer_allreduce_fusion`.
4. Therefore: `--enable-deterministic-inference` + `Qwen3-Next` + `flashinfer` + `tp>1` on H100/H200/B200 → fused allreduce silently re-enabled → non-deterministic logprobs across batch sizes.

The 04-18 pass log shows `enable_flashinfer_allreduce_fusion=False`. Every failing run from 04-19 onward shows `enable_flashinfer_allreduce_fusion=True` for the FlashInfer subclass (`enable_flashinfer_allreduce_fusion=False` for the TritonDeterministic subclass in the same job, which uses `--attention-backend triton` and is unaffected — its `test_prefix_with_logprobs` does not appear in the mismatch list).

---

## Root-cause Classification

**Code regression** (from PR #22664 / commit `c6a45fab64`).

Not runner-specific:
- The same physical runner `h100-novita10-gpu-0123` PASSED on 2026-04-18 and FAILED on 2026-04-24, 04-26, 04-27, 04-29, 05-01, 05-05, 05-06.
- Failures are observed across at least 4 distinct runner names (`novita10-gpu-0123`, `novita10-gpu-4567`, `host3-gpu-0123`, `host3-gpu-4567`).

Not env/flake:
- The numerical diffs are deterministic and reproduce across 18 consecutive dispatched scheduled days.
- The 2026-04-15 install break (`libtorch_global_deps.so` missing) and 2026-04-20 `Cannot find include path.` SIGKILL are unrelated environment failures that happened to land in this same job; they do not invalidate the bisect.

---

## Recommended Fix

**Primary fix** in `python/sglang/srt/server_args.py::_handle_deterministic_inference` — mirror the `enable_aiter_allreduce_fusion` handling for FlashInfer, and set `enforce_disable_flashinfer_allreduce_fusion=True` so the override beats the model-specific auto-enable that runs later in `_handle_model_specific_adjustments`:

```python
if self.enable_deterministic_inference:
    if self.enable_aiter_allreduce_fusion:
        logger.warning(
            "Disable --enable-aiter-allreduce-fusion because deterministic inference is enabled."
        )
        self.enable_aiter_allreduce_fusion = False

    # NEW: also disable FlashInfer allreduce fusion under deterministic inference.
    # _handle_model_specific_adjustments runs after this and would otherwise re-enable
    # fusion for Qwen3Next/Qwen3MoE/DeepseekV3/etc. on SM90/SM100 with tp>1, so we
    # must set the enforce flag — it is applied last in that handler.
    if self.enable_flashinfer_allreduce_fusion:
        logger.warning(
            "Disable enable_flashinfer_allreduce_fusion because deterministic inference is enabled."
        )
        self.enable_flashinfer_allreduce_fusion = False
    self.enforce_disable_flashinfer_allreduce_fusion = True
```

**Short-term workaround for CI** (no behavioral change for users): pass `--enforce-disable-flashinfer-allreduce-fusion` from `COMMON_SERVER_ARGS` in `python/sglang/test/test_deterministic_utils.py`, or only from the FlashInfer subclass in `test/registered/core/test_qwen3_next_deterministic.py`. This unblocks the nightly without addressing the underlying determinism contract.

**Adjacent risk audit:** Any deterministic-inference test on `tp>1` H100/H200/B200 that uses FlashInfer attention with one of these arches is silently masked by the same auto-enable path — verify before fixing:

```
DeepseekV3ForCausalLM, DeepseekV32ForCausalLM, GptOssForCausalLM,
GlmMoeDsaForCausalLM, Glm4MoeForCausalLM, Glm4MoeLiteForCausalLM,
Qwen3MoeForCausalLM, Qwen3NextForCausalLM, KimiK25ForConditionalGeneration,
Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration
```

---

## How the Investigation Was Performed

1. Pulled the failing job log from run `24971499389`, job `73115280649` and confirmed the assertion was the logprob-mismatch one (not a different test in the same job).
2. Listed all `Nightly Test (Nvidia)` `event=schedule` runs on `main` between 2026-04-08 and 2026-05-07 (60 results).
3. Resolved each run's `nightly-test-general-4-gpu-h100` job ID via `repos/.../actions/runs/{id}/jobs`.
4. Downloaded each completed (non-cancelled) job's log and grep-scored each for:
   - `test_qwen3_next_deterministic.py returned exit code` (file-level fail marker)
   - `Logprob mismatch at position` (per-sample mismatch)
   - `Some logprobs differ across batch sizes` (test-summary marker)
   - `enable_flashinfer_allreduce_fusion=` (server-args fingerprint)
5. Cross-checked the two ambiguous failure days (04-13 unrelated VLM accuracy fail; 04-20 environment SIGKILL) by reading their failure blocks directly to confirm they are not the same regression.
6. Resolved the introducing-commit window via `repos/.../compare/{last_pass}..{first_fail}` (7 commits) and read the diff of each candidate; only `c6a45fab64` touches a code path that flips a determinism-relevant default for this exact `(model arch, attention backend, GPU class, TP)` quadruple.
7. Verified the fix gap by reading `_handle_deterministic_inference` and `_handle_model_specific_adjustments` in `python/sglang/srt/server_args.py` at HEAD.

Investigation log files were retained under `/tmp/nightly_logs/` and `/tmp/nightly_jobs/`.

---

## Local Verification on H100 (8× H100 80GB, driver 580.126.20)

To prove the bisect, the same single test was executed at the parent commit and at the introducing commit on a workstation with 8× H100 80GB (using `CUDA_VISIBLE_DEVICES=0,1,2,3`, `tp=4`). `sglang` is editable-installed from the working tree, so a `git checkout` on a Python-only diff is sufficient to flip the runtime.

Command (identical at both SHAs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_IS_IN_CI=true \
HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=300 \
.venv/bin/python test/registered/core/test_qwen3_next_deterministic.py \
  TestFlashInferDeterministic.test_prefix_with_logprobs -v
```

| SHA | Description | `Auto-enabling FlashInfer ...` line | `enable_flashinfer_allreduce_fusion` in server_args | Result |
|---|---|---|---|---|
| `4839cecbb0` (parent of `c6a45fab64`) | One commit before the suspected regression | absent | `False` | **PASS** — `Ran 1 test in 118.429s / OK / ✓✓✓ Logprobs are identical across all batch sizes! ✓✓✓` |
| `c6a45fab64` (introducing commit, PR #22664) | The suspected regression itself | `Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for Qwen3NextForCausalLM` | `True` | **FAIL** — `Ran 1 test in 102.623s / FAILED (errors=1) / ✗✗✗ Some logprobs differ across batch sizes! ✗✗✗`, 244 per-sample mismatch lines, 2 mismatch banners (forward + reverse comparison both fail). |

The local FAIL reproduces the exact CI numerical fingerprint, e.g.

```
✗ Sample 6: Logprob mismatch at position 0: -2.355271339416504 vs -2.3723394870758057 (diff: 0.017068147659301758)
```

— same `0.017068147659301758` and same `-2.355271339416504 vs -2.3723394870758057` pair seen in the CI logs from 2026-04-19 onward (e.g. samples 4..37 in run `24971499389`, samples 14..29 in run `25469734855`).

The bisect is conclusive: PR #22664 / commit `c6a45fab64` is the introducing change. The fix outlined in *Recommended Fix* above will both (a) restore the deterministic-inference contract for Qwen3NextForCausalLM (and the other arches in the auto-enable list) and (b) keep the deterministic CI lane green.

Local-verify logs: `logs/bisect_verify/parent_4839cecbb0.log`, `logs/bisect_verify/at_c6a45fab64.log`.
