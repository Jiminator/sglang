# CI Regression RCA — `test_a_gsm8k` (TestEagleDPAttnServerLarge)

**Test:** `test/registered/spec/eagle/test_eagle_infer_beta_dp_attention_large.py::TestEagleDPAttnServerLarge::test_a_gsm8k`
**Suite:** `nightly-test-specialized-8-gpu-b200` (workflow *Nightly Test (Nvidia)*)
**Model/config:** `nvidia/DeepSeek-V3-0324-FP4`, `--tp-size 4 --dp-size 4 --enable-dp-attention`, `--attention-backend trtllm_mla`, `--moe-runner-backend flashinfer_trtllm`, `--quantization modelopt_fp4`, EAGLE MTP (`--speculative-eagle-topk 1`), `--kv-cache-dtype fp8_e4m3`.

---

## Failure Signature

The server **crashes during the GSM8K eval**, not on a score/accept-length assertion:

```
/pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:109: _assert_async_cuda_kernel:
block: [0,0,0], thread: [0,0,0] Assertion `NaN detected! draft_extend_for_prefill` failed.
[DP0 TP0] Scheduler hit an exception: ...
ERROR: test_a_gsm8k (__main__.TestEagleDPAttnServerLarge)
```

- The NaN is produced by the **EAGLE/MTP draft model's prefill-extend forward** (`EagleDraftWorker._draft_extend_for_prefill` in `eagle_worker_v2.py`). The async probe `maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")` turns the silent NaN into a hard `torch._assert_async`, which kills the scheduler.
- The probe only fires because the test wraps server launch in `envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True)`. Without it the NaN would silently corrupt the draft tokens (degraded acceptance/accuracy) instead of crashing.
- Identical signature in all three reported CI runs (2026‑05‑28 `421bda6d`, 05‑29 `a8cfae0b`, 05‑31 `9b4be9c5`). **Deterministic in CI (3/3).**
- It fires specifically on **multi-sequence chunked-prefill batches under load**, e.g. `Prefill batch, #new-seq: 6, #new-token: 4096, #queue-req: 21`.

---

## Root-Cause Classification

**Code regression — PR #23269** (`19663aafcd`, *"Support batch size > 1 when enable CP"*, merged 2026‑05‑27).

---

## Temporal Boundary (CI nightly history)

| Nightly run | SHA | `nightly-test-specialized-8-gpu-b200` |
|---|---|---|
| 2026‑05‑26 (01:00 UTC, pre-#23269) | `8f2a4e70` | **success** (our test passed, accept_len=3.00) |
| 2026‑05‑27 | `737c6cd6` | cancelled |
| 2026‑05‑28 | `421bda6d` | **failure** (NaN) |
| 2026‑05‑29 | `a8cfae0b` | failure (NaN) |
| 2026‑05‑31 | `9b4be9c5` | failure (NaN) |

Window = `8f2a4e70..421bda6d` (76 first-parent commits). PR #23269 landed inside it.

---

## The Bug

`ForwardBatch.prepare_mlp_sync_batch` (`python/sglang/srt/model_executor/forward_batch_info.py`) pads each DP rank's token count for DP-attention collective communication. #23269 changed the second alignment:

```python
attn_cp_size = get_attention_cp_size()
for i in range(sync_group_size):
-   global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size)        # before #23269
+   global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size * 2)    # #23269
```

The intent (per the new comment) was to "divide the tokens into `2 * CP` chunks for load balance" — meaningful only when **context parallel is enabled** (`attn_cp_size > 1`).

But this test (and every non-CP DP-attention deployment) runs with **`attn_cp_size = 1`**:
- **Before #23269:** `ceil_align(x, 1)` → **no-op** (no padding).
- **After #23269:** `ceil_align(x, 2)` → **pads every rank's token count up to an even number**.

So #23269 *introduced* DP-attention token padding where there was none. When a multi-sequence prefill batch's global token count is **odd**, an extra zero-padded dummy token is appended for the DP sync. The EAGLE/MTP **draft prefill-extend** path (`_draft_extend_for_prefill`, which rebuilds `input_ids`/`extend_lens` for the real tokens and re-runs the NextN draft over the same batch) does not account for this extra padded token; the resulting mismatch in the `trtllm_mla` draft attention (reading misaligned / uninitialized fp8 KV) produces **NaN** draft logits.

**Why it looked flaky locally but is reliable in CI:** the trigger requires a prefill batch with an *odd* global token count, which depends on request-arrival timing. Over a full 200-example / 128-thread GSM8K eval many such batches form, so it fires essentially every run (CI 3/3; local baseline 3/3 — see below). With light/low-concurrency traffic it may not trigger at all.

---

## Evidence

### Local reproduction (8× B200)
Faithful repro required matching CI's sharding **exactly**: `--tp-size 4 --dp-size 4 --enable-dp-attention` → world = 4 GPUs, **`attn_tp = 1`** (DP-attention world = `tp_size`; the "16 GPU" comment in the test is misleading — CI logs show ranks `DP0TP0/DP1TP1/DP2TP2/DP3TP3`, `max_total_num_tokens=783616`). Plus the **real GSM8K eval** (200 ex, 128 threads) to form multi-seq prefill batches. (`--tp-size 4 --dp-size 2`, i.e. `attn_tp=2`, is a *different* config that fails even at the passing commit — do not use it.)

### Bisection (faithful config: tp4×dp4 + real GSM8K eval)
| first-parent pos | commit | verdict |
|---|---|---|
| 0 | `8f2a4e70` (last CI pass) | **GOOD** (score 0.97) |
| 38 | `c317beda99` | GOOD (0.975) |
| 49 | `dea85c30f4` | GOOD (0.965) |
| 50 | `163b970127` (#26380 overlap WAR barrier) | GOOD (0.985) |
| 63 | `a95b4e2e09` | GOOD (0.97) |
| **66** | `ddf0627254` (**parent of #23269**) | **GOOD × 3/3** (0.97 / 0.98 / 0.97) |
| 67 | `19663aafcd` (**#23269**) | culprit (crashes in CUDA-graph capture on this exact commit) |
| 68 | `e06058ed62` (import-only, DSV4) | BAD (inherits #23269) |
| 70 | `24bcb37efb` | BAD |
| 76 | `421bda6d` (first CI fail) | **BAD** (NaN) |

### Causal test (decisive) — at `421bda6d`, toggling only the #23269 line
| Variant | Runs |
|---|---|
| `421bda6d` unmodified | **BAD × 3/3** (NaN, score 0.0) |
| `421bda6d` with line reverted to `ceil_align(..., attn_cp_size)` | **GOOD × 3/3** (score 0.965 / 0.97 / 0.98) |

Reverting that single line deterministically eliminates the NaN — confirming #23269's `attn_cp_size * 2` even-padding is the cause.

---

## Ruled Out
- **#26397** "reland skip-softmax topk==1" — its CUDA (`argmax`) path is identical to the original #26235 already present in the passing commit; net-zero across the boundary.
- **#26335** "async-assert probes; zero `tgt_cache_loc`" — only renames the detection flag (`SGLANG_SPEC_NAN_DETECTION` → `SGLANG_ENABLE_ASYNC_ASSERT`) and adds probes; the `tgt_cache_loc` change is in the V1 verify path. No V2 numeric change. (It is what makes the test *report* the NaN, but it does not *cause* it.)
- **#26380** "overlap WAR barrier" / **#26425** "`req_pool_indices_cpu` mirror" — both pass at the faithful config (pos 49/50 GOOD).

---

## Recommended Fixes

**Fix 1 — gate the even-padding to CP-enabled configs (minimal, recommended).** Restores exact pre-regression behavior for all non-CP DP-attention deployments while keeping #23269's CP load-balancing:

```python
attn_cp_size = get_attention_cp_size()
if attn_cp_size > 1:
    # CP splits each rank's tokens into 2*CP balanced chunks, so the padded
    # length must be a multiple of 2*CP. No-op (and unnecessary) without CP.
    for i in range(sync_group_size):
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size * 2)
```
The causal test above (reverting to `ceil_align(..., attn_cp_size)`, which is identical to this for `attn_cp_size == 1`) already validates this: 3/3 GOOD.

**Fix 2 — make the MTP draft-extend robust to DP-attention padding (deeper).** #23269 merely *exposed* a latent fragility: `_draft_extend_for_prefill` assumes the batch's token layout matches `extend_lens` exactly and does not handle DP-sync padding tokens. Give padding tokens valid (zeroed) KV-cache locations / mask them out of the draft attention so any future padding (CP or otherwise) is safe. Higher effort/risk but removes the underlying hazard.

**Not appropriate:** disabling the NaN probe — it is correctly surfacing a real numerical corruption that otherwise silently degrades draft acceptance/accuracy.

---

## Status on `main`
The offending line is **still present on latest `main`** (`b5d8a646a6`, `forward_batch_info.py:998`) — the regression is **unfixed**. Fix 1 has been applied locally to the working branch and validated on current `main` (gsm8k eval): **GOOD** (no NaN), confirming the fix resolves the regression on the current codebase.

## Reproduction Recipe
```bash
SGLANG_ENABLE_ASYNC_ASSERT=1 SGLANG_SPEC_NAN_DETECTION=1 SGLANG_SPEC_OOB_DETECTION=1 HF_HUB_OFFLINE=1 \
python3 -m sglang.launch_server --model-path <DeepSeek-V3-0324-FP4> \
  --tp-size 4 --dp-size 4 --enable-dp-attention --attention-backend trtllm_mla \
  --moe-runner-backend flashinfer_trtllm --quantization modelopt_fp4 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --kv-cache-dtype fp8_e4m3
# then run the gsm8k eval (200 examples, 128 threads) -> NaN in draft_extend_for_prefill
```
