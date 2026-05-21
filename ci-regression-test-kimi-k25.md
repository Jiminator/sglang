# CI Regression Bisection Report — `TestKimiK25.test_kimi_k25`

**Failing test:** `test/registered/8-gpu-models/test_kimi_k25.py::TestKimiK25::test_kimi_k25`
**Workflow:** `Nightly Test (Nvidia)` → job `nightly-test-general-8-gpu-h200 (2)`
**First detected:** 2026-05-19 nightly run (UTC).
**Bisected regression:** commit `abe2ec2af` — PR [#25390](https://github.com/sgl-project/sglang/pull/25390) ("[AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model.")
**Recommended fix:** narrow the `n_routed_experts` allow-list back to `{256}` for the non-AMD/non-Quark code path, or gate the new `{256, 384}` allow-list on `is_quark_quantized(quant_config)`.

---

## 1. Failure Signature

```
FAIL: test_kimi_k25 (__main__.TestKimiK25.test_kimi_k25)
Run performance and accuracy for all Kimi-K2.5 variants.
Traceback (most recent call last):
  File ".../test/registered/8-gpu-models/test_kimi_k25.py", line 52, in test_kimi_k25
    run_combined_tests(...)
  File ".../python/sglang/test/run_combined_tests.py", line 253, in run_combined_tests
    raise AssertionError(f"Tests failed:\n{failure_summary}")
AssertionError: Tests failed:
  Model 1 (moonshotai/Kimi-K2.5 [TP8]):       accuracy - Accuracy 0.000 below baseline 0.920
  Model 2 (moonshotai/Kimi-K2.5 [TP8+DP8]):   accuracy - Accuracy 0.000 below baseline 0.920
```

Both variants — TP8 and TP8+DP8 — score `gsm8k = 0.000` against a 0.920 baseline. The server starts cleanly, accepts requests, and reports 200 OKs; output is just garbage. **Deterministic across both nightly runs and our local repro.**

---

## 2. CI Boundary

Three consecutive scheduled `Nightly Test (Nvidia)` runs:

| Date (UTC) | Run ID | head SHA | gsm8k score | Result |
|---|---|---|---:|---|
| 2026-05-18 01:02 | [26008032501](https://github.com/sgl-project/sglang/actions/runs/26008032501) | `b3803164c` | 0.941 / 0.942 | **PASS** ✓ |
| 2026-05-19 03:14 | [26069709241](https://github.com/sgl-project/sglang/actions/runs/26069709241) | `dbac46472` | 0.000 / 0.000 | **FAIL** ✗ |
| 2026-05-20 03:26 | [26134864664](https://github.com/sgl-project/sglang/actions/runs/26134864664) | `7f154ba44` | 0.000 / 0.000 | **FAIL** ✗ |

76 commits in the `b3803164c..dbac46472` window. One of them — and only one — touches files that could plausibly affect the Kimi-K2.5 forward pass with a sufficiently large blast radius:

> `abe2ec2af` — [AMD] Enable shared-experts fusion with new KIMI-K2.5-MXFP4 model. (#25390)
> Touches `python/sglang/srt/models/deepseek_v2.py` (+12 / −1) and `python/sglang/srt/layers/quantization/quark/quark.py` (+7 / −1).
> Merged `2026-05-18T08:30:58Z` — squarely between the 05-18 pass and the 05-19 fail.

---

## 3. Root Cause

`deepseek_v2.py::determine_num_fused_shared_experts` decides whether to enable the shared-experts fusion optimization. Before PR #25390 the allow-list was a single value `n_routed_experts == 256` (DeepSeek-V3 / R1). The PR extended that allow-list to `{256, 384}`:

```diff
 elif (
     self.config.architectures[0] != architecture
-    or self.config.n_routed_experts != 256
+    # Allow-list of n_routed_experts values that have been validated
+    # for shared-experts fusion under this code path. Currently:
+    #   256 -> DeepSeek-V3 / R1
+    #   384 -> Kimi-K2.5 (text_config wraps DeepseekV3ForCausalLM)
+    or self.config.n_routed_experts not in (256, 384)
     or self.config.n_shared_experts != 1
 ):
     disable_reason = "Config does not support fused shared expert(s)."
```

Kimi-K2.5 has `n_routed_experts=384`, so **after the PR, fusion fires for every Kimi-K2.5 load**, including on the H200 nightly job which uses the public `moonshotai/Kimi-K2.5` checkpoint (`compressed-tensors` quant — *not* the AMD Quark MXFP4 checkpoint the PR was validated against).

The fusion path assumes the checkpoint's shared-expert weights have already been folded into the routed-expert tensors during conversion. With the standard `moonshotai/Kimi-K2.5` checkpoint the loose tensors are still present and the fused-name lookup misses them. The smoking-gun loader warnings observed during the suspect run on H200:

```
[DP2 TP2] model.layers.59.mlp.experts.w13_weight not found in params_dict.
[DP2 TP2] model.layers.60.mlp.experts.w2_weight  not found in params_dict.
[DP2 TP2] model.layers.60.mlp.experts.w13_weight not found in params_dict.
... (many more, every MoE layer)
```

(No such warnings on the parent commit's run.) Per-layer weight memory also diverges: TP8 `Load weight end. ... mem usage=72.19 GB` on the parent vs. `84.39 GB` on the suspect — fusion adds ~12 GB of merged tensors, but the underlying source tensors never made it in, so the forward pass attends over uninitialized memory.

The PR's own author validated only **MI355X + AMD MXFP4 Quark checkpoint**, where the loose shared-expert tensors really *are* pre-fused. The H200 + `compressed-tensors` path was not exercised in the PR's CI.

---

## 4. Empirical Verification

We reproduced the failure and the fix on a single 8×H200 node (no PD-disagg). All three runs use the same test file, same env, same checkpoint cache:

| Run | sglang HEAD | Model 1 (TP8) | Model 2 (TP8+DP8) | OVERALL |
|---|---|---:|---:|---|
| **A — latest main** | `3a6de13cd` | gsm8k = 0.000 ✗ (Note 1) | gsm8k = **0.000** ✗ | **FAILED** |
| **B — suspect**     | `abe2ec2af` (PR #25390) | gsm8k = **0.000** ✗ | (killed early, but warnings match) | **FAIL** |
| **C — suspect's parent** | `e5589843a` | gsm8k = **0.947** ✓ | gsm8k = **0.946** ✓ | **PASSED** |

> Note 1: On the latest-main run, Model 1's server hit a startup timeout (an unrelated transient); Model 2 still completed the full gsm8k eval and scored exactly 0.000, matching the CI nightly signature. The suspect run reproduces the model-1 gsm8k=0.000 directly with no startup issue.

Both perf channels still produce tokens at >2000 tok/s on the failing commits, confirming the breakage is silent output corruption (model.generate keeps running, just emits garbage) rather than a crash.

**Conclusion: PR #25390 is the regression. Reverting / gating it restores accuracy 0.94+.**

---

## 5. Recommended Fix

Three options, listed from least to most invasive:

1. **Narrow the allow-list back to `{256}`** for the non-AMD/non-Quark path. The AMD MXFP4 checkpoint that motivated this PR is loaded through a different code path on AMD anyway — the H200/B200 hot path doesn't need `384` in the fusion allow-list.
2. **Gate the `384` entry on the active quantization backend.** Allow `384` only when `isinstance(quant_config, QuarkConfig)`, so the standard `compressed-tensors` load on H200 continues to take the safe disabled path.
3. **Implement the actual fused weight loading for the `moonshotai/Kimi-K2.5` (`compressed-tensors`) checkpoint format** so the existing fusion path can serve both checkpoint flavors. This is the proper fix but a larger change.

In the meantime, users running Kimi-K2.5 with the standard checkpoint can work around with `--disable-shared-experts-fusion`.

---

## 6. Reproduction Recipe

8×H200 node, `moonshotai/Kimi-K2.5` cached locally.

```bash
# Pin to suspect — fails (gsm8k = 0.000)
git checkout abe2ec2af
cd /sgl-workspace/sglang
python3 test/registered/8-gpu-models/test_kimi_k25.py

# Pin to parent — passes (gsm8k ≈ 0.947)
git checkout e5589843a
python3 test/registered/8-gpu-models/test_kimi_k25.py
```

Each run takes ~30 min wall-clock (perf + accuracy for both variants).

---

## 7. Timeline

- 2026-05-18 01:01 UTC — Last good nightly (head `b3803164c`).
- 2026-05-18 08:30 UTC — PR #25390 merged at `abe2ec2af`.
- 2026-05-19 03:14 UTC — First failing nightly (head `dbac46472`).
- 2026-05-20 03:26 UTC — Second failing nightly (head `7f154ba44`).
- 2026-05-21       UTC — Reproduced + bisected + verified on Jimmy's prefill devbox.

---

*Bisection performed on an 8×H200 box (no PD-disagg, single-node, TP=8 / TP8+DP8 per the test's two variants). All three runs (main, suspect, parent) executed back-to-back on the same hardware with clean GPU state between checkouts.*
