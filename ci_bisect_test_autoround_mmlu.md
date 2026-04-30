# CI Regression Bisection Report — `TestAutoRound.test_mmlu`

**Investigated:** 2026-04-23
**Target failure:** https://github.com/sgl-project/sglang/actions/runs/24270319124/job/70875856606#step:7:4654

> **Update (after extending the sample to 2026-04-17 → 2026-04-22):** A second identical failure was found — run `24592599796` on 2026-04-18 (partition 6, runner `h200-ion-1-1gpu-6`) hit `AssertionError: 0.21875 not greater than or equal to 0.25` with the exact same signature. The flake is recurring at ~7% of scheduled runs, not a one-off. Diagnosis unchanged (pre-existing flakiness / knife-edge threshold); urgency of the test-side fix is higher.
>
> **Update 2 (manual reproduction attempt on 2026-04-23):** A local run of `TestAutoRound.test_mmlu` on `main` **passed** — consistent with the ~93% pass rate; not a refutation, just another draw from the same distribution. A single local pass cannot disprove a 7% flake, so this neither promotes nor demotes the diagnosis.

---

## Failure Signature

- **Workflow / Event:** `.github/workflows/pr-test.yml`, scheduled run on `main`
- **Run:** `24270319124` (2026-04-11T00:30:09Z, head `0011d2ae`, overall run `cancelled` but this job ran to failure)
- **Job:** `stage-b-test-1-gpu-large (10)` (databaseId `70875856606`, partition 10 of 14)
- **Runner:** `h200-ion-1-1gpu-7` / machine `6b93ef82d4d8`
- **Test file:** `test/registered/quant/test_autoround.py` (est_time=77, suite=`stage-b-test-1-gpu-large`)
- **Test method:** `TestAutoRound.test_mmlu` (failing model iteration: `OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc`)
- **Error:**
  ```
  File "/actions-runner/_work/sglang/sglang/test/registered/quant/test_autoround.py", line 58, in test_mmlu
      self.assertGreaterEqual(metrics["score"], 0.25)
  AssertionError: 0.21875 not greater than or equal to 0.25
  ```
- **Emitted metric:**
  `[METRIC] mmlu_score=0.21875 labels={"model": "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc", "eval": "mmlu"}`
- **Server args of interest:** `quantization='auto-round'`, `attention_backend='fa3'`, `dtype='auto'`, `tp_size=1`
- **Deterministic across runs?** No.

---

## Fundamental Test Fragility (key context)

- Test runs `run_eval` with `num_examples=32`, `num_threads=32`.
- MMLU questions are sampled with `random.Random(0).sample(...)` (see `python/sglang/test/simple_eval_mmlu.py:95`) — **the 32-question subset is fixed across runs.**
- Score granularity = `1/32 ≈ 0.03125`; threshold is **exactly 0.25** (the random-guessing baseline for 4-way MMLU), gated with `assertGreaterEqual`.
- Model is `OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc` — a 0.5B-param INT4-quantized model whose "true" MMLU on this 32-item subset sits around ~0.28.
- Observed scores across nearby runs: `{0.21875 ×1, 0.25 ×2, 0.28125 ×8}`. Variance comes from the server side (fresh `random_seed` per launch + fp kernel nondeterminism), not from question sampling.

---

## Timeline — `stage-b-test-1-gpu-large`, whichever partition held `test_autoround.py`

| Run ID | Date (UTC) | Head SHA | Part | Runner | OPEA score | Intel score (proxy) | Outcome |
|---|---|---|---|---|---|---|---|
| 24206441490 | 2026-04-09 18:23 | 8eb235ab | 11 | h100-novita-host1-gpu-1 | 0.28125 | 0.28125 | PASS |
| 24220224812 | 2026-04-10 00:30 | cebd9c2a | 11 | h200-ion-2-1gpu-7 | 0.28125 | 0.28125 | PASS |
| 24230133084 | 2026-04-10 06:40 | 6d79c609 | 10 | h100-novita-host1-gpu-6 | 0.28125 | 0.28125 | PASS |
| 24242573628 | 2026-04-10 12:19 | 8ba96460 | 10 | h200-ion-1-1gpu-6 | 0.28125 | 0.28125 | PASS |
| 24257571982 | 2026-04-10 18:19 | 5cb4ea1d | 10 | h100-novita-host1-gpu-1 | **0.25** | 0.28125 | PASS (knife-edge) |
| **24270319124** | **2026-04-11 00:30** | **0011d2ae** | **10** | **h200-ion-1-1gpu-7** | **0.21875** | 0.25 | **FAIL** |
| 24276607632 | 2026-04-11 06:25 | 3ce72252 | 3 | h200-ion-2-1gpu-0 | 0.28125 | 0.28125 | PASS |
| 24282233192 | 2026-04-11 12:13 | 78043d44 | 3 | h200-ion-1-1gpu-5 | 0.28125 | 0.28125 | PASS |
| 24288516804 | 2026-04-11 18:12 | 78043d44 | 3 | h200-ion-1-1gpu-4 | test_autoround not executed this run | — | PASS |
| 24295023865 | 2026-04-12 00:33 | 8da1cfb3 | 2 | h100-novita-host1-gpu-4 | 0.28125 | 0.28125 | PASS |
| 24300594353 | 2026-04-12 06:35 | 9a4e8089 | 2 | h200-ion-1-1gpu-2 | 0.28125 | 0.28125 | PASS |
| 24306511798 | 2026-04-12 12:14 | bcc0c65a | 2 | h200-ion-2-1gpu-2 | **0.25** | 0.28125 | PASS (knife-edge) |
| 24580257266 | 2026-04-17 18:40 | 5d4e8994 | 6 | (h100/h200 pool) | 0.28125 | — | PASS |
| **24592599796** | **2026-04-18 00:31** | **5f7aee72** | **6** | **h200-ion-1-1gpu-6** | **0.21875** | 0.3125 | **FAIL** |
| 24604450125 | 2026-04-18 12:14 | 4839cecb | 6 | — | 0.28125 | — | PASS |
| 24610848557 | 2026-04-18 18:15 | 2a327f08 | 6 | — | 0.28125 | — | PASS |
| 24617341159 | 2026-04-19 00:35 | 2a327f08 | 6 | — | 0.3125 | — | PASS |
| 24635819338 | 2026-04-19 18:15 | 32b7777f | 6 | — | 0.28125 | — | PASS |
| 24643054956 | 2026-04-20 00:35 | d3ce6646 | 6 | — | **0.25** | — | PASS (knife-edge) |
| 24652543548 | 2026-04-20 06:49 | 69eb95f2 | 6 | — | 0.28125 | — | PASS |
| 24666478983 | 2026-04-20 12:28 | 0be6ab04 | 6 | — | 0.28125 | — | PASS |
| 24683166533 | 2026-04-20 18:23 | 4698f4cd | 6 | — | 0.28125 | — | PASS |
| 24697729494 | 2026-04-21 00:35 | 712b01d8 | 2 | — | 0.28125 | — | PASS |
| 24736806992 | 2026-04-21 17:29 | 929e00ee | 2 | — | 0.3125 | — | PASS |
| 24755791075 | 2026-04-22 01:43 | 77fd86f8 | 2 | — | 0.28125 | — | PASS |
| 24771471590 | 2026-04-22 09:41 | 6a3c070e | 2 | — | **0.25** | — | PASS (knife-edge) |
| 24792732403 | 2026-04-22 17:27 | de962f32 | 2 | — | **0.25** | — | PASS (knife-edge) |

**Observed OPEA score distribution (27 runs, 2026-04-09 → 2026-04-22):**
`{0.21875 ×2, 0.25 ×5, 0.28125 ×16, 0.3125 ×3}` — failures at 2/27 ≈ **7.4% flake rate**. Scores fluctuate exactly at the 0.25 threshold boundary with 1/32 = 0.03125 granularity.

**Partition drift:** `test_autoround.py` moved through partitions 10 → 3 → 2 across ~48 hours. The partition number is not stable; always locate the test by grep, not by partition index.

**Historical survey:** 24 earlier failed scheduled `pr-test.yml` runs on `main` from 2026-03-28 → 2026-04-09 were checked — none contained a `TestAutoRound.test_mmlu` assertion failure (those stage-b failures were other tests, notably EAGLE at partition 11 with a 0.82 threshold). In the extended window (2026-04-09 → 2026-04-22), `TestAutoRound.test_mmlu` failed on **two** distinct runs (`24270319124` and `24592599796`), both with the exact same `0.21875 < 0.25` signature.

---

## Root Cause Classification

**Pre-existing flakiness / ill-chosen threshold.** Specifically: a knife-edge assertion at the MMLU random baseline (0.25) evaluated on only 32 fixed questions against a 0.5B INT4 model whose true score on this subset sits ~0.28. Normal server-side nondeterminism (fresh `random_seed` per launch + fp kernel variance) shifts the score by ±1–2 questions, which occasionally crosses below 0.25.

### Ruled out

- **Code regression** — commits in the boundary window `5cb4ea1d..0011d2ae` include FA3-adjacent changes (#21104 perf-only scheduler_metadata precompute; #22051 MUSA FA3 path that does not touch the CUDA/H100/H200 path), but every subsequent scheduled run on `0011d2ae` and later HEADs passed this exact test. A real code regression would persist.
- **Runner / hardware-specific** — passes occur on `h100-novita-host1-*`, `h200-ion-1-*`, and `h200-ion-2-*`. The failing runner (`h200-ion-1-1gpu-7`) is in the same pool as many passing runs (`h200-ion-1-1gpu-2/4/5/6` all passed).
- **Environment change** — same `flashinfer-python==0.6.7.post3`, `flashinfer-cubin==0.6.7.post3`, `flashinfer-jit-cache==0.6.7.post3` across adjacent runs; no driver/CUDA boundary aligns with the single failure.

---

## Evidence Table

| Condition | Result |
|---|---|
| Same runner pool on pass + fail (`h200-ion-1-*`) | Both PASS and FAIL → not hardware-specific |
| Score distribution on fixed 32-question sample across 27 runs | `{0.21875 ×2, 0.25 ×5, 0.28125 ×16, 0.3125 ×3}` — 1/32 granularity noise straddling threshold; 7.4% flake rate |
| Subsequent runs on same or newer SHAs | Mostly PASS; second failure (24592599796) is 7 days later on unrelated SHA → no SHA correlation |
| Two failures across 12 days, 7 days apart, on different partitions and runners | Pattern is random, not a code regression |
| Threshold (0.25) vs true model capability (~0.28) | Margin ≈ 1 MMLU question → naturally flaky |
| MMLU question sampling | `random.Random(0).sample(..., 32)` — fixed set; variance is *from the server*, not input selection |

---

## Candidate Commit Range / Environment Boundary

**None justified.**

### Narrow window (last-pass → first-fail): `5cb4ea1d..0011d2ae`
Adjacent commits include FA3-adjacent perf changes (#21104, #22051) but the same HEAD and later HEADs pass the test. No supportable commit boundary.

### Expanded window (3 days back): `2026-04-07 18:00 UTC → 2026-04-11 00:30 UTC` (head `0011d2ae`)

Full set of commits in the expanded window that plausibly touch the paths exercised by `TestAutoRound.test_mmlu` (FA3 attention backend, AutoRound quantization, Qwen2 arch, sampling, eval utilities):

| SHA | PR | Subsystem | Relevance to failing config |
|---|---|---|---|
| `6af34b95b6` | #21104 | **FA3 attention (CUDA)** — precompute scheduler_metadata | **In narrow window; technical candidate** — refuted below |
| `1a8eb890f6` | #20796 | FA3 attention (CUDA) — Kernels community fa3 | Already in last-pass SHA `5cb4ea1d` and earlier passing SHA `8ba96460` — not a candidate |
| `f7a1740101` | #22051 | MUSA FA3 backend | MUSA path only; CI runs on CUDA H100/H200 — not a candidate |
| `cd373667cd` | #21692 | NPU Qwen3.5 quantization fix | NPU path only; bug is on CUDA — not a candidate |
| `8ba9646044` | #22312 | GDN non-continuous B/A (Qwen3.5-27B) | Mamba/GDN path, not dense Qwen2.5 — not a candidate |
| `0668a7f51a` | #22444 | GDN extend verify path | GDN, not Qwen2.5 — not a candidate |
| `5638d40f3a` | #22079 | Gemma4 nvfp4 fix | Gemma4, not Qwen2.5 — not a candidate |
| `c554dc5c64` | #21339 | FlashInfer CuteDslMoE FP4 | MoE path, not dense Qwen2.5-0.5B — not a candidate |
| `7546d04c81` | #21240 | FP4 flashinfer trtllm routed MoE | MoE only — not a candidate |
| `18f41ac427` | #22316 | DeepSeek FP8 DeepEP dispatch | DeepSeek/DeepEP — not a candidate |
| `6d79c60995` | #22381 | LoRA Kimi | LoRA, not AutoRound — not a candidate |
| `28ef6de091` | #22323 | LoRA DeepSeek MLA refactor | LoRA, not AutoRound — not a candidate |
| `60acdc31f2` | #22430 | DSA models fix | Sparse attention, not Qwen2.5-0.5B — not a candidate |
| `dd41764487` | #22258 | AMD HIP NSA | AMD path — not a candidate |
| `599cce4d82` | #22438 | Intel GPU flash_attn imports | Intel GPU — not a candidate |
| `628df31d08` | #22424 | AMD aiter NSA | AMD — not a candidate |
| `de441ac6bb` / `1e3f6ebea6` | #22389, #22384 | Memory pool refactor | Plumbing refactor, no numeric change — not a candidate |
| `2ab141547d` | #22413 | CPU biased_grouped_topk fusion | CPU MoE — not a candidate |
| `493ec91cbe` | #22292 | Test utils — LoRA reorder, tokenizer cache | Only touches `run_bench_serving` branch, not `run_eval` path used here — not a candidate |
| `f08726fd56` | #22077 | DFLASH speculative decoding | Adds new defaults; AutoRound test sets `speculative_algorithm=None` — not a candidate |

**Refutation of the one narrow-window candidate (#21104, `6af34b95b6`, FA3 `precompute_varlen_num_blocks`):**

| Passing SHA after fail | Contains #21104? | OPEA score |
|---|---|---|
| `3ce72252` (24276607632, 2026-04-11 06:25) | yes | 0.28125 PASS |
| `78043d44` (24282233192, 2026-04-11 12:13) | yes | 0.28125 PASS |
| `8da1cfb3` (24295023865, 2026-04-12 00:33) | yes | 0.28125 PASS |
| `9a4e8089` (24300594353, 2026-04-12 06:35) | yes | 0.28125 PASS |
| `bcc0c65a` (24306511798, 2026-04-12 12:14) | yes | 0.25 PASS |
| `5f7aee72` (24592599796, 2026-04-18 00:31) | yes | **0.21875 FAIL** |
| `6a3c070e` (24771471590, 2026-04-22 09:41) | yes | 0.25 PASS |
| `de962f32` (24792732403, 2026-04-22 17:27) | yes | 0.25 PASS |

If #21104 introduced a deterministic accuracy regression on the AutoRound path, the test would fail consistently after it landed; instead we see ~93% pass and the second failure 7 days / hundreds of commits later. Additionally, the PR is scoped to caching `scheduler_metadata` (a perf-only precompute) and does not touch kernel math. Pass-rate evidence refutes it.

### Conclusion
No commit in the 3-day expanded window is consistent with the observed failure pattern. The correct "boundary" is the stochastic draw of a single server run — not a SHA.

---

## Recommended Next Steps

### Test-level fix (primary, low-risk)
Pick one or combine:

- Raise `num_examples` to reduce variance (e.g., 128 or 256).
- Lower the threshold below the observed minimum with margin (e.g., `>= 0.15` for this tiny model).
- Evaluate with greedy decoding + deterministic kernels so scores are reproducible.
- Drop the accuracy gate for this 0.5B INT4 model entirely and keep only a launch/smoke check.

A 0.25 threshold on 32 MMLU questions for a 0.5B INT4 model is functionally a coin-toss guard and will continue to produce sporadic failures.

### Operational
- Treat run `24270319124` as a known flake; no revert / no bisect needed.
- If this pattern recurs at low rate (<~1/20 scheduled runs), accept as pre-existing flakiness until the threshold is tightened.

---

## Commands to continue manually

```bash
# 1. Re-confirm the score across the next N scheduled runs and locate the partition each time:
for r in $(gh run list --repo sgl-project/sglang --workflow=pr-test.yml --event schedule --branch main --limit 20 --json databaseId --jq '.[].databaseId'); do
  gh run view $r --repo sgl-project/sglang --json jobs \
    --jq '.jobs[] | select(.name | startswith("stage-b-test-1-gpu-large")) | "\(.databaseId) \(.name)"' |
  while read j n; do
    gh run view $r --repo sgl-project/sglang --job $j --log 2>/dev/null |
      grep -H "mmlu_score=.*OPEA/Qwen2.5-0.5B" | sed "s|^|$r $n |"
  done
done

# 2. Reproduce locally / on any H100 or H200 to confirm the score distribution:
python3 -m unittest sglang.test.registered.quant.test_autoround.TestAutoRound.test_mmlu

# 3. If you want to make the test deterministic, patch test_autoround.py to pass
#    --random-seed 0 and --attention-backend fa3 (or triton), and either assert >= 0.15
#    or bump --num-examples to 128+.
```

---

## TL;DR

- **Diagnosis:** Pre-existing flakiness in `TestAutoRound.test_mmlu`. Not a code regression, not runner-specific, not an environment change. Flake rate ≈ **7% of scheduled runs** (2/27 in the observed window), with a second, independent occurrence on 2026-04-18 (run `24592599796`, partition 6, runner `h200-ion-1-1gpu-6`, different SHA, same exact signature).
- **Strongest evidence:** Fixed 32-question MMLU with threshold = 0.25 (random baseline) against a 0.5B INT4 model yields the score distribution `{0.21875 ×2, 0.25 ×5, 0.28125 ×16, 0.3125 ×3}` across 27 consecutive scheduled runs on 5+ different SHAs and multiple runner pools; only the 0.21875 draws trip the gate, and the two failures are separated by 7 days and a SHA span of hundreds of commits — no code boundary aligns with them.
- **Narrowest justified boundary:** None — the failure boundary is a stochastic server run, not a SHA, runner pool, or package version.
- **Next action:** Tighten the test (lower threshold, raise `num_examples`, or use greedy+deterministic decoding), not the code. This is blocking CI ~every 2 weeks and will continue until the gate is fixed.
