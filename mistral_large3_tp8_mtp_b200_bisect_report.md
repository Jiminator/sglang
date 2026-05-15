# CI Regression Bisection Report — `test_mistral_large3_all_variants` (TP8+MTP variant) on B200

**Investigator:** Claude (sglang-bisect-ci-regression skill)
**Date:** 2026-05-15
**Repo:** `sgl-project/sglang`
**Reporting run:** https://github.com/sgl-project/sglang/actions/runs/25835354140/job/75909128362
**Related report:** see [`mistral_large3_nvfp4_b200_bisect_report.md`](mistral_large3_nvfp4_b200_bisect_report.md) for the **NVFP4** variant regression — same partition, *different* root cause and *different* fix.

---

## Status

**Empirically isolated.** The TP8+MTP variant of `test_mistral_large3` started failing in scheduled CI run **#608** (2026-05-12) and has been red in every nightly run since (#608, #609, #610, #611, #613). The introducing commit is **`d2c1034`** (PR #24436, "[Gemma 4] Adding MTP support", merged 2026-05-07). The fix is not in any merged PR yet — in particular, **PR #25407** ("Fix Mistral Large 3 nightly test") only fixes the NVFP4 variant and leaves this TP8+MTP regression intact (verified by running the test against the PR head — see "PR #25407 verification" below).

This is a separate regression from the NVFP4 one. The two are unrelated in code, in date, and in introducing commit.

---

## Failure Signature

- **Test:** `test/registered/8-gpu-models/test_mistral_large3.py::TestMistralLarge3::test_mistral_large3_all_variants` — **variant "TP8+MTP"** (model `mistralai/Mistral-Large-3-675B-Instruct-2512` with EAGLE draft `mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle`, `--speculative-algorithm=EAGLE`, TP=8, `--attention-backend=trtllm_mla --moe-runner-backend=flashinfer_trtllm`).
- **Workflow / job:** `Nightly Test (Nvidia)` (run `25835354140`) → `nightly-test-general-8-gpu-b200 (3)` (job `75909128362`), suite `nightly-8-gpu-common`, partition 3/4, runner `b200-novita-1` (8× B200, drv 580.126.09, CUDA 13). Same partition that fails on NVFP4.
- **Step that failed:** "Run common 8-GPU model tests" — exit code 1 / 255.

### Server command that triggers it

```bash
sglang serve \
  --model-path mistralai/Mistral-Large-3-675B-Instruct-2512 \
  --tp=8 --attention-backend=trtllm_mla --moe-runner-backend=flashinfer_trtllm \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --chat-template=mistral \
  --speculative-algorithm=EAGLE \
  --speculative-draft-model-path=mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle \
  --speculative-num-steps=3 --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4 \
  --kv-cache-dtype=auto
```

The server **never reaches model loading** — it crashes in `ServerArgs.__post_init__` during CLI-arg validation.

### Stack trace (verbatim, captured on PR #25407 head `e3fb4ee` + 8× B200 + flashinfer 0.6.11.post1 + sglang-kernel 0.4.2.post2+cu130 + torch 2.11.0+cu130 + `SGLANG_IS_IN_CI=true`)

```
Traceback (most recent call last):
  File "/sgl-workspace/sglang/.venv/bin/sglang", line 10, in <module>
    sys.exit(main())
  File "/sgl-workspace/sglang/python/sglang/cli/main.py", line 40, in main
    serve(args, extra_argv)
  File "/sgl-workspace/sglang/python/sglang/cli/serve.py", line 126, in serve
    server_args = prepare_server_args(dispatch_argv)
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 7716, in prepare_server_args
    return ServerArgs.from_cli_args(raw_args)
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 7101, in from_cli_args
    return cls(**{attr: getattr(args, attr) for attr in attrs})
  File "<string>", line 383, in __init__
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 967, in __post_init__
    self._handle_speculative_decoding()
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 3536, in _handle_speculative_decoding
    self.speculative_algorithm = _resolve_speculative_algorithm_alias(
  File "/sgl-workspace/sglang/python/sglang/srt/server_args.py", line 329, in _resolve_speculative_algorithm_alias
    cfg = AutoConfig.from_pretrained(
  File "/sgl-workspace/sglang/.venv/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 419, in from_pretrained
    raise ValueError(
ValueError: Unrecognized model in mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle.
Should have a `model_type` key in its config.json.
```

The same traceback fires twice per test run (once in the perf phase, once in the accuracy phase when the framework restarts the server), and the final summary names the failing variant:

```
Variant: TP8+MTP
…
Model 2 (mistralai/Mistral-Large-3-675B-Instruct-2512 [TP8+MTP]):
  performance, accuracy
  – Performance test exception … Server process exited with code 1.
  – Accuracy test exception   … Server process exited with code 1.
```

---

## Where the failure does and doesn't show up in CI

| Surface | TP8+MTP failure visible? |
|---|---|
| Per-partition metrics artifact (`metrics-8gpu-b200-partition-3.json`) | **No.** It records only successful benchmark rows; pre-load failures never produce a row, so the variant looks "missing" rather than "failing". |
| Raw step log of `Run common 8-GPU model tests` | **Yes.** Full traceback + a final `Model 2 (… [TP8+MTP])` summary line. |
| Run-page annotations | Partially — annotations only say `Process completed with exit code 1` / `255`; the variant name is *not* in the annotation. |

The failure has been openly visible in the raw nightly logs for the same 8 days that the NVFP4 issue has been red — the dashboard / metrics aggregation just doesn't surface it.

### Exact line locations in run 610's job log (`/repos/sgl-project/sglang/actions/jobs/75909128362/logs`)

Search the log for the timestamps below to land directly on the traceback (line numbers depend on whether you read the raw API output or the rendered web UI; the timestamps are unambiguous):

| Phase | Timestamp | What's there |
|---|---|---|
| Performance server crash | `2026-05-14T07:22:10.69…Z` | Full `ValueError: Unrecognized model in …Eagle` traceback |
| Accuracy server crash (auto-retry) | `2026-05-14T07:22:40.66…Z` | Same traceback, second occurrence |
| Final test summary | `2026-05-14T07:38:25.23…Z` | `Variant: TP8+MTP` and `Model 2 (…[TP8+MTP]): performance, accuracy - …` |

The GitHub annotation at `#annotation:5:51958` is the very last `Process completed with exit code 255` marker — that's the wrap-up, not the actual error.

---

## Boundary (from CI metrics history)

| Run | SHA | Date (UTC) | TP8+MTP rows present in `metrics-…-partition-3.json`? | Job status |
|---|---|---|---|---|
| 607 | `aa7a9af1` | 2026-05-11 01:00 | **Yes** (or not failed in a pre-load way) | partition green |
| 608 | `74d70af0` | 2026-05-12 00:54 | **No** | partition red |
| 609 | `4fb40bf` | 2026-05-13 00:58 | No | partition red |
| 610 | `34c0029` | 2026-05-14 01:00 | No | partition red (the run linked in the question) |
| 613 | `0fde6153` | 2026-05-15 00:58 (post-PR-#25310 revert) | No | partition red |

The pass-to-fail boundary aligns with the merge of `d2c1034` on 2026-05-07 14:08 PDT (next scheduled nightly was #608 on 2026-05-12, with `d2c1034` already in history).

---

## Empirical Bisect (single-step git diff)

All on 8× B200 (drv 580.126.09, CUDA 13), `SGLANG_IS_IN_CI=true`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. Test file reduced to the TP8+MTP variant only.

| SHA | `_resolve_speculative_algorithm_alias` defined? | Outcome | Wall time |
|---|---|---|---|
| **`d2c1034`** ([Gemma 4] Adding MTP support, PR #24436) | **yes** | **FAIL** — `ValueError: Unrecognized model in …Eagle. Should have a model_type key in its config.json.` (crash before any model load) | 60.7s |
| **`f1395af`** (parent, "fix(openai): map reasoning.enabled to thinking AND enable_thinking") | **no** (function does not exist in this commit) | **PASS** — gsm8k 0.949 ≥ baseline 0.85 | ≈ 7 min (model from disk cache; server boot ~30s due to warmed flashinfer autotune) |

One-commit window. The `_resolve_speculative_algorithm_alias` helper *and* its call from `_handle_speculative_decoding` are both added by `d2c1034`. Diff of the introducing commit on `server_args.py` (excerpt):

```diff
+def _resolve_speculative_algorithm_alias(
+    speculative_algorithm: Optional[str],
+    speculative_draft_model_path: Optional[str],
+    trust_remote_code: bool = False,
+) -> Optional[str]:
+    """Resolve CLI speculative algorithm; NEXTN/EAGLE may become FROZEN_KV_MTP for Gemma4 assistant drafts."""
+
+    is_gemma4_draft = False
+    if speculative_draft_model_path:
+        from transformers import AutoConfig
+
+        cfg = AutoConfig.from_pretrained(
+            speculative_draft_model_path, trust_remote_code=trust_remote_code
+        )
+        is_gemma4_draft = "Gemma4AssistantForCausalLM" in (
+            getattr(cfg, "architectures", None) or []
+        )
+    …
```

and:

```diff
+        self.speculative_algorithm = _resolve_speculative_algorithm_alias(
+            self.speculative_algorithm,
+            self.speculative_draft_model_path,
+            trust_remote_code=self.trust_remote_code,
+        )
```

inside `_handle_speculative_decoding`. After this change, every server with `--speculative-draft-model-path` set has its draft config eagerly loaded via HF's `AutoConfig.from_pretrained` — *purely* to detect a Gemma4 draft so it can silently promote `EAGLE`/`NEXTN` to `FROZEN_KV_MTP`. The check is unconditional: it fires even when the user has already specified `--speculative-algorithm=EAGLE` and has no interest in the Gemma4 alias resolution.

---

## Root Cause Classification

**Code regression — `d2c1034` (#24436) is the introducing commit.**

The helper added by that PR makes a sweeping assumption: every `speculative_draft_model_path` resolves to an HF-format checkpoint (with `config.json` containing a `model_type` key). That's true for HF-format drafts and for the Gemma4 drafts the PR was designed for, but it's **false for Mistral-native-format drafts**, which only ship `params.json` and `tekken.json` (no `config.json`). For those, `AutoConfig.from_pretrained` raises `ValueError: Unrecognized model in …. Should have a model_type key in its config.json.`, and that error propagates through `_handle_speculative_decoding` → `ServerArgs.__post_init__` → `prepare_server_args` → `serve` and kills the worker.

The Mistral-Large-3 EAGLE draft (`mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle`) is exactly this case: it ships a Mistral-native `params.json` and no `config.json`. So *any* user of the same combination — `--speculative-algorithm=EAGLE` + Mistral-native draft — can't start a server. This is a production-path break, not just a test fixture issue.

---

## PR #25407 Verification

PR #25407 ("Fix Mistral Large 3 nightly test", head `e3fb4ee`, open at time of writing) is the call-site fix for the NVFP4 variant. It changes `compressed_tensors_w4a4_nvfp4_moe.py:315` so `fp4_quantize` gets `layer.w13_input_scale_quant[:1]` (scalar slice) instead of the full per-expert tensor. It **does not touch `server_args.py`** and therefore does **not fix the TP8+MTP regression**.

Verified locally by checking out the PR head on 8× B200 with `flashinfer==0.6.11.post1`, `sglang-kernel==0.4.2.post2+cu130`, `torch==2.11.0+cu130` and running the **full 3-variant** test:

| Variant on PR #25407 (`e3fb4ee`) | Outcome |
|---|---|
| TP8 | ✓ PASS — gsm8k 0.953 |
| **TP8+MTP** | **✗ STILL FAIL** — same `Unrecognized model in …Eagle. Should have a model_type key in its config.json.` (`_resolve_speculative_algorithm_alias` at `server_args.py:329`) |
| NVFP4 | ✓ PASS — gsm8k 0.957 (PR #25407's `[:1]` slice fix works on flashinfer 0.6.11.post1) |

Total wall time 1574s (26 min). So PR #25407 lands the green light for the NVFP4 path, but the TP8+MTP path needs a separate fix.

---

## Recommended Fix

Smallest blast radius, one-liner-style: wrap the `AutoConfig.from_pretrained` call in a defensive `try/except` and treat the failure as "not a Gemma4 draft". This restores pre-`d2c1034` behavior for every non-HF-format draft and keeps the Gemma4 detection intact.

```python
# python/sglang/srt/server_args.py around line 326

    is_gemma4_draft = False
    if speculative_draft_model_path:
        from transformers import AutoConfig

        try:
            cfg = AutoConfig.from_pretrained(
                speculative_draft_model_path, trust_remote_code=trust_remote_code
            )
        except Exception:
            # Non-HF-format drafts (e.g. Mistral native: params.json only, no config.json)
            # cannot be parsed by AutoConfig. They are by definition not Gemma4 assistants.
            cfg = None
        if cfg is not None:
            is_gemma4_draft = "Gemma4AssistantForCausalLM" in (
                getattr(cfg, "architectures", None) or []
            )
```

A cleaner variant additionally short-circuits when the algorithm is already explicit and not in the alias-eligible set:

```python
    # Only NEXTN/EAGLE/EAGLE3 are subject to Gemma4 alias resolution.
    if speculative_algorithm not in (None, "NEXTN", "EAGLE", "EAGLE3"):
        return speculative_algorithm

    is_gemma4_draft = False
    if speculative_draft_model_path:
        from transformers import AutoConfig
        try:
            cfg = AutoConfig.from_pretrained(
                speculative_draft_model_path, trust_remote_code=trust_remote_code
            )
            is_gemma4_draft = "Gemma4AssistantForCausalLM" in (
                getattr(cfg, "architectures", None) or []
            )
        except Exception:
            pass
    …
```

Either is sufficient. The `try/except`-only version is the minimal fix.

A complementary, longer-term improvement: when the draft is Mistral-native-format, sglang could read `params.json` and surface "Mistral" as the architecture — useful for any future helper that wants to know more than "is this Gemma4 or not". Not required for unblocking this test.

---

## Reproduction Recipe

```bash
# Reduce test to TP8+MTP variant only (the only failing one for this bug)
# (the diff: drop the "TP8" and "NVFP4" ModelLaunchSettings entries from
# test/registered/8-gpu-models/test_mistral_large3.py)

# Empirical pass (parent of d2c1034)
git checkout f1395af
uv pip install "flashinfer_python==0.6.8.post1" "flashinfer_cubin==0.6.8.post1"
# sglang-kernel 0.4.2.post1+cu130, torch 2.11.0+cu130 already aligned by d2c1034-era pyproject
uv pip install -e python/ --no-deps
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → ✓ Performance, ✓ Accuracy 0.949 ≥ 0.85

# Empirical fail (introducing commit)
git checkout d2c1034
uv pip install -e python/ --no-deps
SGLANG_IS_IN_CI=true SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  python -m unittest test.registered.8-gpu-models.test_mistral_large3.TestMistralLarge3
# → ValueError: Unrecognized model in …Eagle. Should have a `model_type` key in its config.json.
# Test exits with AssertionError in 60.7s (pre-load crash).
```

---

## TL;DR

- The TP8+MTP variant of `test_mistral_large3` has been red in every B200 nightly since 2026-05-12 (#608+). It does not show up on the metrics dashboard because the failure happens before any benchmark row is written, but it is fully visible in the raw nightly **step log** with a stack trace and variant name.
- Introducing commit is **`d2c1034`** (PR #24436, "[Gemma 4] Adding MTP support") — empirically isolated by a one-commit git bisect (parent `f1395af` passes with gsm8k 0.949; `d2c1034` fails at server-arg parse time).
- The mechanism is that `d2c1034`'s `_resolve_speculative_algorithm_alias` unconditionally calls `AutoConfig.from_pretrained(speculative_draft_model_path)` to detect Gemma4 drafts. It crashes on any draft in Mistral native format (`params.json` only, no `config.json`).
- **PR #25407 does NOT fix this** — verified by running the full test on the PR head. It only fixes the NVFP4 variant. TP8+MTP still fails identically.
- Minimal fix: wrap the `AutoConfig.from_pretrained` call in a `try/except` and treat the failure as "not a Gemma4 draft" — one-line patch at `python/sglang/srt/server_args.py:328-330`.
