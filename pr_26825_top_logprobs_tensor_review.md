# Code Review — PR #26825

**Title:** Fix TokenizerManager crash on `top_logprobs` with tensor values
**Author:** kflansburg (Kevin Flansburg)
**URL:** https://github.com/sgl-project/sglang/pull/26825
**Files:** `python/sglang/srt/managers/tokenizer_manager.py` (+1/-1), `test/registered/unit/managers/test_tokenizer_manager_top_logprobs_tensor.py` (+82)

> **Updated 2026-06-02** after live A/B validation on a real PD deployment
> (2×8×H200, mooncake/mlx5_0, sglang-router PD mode, Llama-3.1-8B-Instruct).
> The first draft of this review was static-analysis only; testing **refuted its
> central objection and flipped the verdict**. The original claims are kept below,
> each marked confirmed/refuted.
>
> **Addendum (same day):** suggestion #2 (the `float()` coercion bonus) has now
> been verified empirically — configs F1/F2 and G below. `float(logprob)` alone is
> insufficient (the idx tensors leak too), and the coercion must cover **both
> branches** of `detokenize_logprob_tokens`; with that, non-streaming, streaming,
> `token_ids_logprob`, and `return_text_in_logprobs` all return output
> bit-identical to the clean baseline.

---

## The change

```diff
 for i in range(len(token_logprobs_val)):
-    if token_logprobs_val[i]:
+    if token_logprobs_val[i] is not None:
         ret.append(
             self.detokenize_logprob_tokens(
                 token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
```

---

## Verdict (post-testing)

**Approve with non-blocking suggestions.** Previous verdict ("send it back, fix the
source") is withdrawn: the source fix it demanded **already merged as #26299 on
May 26** — four days before this PR was filed. The conversion block the first
draft pointed at in `disaggregation/prefill.py` *is* that fix, not evidence of a
sibling-path bug. What this PR adds is the consumer-side hardening #26299 cannot
provide, and live testing shows it does exactly what defense-in-depth should do.

## Context the PR description omits

- **#26286** (May 25): PD + `/v1/chat/completions` with `logprobs=true, top_logprobs>0`
  → `sampler.py` keeps top-logprobs as tensors (`no_copy_to_cpu=True`, overlap
  schedule), `disaggregation/prefill.py` forgot `.tolist()` → multi-element tensor
  hits `if token_logprobs_val[i]:` → `RuntimeError` → `print_exception_wrapper` →
  `kill_process_tree` → prefill restart storm.
- **#26299** (merged May 26, `c47f0e7cd`): fixed the producer. **Latest main does
  not crash** — verified live.
- **This PR** (May 31): fixes the consumer predicate.

## Empirical validation

Trigger = issue #26286's exact request (`logprobs:true, top_logprobs:5`, temp 0).

| Config | Producer | Consumer | Outcome |
|---|---|---|---|
| A — `c47f0e7cd~1` (pre-#26299 main) | 🐛 | 🐛 | 💥 Exact crash: `RuntimeError: Boolean value of Tensor...` at `tokenizer_manager.py:2144` (the line this PR changes); prefill tree SIGKILLed (procs 2→0); client sees 500 `KVTransferError/AbortReq` |
| B — `c47f0e7cd` (#26299) | ✅ | 🐛 | 200, correct 5-candidate top_logprobs |
| C — latest main (`547b886b3`) + #26299 locally reverted | 🐛 | 🐛 | 💥 Same crash; debug instrumentation confirmed `tensor shape=(5,)` reaching the consumer |
| D — **this PR** + #26299 reverted, chat path | 🐛 | ✅ | **No crash**; tensor confirmed arriving; HTTP 200 with logprobs **bit-identical to B** (`-0.00001990775308513548`, `-11.250020027160645`, …) |
| D2 — same, native `/generate` path | 🐛 | ✅ | Per-request 500 (`TypeError: Type is not JSON serializable: Tensor` at `json_response.py:16`); **process survives**, no restart |
| E — this PR, clean | ✅ | ✅ | Chat/generate/streaming battery all green; unit test 3/3 on PR branch; same test vs main's code: 2/3 fail with the exact RuntimeError, plain-list case passes |
| F1 — D2 + `float(logprob)` coercion in `detokenize_logprob_tokens` | 🐛 | ✅+ | **Still 500** — the idx tensors leak alongside the values (#26299's `.tolist()` covers both, so reverting it leaks both); orjson chokes on the 0-dim int tensor in `token_id` |
| F2 — D2 + `(float(logprob), int(token_id), None)` | 🐛 | ✅+ | **200, bit-identical to the clean baseline** — every logprob, token id, and the generated text match exactly |
| G-a — F2 setup, streaming `/generate` | 🐛 | ✅+ | 200 SSE — the first (leaked) chunk bit-identical to baseline |
| G-c — F2 setup, `token_ids_logprob` | 🐛 | ✅+ | 200 — requested-id logprobs as floats, bit-identical to baseline |
| G-b — F2 setup, `return_text_in_logprobs: true` | 🐛 | ✅+ | **Still 500** — F2 only coerced the `decode_to_text=False` branch; the `True` branch ships raw tensors via `list(zip(...))` (process survives) |
| G-b2 — coercion in **both** branches | 🐛 | ✅+ | 200 — correct decoded text, bit-identical values (`batch_decode` tolerates tensor idx) |

*(Side observation, pre-existing and unrelated to these patches: in PD,
`output_token_ids_logprobs` returns 3 entries for 4 completion tokens — identical
on the clean baseline; possibly worth a separate issue.)*

## First-draft claims, reconciled

1. **"The patch relocates the crash to the serializer"** — *half right; the wrong
   half is the one that matters.* On the OpenAI chat path (the reported incident
   path) the serving layer coerces 0-dim tensors during response construction:
   config D returned **bit-identical correct floats**, not artifacts, not a 500.
   On native `/generate` the relocation is real (D2) — but it lands as a contained,
   loud, per-request 500 with the process alive and a stack trace pointing at the
   leaking producer. Severity drops from "fleet restart storm" to "one failed
   request." That is the correct failure mode for a guard rail.
2. **"Fix the source instead"** — *already done* (#26299, merged before this PR
   existed). Config C is the counter-argument to "the source fix is sufficient":
   revert any one producer conversion and latest main's consumer still kills the
   whole process. With ≥4 result-processing paths and several
   `no_copy_to_cpu=True` producers (spec-v2 verify, multi-item scoring, PP proxy),
   "every path remembers `.tolist()`" is not an invariant worth betting the fleet on.
3. **"The test passes by lying"** — *stands.* `assertEqual` over tuples accepts
   0-dim tensors because `tensor(-0.1) == -0.1` is truthy. The test pins down
   "doesn't raise" but not "returns floats." See suggestions.
4. **Empty-list semantics flip (`[]` now detokenized instead of mapped to `None`)** —
   *real but latent.* Traced the producers: per-*position* empty lists don't occur
   today (per-request empties vanish in `extend()`; the `[None]` first-position
   sentinel stays `None`, covered by the PR's own `test_none_position_yields_none`).
   Worth one sentence in the PR description; not a blocker.

## Non-blocking suggestions

1. **Strengthen the test:** assert `isinstance(logprob, float)` (or that
   `json.dumps(ret)` succeeds) so it also guards the D2 gap; add an empty-list
   case to make the semantics change intentional.
2. **Optional bonus (now fully verified):** coercing in **both branches** of
   `detokenize_logprob_tokens` — `(float(logprob), int(token_id), text_or_None)` —
   makes the native `/generate` path return *correct output* instead of a 500 when
   tensors leak. Empirically: `float(logprob)` alone is **insufficient** (F1 — the
   idx tensors leak too); the `decode_to_text=True` branch needs the same
   treatment (G-b — `return_text_in_logprobs` still 500s if only the `False`
   branch is coerced). With both branches coerced, non-streaming, streaming,
   `token_ids_logprob`, and `return_text_in_logprobs` all return output
   bit-identical to the clean baseline (F2, G-a/c/b2). Caveat: `float()` raises on
   multi-element tensors, so this is shape-dependent recovery for the known 1-D
   leak — the `is not None` check remains the only universally-safe layer. Fine as
   a follow-up.
3. **Update the PR description** to reference #26286/#26299 and reposition as
   hardening — saves every future reviewer the "can't reproduce on main" confusion.
4. **Architectural follow-up (out of scope):** the disease is N result-processing
   paths each owning tensor→list conversion. One choke point (at `copy_to_cpu`
   finalization or IPC serialization) would delete the bug class; worth an issue.

## Bottom line

Merge it. One line, correct sentinel semantics, regression-tested, and
live-verified to convert a client-triggerable prefill SIGKILL into either a fully
correct response (chat) or a contained per-request error (`/generate`).

---

## Suggested reviewers

CODEOWNERS auto-requested `@merrymercy @Ying1123 @hnyls2002 @xiezhq-hermann` (the `/managers` owners). High-signal additions given the substance:

- **@fzyzcjy** — recently refactored the whole logprob assembly pipeline (`SchedulerLogprobResultProcessor` #25632/#25633, packed logprob container #25712); best person to say whether a tensor should ever reach this path.
- **@ByronHsu** — disaggregation CODEOWNER + prior logprob+streaming fixes (#17005); crash is a PD prefill restart storm.
- **@ShangmingCai** — the other active `/disaggregation` owner.
- **@hnyls2002** — already requested; also a disaggregation owner, so the most relevant of the four auto-assignees.
