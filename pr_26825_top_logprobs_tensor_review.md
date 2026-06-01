# Code Review — PR #26825

**Title:** Fix TokenizerManager crash on `top_logprobs` with tensor values
**Author:** kflansburg (Kevin Flansburg)
**URL:** https://github.com/sgl-project/sglang/pull/26825
**Files:** `python/sglang/srt/managers/tokenizer_manager.py` (+1/-1), `test/registered/unit/managers/test_tokenizer_manager_top_logprobs_tensor.py` (+82)

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

The PR claims this fixes a `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`, which propagates out of the detokenization handler, gets caught by `print_exception_wrapper`, and SIGKILLs the prefill process (exit code -9) — a "prefill restart storm" in disaggregated PD deployments whenever clients send `top_logprobs > 0`.

---

## Verdict (sober version)

The diagnosis is **correct** and the one-liner is *directionally* right: `None` genuinely is the sentinel — the `else: ret.append(None)` two lines down proves the original intent was always "None means skip," so `is not None` reads that intent better than truthiness did. Good catch, good writeup, has a regression test. That's the part that's fine.

But the patch **treats the symptom, not the disease**, and the test **passes by lying**. Both verified below. Mergeable as defensive hardening — **not** as "the fix for the PD prefill crash."

---

## What Linus would say

> **"You've got a `torch.Tensor` showing up in a function whose signature says `List[float]`, and your fix is to change the `if` statement in the function that *receives* the garbage? No. You don't paper over a type violation at the leaf. You find who put a tensor there and stop them."**

### 1. The patch doesn't fix the bug — it relocates it

After the `is not None` guard lets the tensor through, `detokenize_logprob_tokens` does `zip(token_logprobs_val[i], token_logprobs_idx[i])`. Iterating a 1-D tensor yields **0-dim tensors**, not floats:

```
returned tuples: [(tensor(-0.1000), 10, None), (tensor(-0.2000), 20, None), (tensor(-0.3000), 30, None)]
type of first logprob: <class 'torch.Tensor'>
JSON serializable: NO -> Object of type Tensor is not JSON serializable
```

So instead of `RuntimeError: Boolean value of Tensor ... is ambiguous` at detokenization, you now get `TypeError: Object of type Tensor is not JSON serializable` when `meta_info` is serialized into the response. The crash moves from line 2147 to the serializer. The "restart storm" may still be a storm — just with a different stack trace.

*(Caveat: depends on whether the real production path coerces before serialize; the artificial repro in the test does not, and neither does the `decode_to_text=False` path traced here.)*

### 2. The test is green but certifies broken behavior

```python
ret[0] == [(-0.1, 10, None), (-0.2, 20, None), (-0.3, 30, None)]   # assertEqual PASSES
```

It passes (`assertEqual would pass? -> True`) because `tensor(-0.1) == -0.1` evaluates loosely to a truthy tensor inside tuple comparison. The author *thinks* they asserted "output is a list of float tuples." They actually asserted nothing about type. The function returns **0-dim tensors** and the test waves them through — it **enshrines the broken data type as the expected contract.** This is the part that should block the merge.

### 3. The conversion already exists at the source

`python/sglang/srt/disaggregation/prefill.py:517-527` already does the coercion that's missing:

```python
if logits_output.next_token_top_logprobs_val:
    logits_output.next_token_top_logprobs_val = [v.tolist() for v in ...]
    logits_output.next_token_top_logprobs_idx = [x.tolist() for x in ...]
if logits_output.next_token_token_ids_logprobs_val:
    logits_output.next_token_token_ids_logprobs_val = [v.tolist() for v in ...]
```

The PD prefill path *knows* these must be Python lists and converts `next_token_*`. The bug is almost certainly a **sibling path that skips this** (e.g. `input_top_logprobs`, or a branch not gated the same way). That's the line to fix — convert at the source where the contract is established, not 1500 lines downstream where four callers all hope someone upstream did the right thing.

### 4. Unexamined behavioral change for empty lists

`is not None` also changes behavior for empty lists:
- Old: `if []:` → False → appends `None`
- New: `[] is not None` → True → calls `detokenize_logprob_tokens([], [], ...)` → appends `[]`

Empty-position results flip from `None` to `[]`. Probably harmless, but it's an *unexamined* behavioral change riding along in a "one-line bugfix."

---

## What the patch should be

1. **Fix the source.** Find the path in `disaggregation/prefill.py` (or wherever) that leaves `*_top_logprobs_val` as tensors and `.tolist()` it there, next to the conversion that already exists. Keep the `is not None` change as a defensive sentinel fix — it's more correct than truthiness — but it cannot be the *whole* fix.
2. **Make the test honest.** Coerce to float in the fix and assert `isinstance(logprob, float)` (or assert `not isinstance(..., torch.Tensor)` and that `json.dumps(...)` succeeds). A test that a 0-dim-tensor output passes is not a regression test for this bug.
3. **Decide the empty-list semantics on purpose**, and add a case for it.

---

## Bottom line

Send it back: keep the `is not None` guard, add the `.tolist()` at the real source, and rewrite the test to assert types and JSON-serializability. As written, it hardens the detokenizer but does not fix the reported PD prefill crash — it relocates it past a test that can't see the relocation.

---

## Suggested reviewers

CODEOWNERS auto-requested `@merrymercy @Ying1123 @hnyls2002 @xiezhq-hermann` (the `/managers` owners). High-signal additions given the substance:

- **@fzyzcjy** — recently refactored the whole logprob assembly pipeline (`SchedulerLogprobResultProcessor` #25632/#25633, packed logprob container #25712); best person to say whether a tensor should ever reach this path.
- **@ByronHsu** — disaggregation CODEOWNER + prior logprob+streaming fixes (#17005); crash is a PD prefill restart storm.
- **@ShangmingCai** — the other active `/disaggregation` owner.
- **@hnyls2002** — already requested; also a disaggregation owner, so the most relevant of the four auto-assignees.
