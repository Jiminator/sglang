# Round 14 Contract

## Situation
Codex's Round-13 review: **ADVANCED**; all Round-13 work accepted (comparator per-side mem-fraction
fix + regression, calibrate docstring, and the `top_k` selection-budget investigation are verified;
409 CPU tests pass). Codex will not emit `COMPLETE` solely because **AC-12 is an unmet hard gate**,
and its "continue toward AC-12" directive is **explicitly conditional**: *"if the loop continues
toward the original AC-12 instead of accepting/re-scoping it."* That directive is a large net-new
R&D effort:
- a new **DS-flex decode backend** that doesn't reuse V3.2's `flashmla_kv` kernel (so `top_k` can
  exceed the model's `index_topk=2048`), and
- **TokenLabelTable fp8/int8 quantization** to free KV budget for 64K admission, plus validator
  changes, tests, and multi-point hardware sweeps.

**The user already adjudicated the scope of this in Round 13**, directing: do the queued cleanups,
test whether a bigger budget passes NIAH ("if not, serious issue"), and **"document all issues which
we can knock out in the next rlcr loop."** I executed all of it. The investigation was conclusive:
the budget is **kernel-locked** (can't be raised on this backend), DS decode is **sound** (dense
recall 100%), and the AC-12 gap is **selection quality** vs V3.2's trained DSA indexer — i.e., the
exact R&D the user deferred to a "next rlcr loop."

## Mainline Objective (exactly one)
**Surface the now-narrowed final AC-12 disposition decision to the user and act on their answer** —
because the user's Round-13 directive is fully complete, the investigation closed the "raise the
budget" hypothesis, and every remaining way to end this loop requires the user's authority (cancel
the loop, authorize an immutable-AC re-scope, or authorize the deferred R&D now). I am forbidden to
cancel the loop, change the immutable AC, or fake AC-12 green, and I will not silently launch days
of R&D the user explicitly deferred.

## Target AC
- AC-12 (the sole unmet AC). No threshold/immutable-AC change is made unilaterally; the disposition
  is the user's decision.

## Blocking Side Issues
- **None.** Codex found 0 blocking issues and no Round-13 implementation bug. There is no small
  mainline code change available: the only AC-12 path is the heavy, user-deferred R&D.

## Queued (explicitly OUT of scope)
- DS-flex decode backend + TokenLabelTable fp8 quant + sweeps (Codex's directive) — the user-deferred
  R&D; begin only if the user authorizes it this loop (option 3 of the escalation).
- AC-11 TTFT/effective-concurrency follow-up; pre-existing serve-header plan terms (cosmetic);
  copying full boot logs alongside the excerpt if used outside RLCR. All in `next_loop_issues.md`.

## Success Criteria
1. Re-confirm the state is unchanged and clean (git clean + pushed, 409 CPU tests still green, no
   active plan tasks, AC-12 the sole unmet AC).
2. The final disposition decision is surfaced to the user with concrete, now-narrowed options
   (close as smoke milestone / re-scope AC-12 to a DS-fair gate / authorize the deferred R&D now),
   including that ending the loop needs their cancel and that I cannot change the immutable AC.
3. Act on the user's answer (e.g., implement a re-scope if approved; or begin the R&D if authorized;
   or hold for their cancel).
4. No fabricated work, no fake AC-12 pass, no immutable-AC/threshold change, no loop-state edit/cancel.

## Out-of-Scope Confirmation
This round writes no production/test code unless the user authorizes a specific path. It does not
modify any AC or threshold and does not attempt to game AC-12 green.
