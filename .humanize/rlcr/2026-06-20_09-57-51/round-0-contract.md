# Round 0 Contract

## Mainline Objective
Establish the pinned, reproducible measurement foundation for the entire diagnosis loop: boot DSA (native), DSA-radix-off control, and current production DS via the guarded `development/loop13/` harness on the dev clone, record full per-arm metadata, and **reproduce the known GSM8K regression** (DSA ≈0.970/0.953 vs DS ≈0.625/0.000) serial + batched. This is the sanity gate — if the regression does not reproduce, the build/harness is unsound and the loop halts before any diagnosis.

Because the production-DS server boot is also the source of the captured per-head scores / selected indices that the cheap localization controls consume, opportunistically capture that data and the recall-oracle figures during the same DS boot so the AC-2 controls (forced-all dense, TP head-agg micro-test, selected-index equivalence) can run without an extra boot.

## Target ACs
- **AC-1** (primary): pinned baseline reproduction with full metadata; regression reproduced serial + batched.
- **AC-2** (stretch, same DS boot): cheap localization controls + the explicit H3 fork (forced-all dense control, TP head-agg micro-test, selected-index equivalence, recall-oracle corroboration).

> Note on round semantics: the full plan (AC-3 reference selector, AC-4 evidence table, AC-5 gate, AC-6/AC-7 conditional branch, AC-8 writeup) is the eventual exit condition. Round 0 lays the load-bearing foundation (AC-1) and the decisive cheap fork (AC-2) first, per the plan's "cheapest decisive experiment first" sequencing; subsequent work continues toward plan completion within this loop.

## Blocking Side Issues In Scope
- None known yet. If `serve.sh`/`run_gsm8k.sh` lack a clean batched-vs-serial knob or per-arm metadata capture, the minimal harness plumbing to record AC-1's required fields (git SHA, mask sha256, server args, sample IDs/order, concurrency, serial/batched) is in scope — it is diagnostic harness, not a selection/adapter fix.

## Queued Side Issues Out Of Scope
- Any selection/adapter **code fix** (this loop lands no fix — diagnostic code only).
- Performance optimization of the reference selector (slow-but-correct is the point).
- Mask recalibration as an action (only measured as an AC-7 ablation if the BAD branch is taken).

## Round Success Criteria
1. DSA, DSA-radix-off, and production DS each booted from the dev clone (guard passed), one server at a time, torn down cleanly between arms (all 8 GPUs back to ~0 MiB).
2. GSM8K measured for each arm on the pinned configs (5-shot/200 dense, 24-shot/150 sparse, temp 0, completion API), serial + batched, with the regression reproduced within run-to-run noise.
3. A per-arm metadata record exists with all AC-1 fields populated.
4. DSA-radix-on vs DSA-radix-off shown bit-comparable at temp 0 (radix output-neutrality verified, not assumed).
5. (Stretch) Captured scores / selected indices / recall-oracle figures collected from the DS boot, enough to run the AC-2 cheap controls and record the H3 fork.
6. goal-tracker.md, round-0-contract.md, and round-0-summary.md written; changes committed.
