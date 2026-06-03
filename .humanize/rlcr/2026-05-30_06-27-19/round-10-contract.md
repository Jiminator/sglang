# Round 10 Contract

## Mainline Objective (exactly one)
**Land the DS-opt-in / DSA-default product property (AC-6) and the real-token within-budget gate (AC-9) on hardware, via one cross-node DS(node0)+DSA(node1) bring-up.** Codex's R9 review verified the AC-5 evidence/attribution as RESOLVED and directed: stop working AC-5 evidence; move to hardware AC-6 + AC-9 (stagnation warning against another evidence round). AC-5's directional verdict + open strict-SLO blocker stay tracked but are NOT this round's objective.

## Target ACs (2)
- **AC-6** (`coding`, hardware) — DSA-default boot meets the SLO unchanged and allocates **no** DS `TokenLabelTable`; the DS opt-in flag activates the compact int8 path. (DEC-2 "Both": DSA stays the production default; DS ships opt-in.)
- **AC-9** (`coding`, hardware) — `test/manual/test_double_sparsity_v32.py` asserts `within_budget` from the **actual** `usage.prompt_tokens` (fail-closed if usage missing/inconsistent), adds `input_tokens`, renames the word-count field `length_tokens`→`length_words`; **DS-fair gate definition UNCHANGED** (INDEX_TOPK=2048, 5pp recall tolerance, 1024/1536-word lengths); re-run the gate live + copy artifacts; diff shows the word-count proxy was safe (or correct it).

## Blocking Side Issues in Scope
1. **AC-9 harness edit (code, do first).** Thread `usage.prompt_tokens` (chat) / `meta_info.prompt_tokens` (generate) through `_generate`→`_generate_attempt`→`_run_niah`→`_niah_record`; compute `within_budget` from real `input_tokens`; fail closed if usage absent for a served prompt; rename `length_tokens`→`length_words`. No change to the gate thresholds/definition.
2. **Cross-node bring-up.** DS int8 @ 0.7 radix-on on node 0 (`serve_double_sparsity.sh SIGNATURE_DTYPE=int8`); DSA-default on node 1 (`serve_native_nsa.sh`, mem 0.85, no DS). Kill stale `sglang::router`/workers first (`pkill -f 'sglang::router'`).

## Queued / Out of Scope (explicitly NOT downgraded)
- **Strict-SLO failure stays the open mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Not this round's objective; remains tracked.
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, 120/600 s), **AC-8** (~70K servability probe), gated **AC-10** — later rounds. No FlashMLA decode-assert changes (AC-3.3). Do not change DS-fair thresholds (AC-9). The AC-6 SLO confirmation is a representative client run (full 3-trial sweep is AC-7, not here).

## Round Success Criteria
1. **AC-9 code:** harness edit landed; `within_budget` from `usage.prompt_tokens`, fail-closed; `input_tokens` + `length_words` in the artifact; gate definition unchanged. Verified by a local dry-run of the parsing logic (mock response) before hardware.
2. **AC-9 hardware:** within-budget NIAH gate re-run live (DS node0 + DSA node1), artifacts copied to `runs/20260530_dsv32_loop6/` showing per-length `input_tokens` (real) + `within_budget` from it + gate verdict; a diff/note vs the word-count proxy (was it safe?).
3. **AC-6 hardware:** tracked evidence that DSA-default boot has no DS `TokenLabelTable` (`/get_server_info` + boot-log: no table alloc, DS not enabled) and meets the SLO (a representative client run: sub-22s P99 TTFT, full admission, ≥30 TPS); and that DS opt-in toggles the compact int8 path (`token_label_table dtype=int8`, DS enabled). Artifacts under `runs/20260530_dsv32_loop6/`.
4. `git diff --check` clean; commit + push to `jimmy` (immediately after each commit); goal-tracker updated (task7/AC-6, task10/AC-9); `round-10-summary.md` with BitLesson Delta.

## Out-of-Scope Guards
- Do not weaken the strict SLO or mark the loop done; the SLO miss stays a mainline blocker.
- Do not change the DS-fair AC-12 gate thresholds/definition (AC-9). No new serve/bench scaffolding (reuse Loop-5 scripts). No FlashMLA decode-assert change.
- If cross-node hardware is unavailable/pre-empts, land the AC-9 code edit + the single-node AC-6 proofs achievable, and record the rest honestly (do not fabricate hardware results).
