# Round 19 Summary — Loop 7

## Mainline objective (round-19-contract.md)
**task19 (AC-6) — record the conc-1/conc-16 perf guardrails for the landed DS paths
at the Loop-7 op-point and write the consolidated DS-vs-DSA recall/perf/non-regression
report.**

## Outcome: ACHIEVED — AC-6 MET; task19 done.

## Work Completed (`coding`, live measurement)
Booted DS-default, DS-hybrid (`scorer_norm=hybrid, head_agg=mean`), and DSA/native-NSA
under **CUDA graph** at the Loop-7 op-point (int8 / mem 0.7 / fp8-KV / TP=8 / page 64 /
radix-off) and measured per-request decode TPS at conc-1/16 via a **closed-batch probe**
(`perf_closed_batch.py` — concurrent `/generate`, short prompt, `ignore_eos`, OSL=256;
the server-log `gen throughput / #running-req` cross-checks the client number). This is
the trustworthy pure-decode method (NOT the GSP window mode, which can fabricate
empty-stream throughput per the loop-6 fail-closed lesson).

| variant | conc-1 TPS/req | conc-16 TPS/req | mem/GPU | graph |
|---|---|---|---|---|
| DSA (native-NSA) | 83.2 | 55.4 | 133 GB | replay ✓ |
| DS-default (top_k=2048) | 39.8 | 27.6 | 125 GB | replay ✓ |
| DS-hybrid (Tier-2.B) | 40.1 | 27.6 | 125 GB | replay ✓ |
| DS-lifted-4096 (opt-in, R17) | ~14.5 | — | ~114 GB | replay ✓ |

## Key findings (non-regression)
- **The Tier-2.B hybrid scorer is decode-free**: DS-hybrid == DS-default decode TPS
  (27.6 == 27.6 conc-16; 40.1 ~ 39.8 conc-1; same 125 GB). The landed long-context
  recall winner (AC-3: 16K 6%→38% material, MMLU −0.5pp) costs nothing on the hot path.
- **DS is structurally ≤ DSA (~0.5×)** — the known offline-channel-mask selector +
  logical→physical adapter cost (present since the Tier-1 spine), NOT a Loop-7 regression.
- **DS-default conc-16 27.6 == the Loop-6 closed-batch 27.1** → the Tier-1 admission/decode
  spine is intact and the directional AC-5 conc-16 TTFT (13.13 s < 22 s) still holds
  (the decode/admission path is unchanged; all Loop-7 work is opt-in/default-off).
- **DSA/fp16 defaults behavior-unchanged**; the opt-in lifted path's slower decode
  (~14.5 tok/s) is the recorded 4K-lever tradeoff (default-off, doesn't affect the
  default budget).

## Files Changed
- `development/loop7/m11_perf_consolidation.md` (the AC-6 report + the task20 source),
  `perf_closed_batch.py` (closed-batch probe), `perf_{ds_default,ds_hybrid,dsa}_c{1,16}.json`.
- Commit `68969deb0` (local — loop hook). **No production-code change.**

## Validation
- Closed-batch decode TPS per variant (above), cross-checked against the server-log
  `gen throughput / #running-req`; GPU mem from `nvidia-smi`; graph-replay from the
  `cuda graph: True` decode batches; admission served 16/16 all variants.
- Full DS unit suite → **350 passed + 9 subtests** (unchanged — no code touched).
- GPUs freed + all servers stopped at round end.

## AC status after R19
- **AC-6 → MET**; task19 done. With AC-1/3/4/5 (prior), **5/6 ACs MET**.
- **AC-2 PARTIAL** — only task20 (the final strategic-gate supersession decision record)
  remains. After task20, all 6 ACs are met and the loop can close.

## Remaining Items (active mainline)
- **task20 (AC-2, next mainline + loop close)** — the final gate-supersession decision
  record: cite M0 regime attribution, AC-1 closure, AC-3 hybrid scorer, AC-4 production-ready
  lifted, AC-5 servability, AC-6 perf guardrails (this `m11`); explicitly state what
  measured evidence superseded the Loop-6 Tier-2.A-primary ordering; cite/preserve the R8
  oracle-sink provenance.
- Evidence-hygiene queued (fold into task20): R8 stride/oracle provenance citation;
  plan-marker cleanup (pre-existing).

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: a perf-measurement consolidation round. The closed-batch decode-TPS method is
  already captured (`BL-20260531-ds-selection-fullwidth-overscan`); the findings
  (Tier-2.B scorer is decode-free; DS ~0.5× DSA; lifted is the 4K-lever tradeoff) are
  project evidence recorded in `m11_perf_consolidation.md`, not a reusable cross-round
  engineering pitfall.

## Goal Tracker
Updated directly (Plan Version 27): R19 row; task19 → Completed and Verified;
**AC-6 MET**; Active = task20 only. No Goal Tracker Update Request needed.
