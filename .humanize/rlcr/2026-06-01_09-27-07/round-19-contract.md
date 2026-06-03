# Round 19 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task19 (AC-6) — record the conc-1/conc-16 perf guardrails for the landed DS paths
at the Loop-7 op-point and write the consolidated DS-vs-DSA recall/perf/non-regression
report.**

Measure, under CUDA graph at the Loop-7 op-point (DS int8, `mem_fraction_static=0.7`,
fp8 KV, TP=8, page 64, radix-off), the per-request decode TPS, GPU memory, and
graph-replay status at conc-1 and conc-16 for **DS-default (top_k=2048)**,
**DS-hybrid (the AC-3 Tier-2.B graph-safe scorer)**, and **DSA / native-NSA** (the
reference), separating served vs admission. Confirm the landed DS paths do **not
regress** the Loop-6 Tier-1 operating point and that DSA defaults are behavior-unchanged;
cite the R17 lifted graph-mode perf (~14.5 tok/s conc-1, ~114 GB mem). Produce the
consolidated report — the source artifact for task20.

## Target AC(s)
- **AC-6** (perf guardrails + Tier-1 non-regression) + **AC-2** (the consolidated
  DS-vs-DSA recall/perf report that feeds the task20 decision).

## Blocking issues (truly block the mainline)
- **None.** Measurement-only; no production-code change to the default path.

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- **task20 (AC-2)** — the final strategic-gate supersession decision record. Depends on
  this report; next mainline.
- Evidence-hygiene queued: cite/preserve the R8 oracle-sink provenance (will be folded
  into task20); plan-marker cleanup (pre-existing).
- Learned/distilled selector (DEC-5) — out of scope.

## Concrete success criteria
1. **Closed-batch decode-TPS** (the trustworthy pure-decode method per the loop's bench
   lessons — NOT the GSP window mode that can fabricate empty-stream throughput): a small
   probe fires conc-1 and conc-16 concurrent `/generate` (short prompt, `ignore_eos`,
   fixed OSL) and records per-request decode TPS for DS-default, DS-hybrid, and DSA at
   the Loop-7 op-point, under CUDA graph.
2. **GPU memory + graph-replay + admission** captured per variant (from `nvidia-smi` +
   the server log `cuda graph: True` decode batches + served/admission counts).
3. **Non-regression conclusion**: DS-hybrid decode TPS ≈ DS-default (the graph-safe
   scorer adds no material decode cost), both within the Loop-6 budget; DSA is the
   reference; the default `flashmla_kv` path + fp16/DSA defaults are behavior-unchanged.
4. **Consolidated report** (`development/loop7/m11_perf_consolidation.md`): the conc-1/16
   perf table + the recall summary (DS-default vs DS-hybrid vs DSA, citing the R5/R7
   recall matrices + the R14/R17 lifted recall) + the Tier-1 non-regression statement,
   with exact server args, DS configs, commit SHA, GPU type, and artifact paths — framed
   as the source artifact for task20.
5. Full DS unit suite still passes (no code regression); GPUs freed + servers stopped at
   round end.
6. `goal-tracker.md` updated (task19 done; AC-6 MET); commit.

## Tag routing
- task19 is a **`coding`** task → Claude executes directly (live measurement + report).
