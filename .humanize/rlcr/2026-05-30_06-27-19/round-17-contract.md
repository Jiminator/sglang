# Round 17 Contract

## Mainline Objective (exactly one)
**AC-5 strict-SLO remediation — decode-throughput first.** At the lifted DS int8 / mem-0.7 / radix-on
operating point, attack the open strict-SLO blocker (per-req TPS < 30 at every conc; conc-32/64 P99 TTFT
> 22 s). Per Codex's R16-review Required Plan, this round is a **decode-throughput-first** problem:
(1) PROFILE the DS conc-16 decode hot path with a served-workload profiling artifact that breaks the
per-decode-step time into **DS selection/top-k**, **FlashMLA KV decode**, **token-label write/update**,
and **scheduler/interleave** overhead; (2) determine whether the batch-16 DS decode per-request TPS is
genuinely < 30 (isolating real decode cost from the WARMUP=0 cold-flood prefill-interleave artifact the
AC-5 report flagged); (3) make the **smallest** decode-path code change the profile justifies to move
conc-16 per-req TPS toward ≥ 30, **preserving the Tier-1 ABI lock** (`indices.shape[-1] == dsa_index_topk
== 2048`; no adjustable top_k — that is AC-10); (4) re-measure conc-16 and, if it passes, tune conc-32/64.

## Target AC(s)
- **AC-5 (task6)** — the done-criterion strict SLO. `coding` / hardware-run / owner claude.
- (AC-9 NIAH within-budget gate is a *guard* to re-run if a decode-selection change could affect recall.)

## Truly Blocking This Objective
- **DS decode per-request throughput at the lifted point.** Codex: "capping or delaying prefill cannot
  make a batch-16 decode path exceed 30 TPS/req" — so a pure queue/admission tweak is NOT sufficient; the
  decode token-time must be understood and reduced first. This is the mainline, not a side issue.
- A clean profiling artifact is a prerequisite: without it, any code change is unjustified (theory-over-
  pragmatism + durable-evidence: measure before optimizing, base the change on the measured bottleneck).

## Queued / Explicitly Out of Scope This Round
- **AC-10** (Tier-2 adjustable-`top_k` kernel / learned selector) — gated until AC-5 strict is verified.
  The ABI lock (`indices.shape[-1] == dsa_index_topk`) must NOT be touched this round.
- **Cross-node wrapper smoke** — future-gated; this round is single-node localhost.
- **DSA-default conc-64 TPS ~29.4** — queued pre-existing DSA limit (R12 user decision).
- More footprint reduction — explicitly ruled out by the AC-5 attribution (KV pool fits: 64×4608≈295K <
  396K). The lever is decode/scheduling, not footprint.
- More AC-7/AC-8 evidence churn — both verified; do not revisit.

## Concrete Success Criteria (this round)
1. A **served-workload decode profile at conc-16** (lifted DS int8/mem-0.7/radix-on) committed as durable
   tracked evidence: per-decode-step wall-time broken into DS selection (score+top-k+gather), FlashMLA KV
   decode, token-label write/update, and scheduler/overlap/interleave — with the method stated and the
   numbers recomputable/reproducible. This localizes the bottleneck (required before any code change).
2. A **clean steady-state conc-16 per-req TPS** measurement (proper warmup so arrivals stagger, or a
   controlled closed-batch-16 decode) to separate the real batch-16 decode cost from the WARMUP=0
   cold-flood artifact — so the "17.6 < 30" claim is re-grounded honestly.
3. If the profile shows a reducible DS-specific decode cost, the **smallest flag-gated/reversible code
   change** that the profile justifies (default-off until validated), preserving the ABI lock and DSA
   non-regression; re-measure conc-16 per-req TPS at the lifted point. If conc-16 reaches ≥ 30, proceed to
   conc-32/64 TTFT tuning (recording any locked-flag change as plan evolution with before/after sidecars).
4. If conc-16 per-req TPS genuinely cannot reach ≥ 30 with a small decode change this round, the round
   still delivers the profile + the attempted change + an honest characterization of the floor (what the
   DS decode-step time is vs DSA's, and which component dominates) — legitimate measured progress on the
   hard blocker, not a stall. Strict-SLO claim stays a live mainline blocker (DEC-3).
5. GPUs freed at round end; commit + push to `jimmy`; goal-tracker mutable section + round-17-summary +
   BitLesson Delta updated. If a code change lands, re-run the AC-9 within-budget NIAH guard if recall
   could be affected.

## Applicable BitLessons (confirm per-task via bitlesson-selector)
- `BL-20260530-admission-restore-tps-tradeoff` — per-req TPS ≈ 1/decode_step_time; measure at the actual
  operating point; expect the batch↔TPS curve to dominate.
- `BL-20260530-cold-flood-not-steady-state-slo` — the AC-5 WARMUP=0/320 run is a cold flood; a clean
  steady-state measurement is required before trusting "17.6 < 30" as the pure decode cost.
- `BL-20260528-dsv32-ds-serving-boot-chain` + `BL-20260529-ds-radix-flip-config-bound-artifact` — boot
  DS int8/mem-0.7/radix-on (the verified lifted point).
- `BL-20260527-ds-metadata-via-forward-context` + `BL-20260528-dsv32-ds-decode-degeneration` +
  `BL-20260528-ds-radix-capture-cuda-graph-safe` — the decode selection reads metadata via ForwardContext;
  any decode-path change must stay CUDA-graph-safe (no host sync / dynamic alloc under capture).
- `BL-20260530-remote-server-launch` — background boot; `ps | grep "[s]glang.launch_server"`; `pkill || true`;
  no foreground `sleep`.
- `BL-20260530-durable-tracked-acceptance-evidence` — tracked `.txt`/`.json`/`.md`; exact arrays + fail-closed
  verifier for any TPS/TTFT claim; `git diff --check`.
- `BL-20260529-ds-greedy-decode-degeneration-vs-dsa` / `BL-20260529-ds-longcontext-needle-recall-vs-topk` —
  if a decode-selection change is made, guard recall (AC-9 within-budget NIAH) before claiming it safe.

## Out-of-bounds reminders
No FlashMLA decode-assert / `top_k` changes (AC-3.3 ABI lock — that is AC-10). Compact path stays flag-gated,
fp16 default. No new serve/bench scaffolding — reuse Loop-5 scripts. No plan-process tokens in code/comments.
Do not change the DS-fair AC-12 gate. Must not exit by lying / editing loop state / cancel.
