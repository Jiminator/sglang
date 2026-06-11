# Loop 10 Draft — Close the remaining DS-on decode gap (the dead-width tax)

> Written 2026-06-11, after **Loop 9 closed** (all ACs met: Case-1 decode 632,239 → **480,989 µs**,
> 1.84× → **1.403×** vs the frozen DSA floor; ring reduce eliminated, deterministic radix top-k
> shipped, persistent-worker logical-score landed; every change recall-gated and cross-rank
> bit-identical). This loop continues the **same goal with the same scoring method**: drive the
> frozen Case-1 one-batch decode number further toward the frozen Case-2 DSA floor (342,857 µs).
> Feed this through `gen-plan` once scope is confirmed.

---

## What this work is (and is NOT) — read first

**This is not frontier LLM development. There is no novelty here.** We are reproducing the results
of a **2-year-old paper** (Double Sparsity, arXiv:2408.07092, `development/past_implementations/double_sparsity_paper_2408.07092.pdf`)
on top of a **fully open-source codebase** (SGLang), and checking whether it can be made to perform
on **MLA models that are well behind the frontier** (GLM-5.1). The contribution, if
any, is engineering: making an existing sparse-attention idea cheap enough to be worth its opt-in
slot on a model that already ships a *trained* sparse indexer (DSA). This is purely educational and the algorithm theoretically has no hope of beating the highly optimized DSA algorithmn.theres a reason why the algorithm was introduced 2 years ago and nobody has actually decided to use it in a frontier model, we just want to see how it would perform and learn about basic performance engineering skills when doing so.

## Objective

Close the remaining **138,132 µs gap** (480,989 vs the 342,857 µs floor, 1.403×) on the **frozen
Case-1 benchmark** — GLM-5.1-FP8, 8×H200 TP=8, mem 0.7, bs 29, ISL 4096 / OSL 512, decode
GPU-kernel µs per 10-step window at torch TP-0, one trial per run. Scoring method identical to
Loop 9.

**Scope (per user): any scale of refactoring — from complete rewrites to small code changes — is
welcome**, provided it:

1. **Keeps the core DS concept** (offline channel mask → per-token signatures → query·signature
   scoring → top-k selection → sparse MLA decode).
2. **Introduces no extra lossiness in quality.** Same bar as Loop 9: recall@2048 within ±0.5pp
   (fail-closed) per landed change, cross-rank selection bit-identity HARD; and any change that is
   *supposed* to be exact (layout/transport/shape refactors) must prove **bit-identical logical
   selection indices** vs the pre-change state, not just recall parity.

Not in scope: SLO re-validation, recall R&D, the bs-64 re-tuned op point (loop-9 follow-on 4 —
separate characterization loop), multi-node.

---

## The gap we are closing (current state, from `development/loop9/results.md`)

| Reference | µs / 10-step window | ratio | status |
|---|---|---|---|
| Loop-9 frozen baseline (20260609) | 632,239 | 1.84× | historical |
| **Loop-9 final landed = THIS LOOP'S BASELINE** (`loop9/runs/20260611_r1/`) | **480,989** | **1.403×** | starting point |
| Case-2 DSA floor (frozen, never re-run) | 342,857 | 1.0× | target |

Per-bucket residuals at R1 (torch TP-0), i.e. what is left to attack:

| Bucket | µs / window | note |
|---|---|---|
| `all_reduce_two_shot_kernel<bf16,8u>` (DS score reduce) | **93,480** (+ ~15–18k bf16 cast/copy-back) | **the dominant residual** |
| new radix top-k suite (5 Triton launches/layer/step) | ≈36,300 | loop-9 gate was ≤80k; measured floor ~17.7 µs/call |
| `_logical_score_kernel` (persistent-worker grid) | 36,908 | loop-9 gate was ≤40k |
| shared non-DS topk/sort | 20,524 | present identically in Case 2 — NOT DS-attributable |

DSA spends ~17.2k µs on its entire fused indexer. DS currently spends ~167k (+casts). The
structural cause of nearly all of it is **the dead-width tax**: under CUDA graphs every DS selector
tensor is shaped by the static `req_to_token` width — `[bs, 202752]` — while the served op point
has only **≤4,608 live tokens** per row. The score reduce moves ~11.76 MB bf16 per call when live
data is ~0.27 MB: **~98% of reduced bytes are dead.**

Fresh evidence (post-loop-9 nsys capture, `loop9/runs/20260611_nsys/nsys_vs_baseline.md`): the
bf16 two-shot reduce is now the **single largest DS kernel on the timeline** — 13.3% of all
GPU-kernel time, 106 µs/call average *including cross-rank arrival skew absorbed as in-kernel
waiting* (so shrinking bytes also shrinks the skew-absorption window). Loop-9's kernel wins are
independently confirmed (top-k stack −76%, logical-score −41%, control buckets bit-for-bit clean).

**Why we believe the gap can close further:** the loop-9 M5 wildcard analysis
(`loop9/reviews/task15_m5_wildcard_proposal.md`, Codex analyze artifact) ranked the candidates and
projected that width-bucketed selector graphs with compact score buffers land the total at
**~1.10–1.15× the floor** (projection made against the M2-era 512.7k baseline; re-derive against
480,989 at plan time — R1 already banked part of its logical-score line). The last ~0.10–0.15× is
DS's intrinsic TP=8 obligation (heads are sharded → a per-layer cross-rank score exchange DSA
never pays) — full parity is NOT the bar.

---

## Candidate ideas (a menu — `gen-plan` picks/sequences; each is one implement→measure cycle)

1. **PRIMARY — width-bucketed DS selector graphs + compact per-bucket score buffers**
   (M5 proposal rank 1, "B + C-lite"; its disposition was needs-user-decision — this loop IS that
   decision). Capture the DS selection stages over a small ladder of *live-width* buckets (e.g. an
   8k bucket covers the served ≤4,608-token window) so the expensive selector shapes become
   `[bs, 8192]` instead of `[bs, 202752]`. Expected from the proposal: reduce ~110k → **~31–35k**
   (0.46 MB at the 8k bucket goes one-shot custom-AR, ~40 µs/call floor), logical-score
   36.9k → **~10–15k**, top-k 36.3k → **~16–24k**. Known risks (from the proposal — design for
   them, don't discover them): (a) the runner is bs-bucketed today; width-bucketing is a **real
   cuda-graph-runner integration change**; (b) bucket dispatch must come from **host-visible
   scheduler metadata**, not a device-computed max (no sync before replay); (c) dead logical
   positions must stay equivalent to −inf, and selected outputs must preserve **logical** positions
   exactly — the compact→logical inverse mapping becomes part of the correctness contract; (d) more
   captured variants = more capture memory and replay bookkeeping (the loop-9 M4 audit freed
   16.8 GB of over-captured ladder, so headroom exists — budget it explicitly); (e) custom-AR is
   out-of-place — the bucketed reduce needs a captured copy-back or consumers reading the compact
   buffer directly; (f) overflow beyond the largest bucket takes a graph-safe full-width fallback
   that is bit-identical. *This is an exact layout/transport change: gate on bit-identical logical
   indices, not just recall. Scope: large refactor (runner + DS cuda_graph.py + selection
   pipeline).*

2. **Fused multi-block top-k redesign** (loop-9 follow-on 2). The shipped Triton radix suite is 5
   launches/layer/step; the loop-9 AOT op proved a true single launch at 43.2 µs op-point but lost
   at long contexts (one block per row). Redesign with several blocks per row + cross-block
   coordination targeting the measured **17.7 µs/call floor across ALL widths** — ideally expressed
   directly on the compact bucketed buffers from Idea 1 (smaller rows make single-launch easier).
   The AOT build pipeline, env-gated tests, and non-finite contract fixtures all exist from loop 9.
   If a rebuilt wheel is to be *installed*, that is gated: mandatory DSA regression (DS-off smoke +
   Case-2 re-validation) per loop-9 follow-on 3 — or stay Triton-only and skip the wheel question.
   *The deterministic tie-break contract (score desc, pos asc; −inf/NaN never selected, +inf
   maximal) must be preserved bit-exactly.*

3. **Reduce-path residuals after bucketing**: pick the cheapest correct transport per bucket
   (one-shot vs two-shot custom-AR by size), eliminate the separate fp32↔bf16 cast kernels by
   fusing the cast into producer/consumer (the ~15–18k cast tax), and re-measure skew absorption
   (nsys timeline) once bytes shrink. *Lossless transport changes; selection bit-identity gates
   each.*

4. **Logical-score on compact width**: the persistent-worker grid already loops live blocks only;
   on compact buffers the dead-grid floor disappears entirely (M5 projection ~10–15k). Likely
   falls out of Idea 1 nearly for free — measure, don't assume.

5. **WILDCARD (only if 1–4 plateau) — fused score+select with exact two-round reduce** (M5 rank 4,
   explicitly low-confidence): exact global top-k over TP-summed scores from bounded local
   candidate unions is NOT guaranteed; an exactness round can fix it but worst-case expands to
   full width. Touch only with an exactness proof; anything approximate fails the
   no-added-lossiness bar and the cross-rank bit-identity hard gate.

Ideas found while reading the code that beat these replace them — the menu is a starting point,
not a contract. Complete rewrites of the selection pipeline are acceptable per the scope
statement, under the same gates.

---

## Open scope + the task queue (`development/loop10/queue.md`)

**The scope of this loop is deliberately NOT fixed to the menu above.** The agent is expected —
and incentivized — to invent additional optimization ideas while working (reading code, reading
profiles, watching where the bottleneck moves after each landed change) and to pursue them **in
the same loop**, as long as each new idea is compatible with the optimizations already completed
and accepted.

`queue.md` (already created, deliberately empty) is the loop's **self-contained task queue and
checklist** — the single source of truth for what is planned, in flight, done, or dropped:

- **Populating the queue is the FIRST task of the loop**, once plan refinement has completed and
  the loop kicks off — seed it from the final plan's tasks plus any further ideas generated at
  kickoff. Do NOT populate it during plan generation.
- Every task gets a queue entry: id, description, targeted bucket, expected effect, lossiness
  posture, compatibility note vs already-landed changes, status. New ideas discovered mid-loop are
  **appended as queued entries** (with that one-line compatibility check) rather than expanding
  the current task's scope.
- A task is marked completed only after its gates pass (losslessness teeth + profile). Dropped or
  superseded tasks stay listed with the measured/reasoned cause — no silent deletions.
- The queue is committed with each round so reviews see the same checklist the agent works from.

## Subagent usage (context discipline)

Use subagents **liberally** to keep the main context from bloating across a long multi-task loop:
Explore subagents for code/call-chain reconnaissance, implementation subagents for well-scoped
task slices, analysis subagents for trace/CSV digestion. Two hard rules:

1. **Every subagent's work is carefully reviewed by the main agent before it is trusted** — diffs
   read in full, claims checked against the actual artifact, never relayed unverified.
2. Nothing a subagent produced lands without passing the **same verification gates**
   (losslessness teeth + profiling) as main-agent work. Subagents save context, not review.

---

## The iterate→measure protocol (unchanged heartbeat, loop-9 tooling reused)

For **each** queue task, exactly one cycle:

1. **Implement** (DS path; shared surfaces touched → see AC-4, stricter than loop 9).
2. **Verify losslessness FIRST** — three teeth, all existing loop-9 tools:
   a. **Selection-capture bit-identity** (`loop9/selection_capture_tool.py` run/verify/diff, graph
      mode, 8 ranks): logical selected indices identical to the pre-change served state for exact
      changes. NOTE for Idea 1: capture mirrors must record **logical** indices after the
      compact→logical mapping; the diff target is the loop-9 R1 digest
      (`loop9/runs/20260611_r1/selcap_digest.json`).
   b. **Cross-rank bit-identity** (hard, every gate run).
   c. **NIAH recall@2048** (`loop7/niah_oracle_sweep.py` + `loop9/oracle_recall_summary.py`,
      fail-closed ±0.5pp vs the frozen `loop9/runs/20260610_m0/recall_baseline.json`).
3. **Profile after EVERY task implementation** (not just at round gates): re-profile Case 1 with
   the frozen recipe verbatim via
   `development/profiling/runs/20260609/run_case.sh <out> case1 torch 29` — torch profiler is the
   default; add an nsys capture when the question is timeline-shaped (skew, overlap, launch gaps).
   The purpose is twofold: confirm the optimization did its job in its targeted bucket, and see
   **whether the bottleneck has shifted**. **One trial per run**; Cases 2/3 and all loop-9 run
   dirs are frozen references, never re-run. (nsys note: NVTX ranges do NOT fire under graph
   replay — attribute by kernel name; see `loop9/runs/20260611_nsys/nsys_vs_baseline.md`.)
4. **Read the gap** per-bucket vs the R1 column and the Case-2 floor (`summarize_torch.py`,
   `compare_decode.py`). Per-bucket attribution primary, totals secondary (boot-to-boot
   shared-kernel variance up to ~27k µs — established in loop 9, DEC-1). A shifted bottleneck is
   a queue-feeding event: append the newly exposed target to `queue.md` as a candidate task.
5. **Keep or revert.** Bank only what passes step 2 and shrinks its targeted bucket. Carry the
   running Case-1 number forward. Ledger: `development/loop10/results.md` (rewrite-over-append
   when state changes; one authoritative current-state section — loop-9 lesson).

Measurement discipline carried from loop 9 (BitLesson-backed): eager microbenches measure host JIT
dispatch — the **CUDA-graph captured replay number is the binding one**; run seq≈0 floor probes
before believing a hypothesis; build the loser before issuing a comparison verdict.

---

## Acceptance criteria (draft — `gen-plan` formalizes the numbers)

1. **Gap closed and attributed.** Case-1 decode GPU-kernel time materially below 480,989 µs and
   measurably closer to 342,857 µs, attributed per-bucket. Suggested bars for discussion
   (`gen-plan` sets final): minimum **≤420k** (~1.23×); strong **≤395k** (~1.15×, the M5
   projection band). Per-bucket suggestions: score-reduce bucket (named AR kernel + casts)
   **≤40k**; logical-score **≤15k**; DS top-k **≤25k**.
2. **No extra lossiness** (the user's hard bar): recall@2048 Δ ≤0.5pp fail-closed per landed
   change; cross-rank bit-identity HARD; exact-by-design changes (layout/transport/bucketing)
   additionally prove bit-identical logical selection vs the pre-change state.
3. **DS concept intact** — no dense fallback, no DSA-indexer substitution.
4. **DSA-native default un-regressed — STRICTER THIS LOOP.** Idea 1 touches the shared CUDA-graph
   runner, not just DS files. Any change to shared capture/replay code triggers the mandatory DSA
   regression (DS-off smoke + frozen Case-2 recipe re-validation) **in the same round**, plus
   DS-off byte-identity where feasible. The shipped DSA default must stay untouched in behavior
   and performance.
5. **Protocol/ledger discipline**: one trial per run; Case 1 only re-profiled; frozen references
   reused; deviations logged in the goal tracker's Plan Evolution Log; evidence pre-flight before
   each round handoff (artifact exists + is tracked + claim matches artifact — loop-9 methodology
   lesson; watch the repo's `*.log` gitignore, `git add -f` cited evidence). `queue.md` kept
   current every round: statuses accurate, new ideas appended with compatibility notes, drops
   recorded with their cause; queue population itself is the loop's first task.

## Files to read first

- **Current state + follow-on definitions:** `development/loop9/results.md` (esp. "Structural
  headroom" items 1–3), `development/loop9/reviews/task15_m5_wildcard_proposal.md` (the design
  being executed), `development/loop9/runs/20260611_nsys/nsys_vs_baseline.md` (fresh timeline
  evidence).
- **The selection pipeline as loop 9 left it:**
  `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py` (`reduce_token_scores`,
  `_logical_score_kernel`, `retrieve_topk_graph_safe`), `topk_kernel.py` (radix suite),
  `cuda_graph.py` (DSGraphState, scratch bundles, capture parity), `selector.py`, `config.py`.
- **The graph-runner integration surface for width-bucketing:**
  `python/sglang/srt/model_executor/cuda_graph_runner.py` (bs-bucket ladder, capture/replay
  dispatch), the `allocate_graph_state` call sites in
  `python/sglang/srt/layers/attention/dsa_backend.py`, and the bind site in
  `python/sglang/srt/models/deepseek_v2.py` (`_select_topk_indices`).
- **AOT op (Idea 2):** `sgl-kernel/csrc/elementwise/ds_topk.cu`,
  `sgl-kernel/python/sgl_kernel/top_k.py`, build recipe in
  BL-20260611-sgl-kernel-build-cuda13-cccl (`.humanize/bitlesson.md`) +
  `loop9/runs/20260611_r1/sgl_kernel_build.log`. **Never force-install a rebuilt wheel over the
  frozen-reference binary** without the AC-4 gate.
- **Gate runners to clone:** `loop9/run_r1_gates.sh` (selcap → recall → torch re-profile pattern),
  `loop9/selection_capture_tool.py`, `loop9/oracle_recall_summary.py`.
- **Doctrine:** `CLAUDE.md`; `.pensieve/` maxims.

## Hardware / op-point

Single node 8×H200, TP=8, FP8 e4m3, page 64, fp8 KV, custom-all-reduce ON. **Never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving.** One TP=8 server at a time. The frozen
Case-1 recipe (server args, DS_CONFIG, bench command) is `development/profiling/plan.md` +
`runs/20260609/_env.sh` — scoring method unchanged from loop 9.

## Pending decisions (resolve in `gen-plan` discussion)

- **Numeric bar for AC-1** — accept the suggested ≤420k min / ≤395k strong, or re-derive from the
  M5 projection against the 480,989 baseline? (The proposal's ranges predate R1's logical-score
  landing; the strong bar should not double-count that win.)
- **Bucket ladder** — single 8k live-width bucket + full-width fallback, or a small ladder
  (e.g. 8k/32k/full)? Dispatch from which host-visible metadata (max seq_len in batch at schedule
  time)? Capture-memory budget (width buckets × bs buckets)?
- **Idea-2 delivery form** — Triton multi-block redesign (no wheel question) vs AOT CUDA (faster
  floor, but wheel install gated on the AC-4 DSA regression)? Or defer Idea 2 entirely if Idea 1's
  compact rows already put top-k under the bar?
- **DS-off regression depth for AC-4** — is DS-off byte-identity provable (preferred), or is the
  frozen Case-2 recipe re-validation + smoke the practical gate when the shared runner changes?
- **Loop budget** — single-idea loop (Idea 1 only, done right) vs the full menu? Idea 1 alone is
  projected to clear the strong bar; Ideas 2–4 are insurance/cleanup.
