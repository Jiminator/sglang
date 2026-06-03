# Round 21 Contract — Loop 7

## Mainline objective (EXACTLY ONE)
**task20 (AC-2) — write the final strategic-gate supersession decision record that closes
Loop 7**: a single coherent decision artifact citing the full evidence chain (M0 regime
attribution → AC-1 oracle closure + R8 stride provenance → AC-2 recall matrix/CI rule →
AC-3 hybrid-scorer non-regression → AC-4 production-ready lifted-budget disposition → AC-5
64K servability → the corrected AC-6 perf/TTFT guardrails), and explicitly stating what the
M0/R4–R8 measured evidence changed in the Loop-6 strategic gate's Tier-2.A-primary ordering
(Tier-2.B is the primary long-context path; Tier-2.A is a bounded 4K lever).

## Blocking prerequisite (must be cleared FIRST — it truly blocks the mainline)
**Repair the AC-6/task19 TTFT evidence provenance** (R20-review Mainline Gap #1), because
task20 must cite `m11_perf_consolidation.md` as a correct source artifact:
- Add a `run_provenance` object to every `ttft_*.json` (server-code commit used for the
  live run, measurement-tool commit, tree-dirty-during-run flag, GPU type/count, exact
  server launch command, effective DS/native-NSA config, mem fraction, graph/radix/overlap
  flags, admission/served count, graph-evidence log line, memory source/value, artifact
  path; mark it reconstructed-in-R21 with metric values unchanged from the R20 run).
- Correct `m11_perf_consolidation.md` so the commit story is EXACT: `f9f6ec056`=R18,
  `68969deb0`=R19, `30173f08b`=R20; the R20 TTFT servers were launched from the R19 tree
  `68969deb0` (DS/DSA production serving code unchanged — R19+R20 touched only
  `development/loop7/`) with the `--stream` probe uncommitted, committed as `30173f08b`.
- Reconstruct (do NOT rerun): the exact run state is reconstructable from the boot logs +
  launch commands + `nvidia-smi` + commit history.

## Target AC(s)
- **AC-2** (the final decision record — the loop-close artifact) + **AC-6** (the corrected
  perf/TTFT source artifact the decision cites). After these, all 6 ACs are MET → loop close.

## Queued — explicitly OUT of scope this round (NOT closed/deferred)
- Remove plan/workflow markers (`AC-*`, `task*`, `Tier-2`, `DEC-`) from production
  code/comments/tests — pre-existing; do before final cleanup/merge, not now.
- Learned/distilled selector (DEC-5) — out of scope unless explicitly owner-approved.

## Concrete success criteria
1. **Provenance repaired**: each of the 12 `ttft_*.json` carries a complete `run_provenance`
   object (the fields above); `m11`'s commit story is exact and internally consistent with
   the R20 summary's `30173f08b`. The probe is also extended so future `--stream` runs
   self-document provenance (durable fix), and the backfill reuses that schema.
2. **Final decision record written** (`development/loop7/m12_final_decision.md`): the
   gate-supersession decision as the loop-close artifact, citing in one chain M0 regime
   attribution, AC-1 oracle closure **and the R8 stride/oracle provenance explicitly**
   (committed `oracle_stride_reference.json` + the `selection_kernel.py` stride=1 call
   site), AC-2 recall matrix + CI/materiality rule, AC-3 hybrid scorer non-regression,
   AC-4 production-ready lifted disposition (`m9`), AC-5 64K servability, AC-6 perf+TTFT
   guardrails (`m11`); stating exactly what changed from the Loop-6 gate
   (`ds_on_v32_decision.md`) — Tier-2.A was sound primary before M0; M0/R4–R8 showed 16K
   budget-partial / 64K scorer-limited, making Tier-2.B the primary long-context path and
   Tier-2.A a bounded 4K lever; the prior rationale was sound when written.
3. **DEC-4 close-gate satisfied**: the decision record confirms the AC-4 Tier-2.A landing
   disposition (`m9`, production-ready) exists so the close does not leave a dangling
   pursued-hardening item; all 6 ACs cross-referenced to their evidence artifacts.
4. Full DS unit suite still passes (no production-code change this round); GPUs not needed
   (no rerun); working tree consistent.
5. `goal-tracker.md` updated (task19 verified with provenance; task20 done; all 6 ACs MET;
   loop ready to close); commit.

## Tag routing
- task19 provenance repair → **`coding`** (Claude: probe/backfill/m11 edits).
- task20 decision record → **`coding`** (Claude writes it; per the plan task table task20 is
  `coding`). Optionally sanity-check the drafted record via `/humanize:ask-codex` (as the
  `m9` disposition was), then integrate — not a routing change, a review step.
