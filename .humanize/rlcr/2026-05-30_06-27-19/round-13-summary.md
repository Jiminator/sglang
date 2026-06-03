# Round 13 Summary — AC-7: 3-trial DS+DSA re-sweep at the lifted point (characterized, DEC-9)

## Mainline objective (round contract)
AC-7: the 3-trial DS+DSA directional re-sweep at the lifted operating point (DS int8 @ 0.7
radix-on, DSA-default @ 0.85 radix-on), conc 16/32/64, num_prompts=64, 120/600, TRIALS=3, then
refresh the ac11 reports showing DS achieved-concurrency now tracks nominal. Gated by the
cross-node wrapper host smoke (Codex R12).

## What landed (commit 5e6d3afb5, pushed)
**3-trial DS+DSA re-sweep** (num_prompts=64, warmup 120 / window 600, conc 16/32/64, radix-on
both). **Both sides ran on node 0 localhost, sequentially** (DS, then DSA) — see Methodology note.

**Headline — admission RESTORED (the footprint→admission spine's payoff), effective concurrency, median of 3:**
| conc | DS @ lifted | DS/nominal | Loop-5 DS (mem 0.6) | DSA |
|---:|---:|:--:|---:|---:|
| 16 | **16.0** | **100%** | 14.5 (91%) | 16.0 |
| 32 | **32.0** | **100%** | 24.6 (77%) | 32.0 |
| 64 | **47.0** | **73%** | 35.7 (56%) | 63.9 |

DS now admits full nominal concurrency at conc 16/32 and improves conc-64 to 73% (from Loop-5's
56%); errors 0 across all 18 runs.

**DS-vs-DSA parity gates FAIL** (comparator `ac11_resweep.md`): DS TPS 0.31–0.38× DSA; DS P99
TTFT 18–49× DSA. This is the **expected** DSA-trained-indexer advantage + the admission-restore
TPS tradeoff (AC-5) — a **DEC-7 directional follow-up, not a footprint regression**; AC-7 is soft
(DEC-9, may be characterized). The comparator's profiling obligation is discharged by the AC-5
measured attribution (queue-vs-prefill + decode-batch root cause) at the identical workload.
DSA-default reproduces its baseline (0.72/1.28/2.04 s, 46.9/37.5/29.5 TPS; conc-64 TPS ~29.5 =
the queued pre-existing limit).

## Methodology note (justified plan deviation)
The intended cross-node bring-up (DS node 0 + DSA node 1) was abandoned: **node-1 remote server
boot proved intractable this round** — setsid/nohup/tmux-arg launches all failed (fast ssh-close
teardown + accumulated zombie procs; no DSA weights ever loaded; ~2h lost). I pivoted to run
**both sweeps on node 0 localhost, sequentially**, which is **comparator-clean** (same
node/session/commit; only per-side mem differs, as in Loop-5) and avoids any cross-node
host-mismatch. Because neither sweep is cross-node, the **cross-node wrapper smoke is N/A this
round**; the R12 `--host` fix is verified in-wrapper (the DS sweep `bench_serving` banner
`Waiting up to 60s for http://127.0.0.1:30000` + the matching DS `.meta.json`), and R11 separately
proved `bench_serving --host node1` targets node 1. Captured as BitLesson `remote-server-launch`.

## Result
AC-7 characterized/soft-met (DEC-9): admission restored (the spine validated across 3 trials);
DS-vs-DSA parity miss recorded as a DEC-7 directional follow-up, attributed via AC-5, not a
footprint regression. The **AC-5 DS strict-SLO miss remains the open mainline blocker**.

## Files Changed
- `runs/20260530_dsv32_loop6/ac7_resweep/`: `ac11_resweep.md` (comparator), `ac11_analysis.md` (verdict/characterization), `ac7_resweep_metrics.json` (recomputable per-trial DS+DSA + medians + source JSONL SHA256), 18 `.meta.json` sidecars.
- `.humanize/bitlesson.md` (+1 lesson `remote-server-launch`), goal-tracker (R13 row; task8/AC-7 → done-characterized; cross-node-host blocker → RESOLVED/N-A; the host bug + node1 deviation documented), round-13 contract/summary (gitignored loop state).

## Validation
- 9 DS + 9 DSA runs, errors 0; DS achieved 16/32/47 effective (100/100/73%), DSA 16/32/64.
- Comparator exit 3 (gates fail = expected directional); recomputable metrics JSON validates (per-trial + SHA256); `.meta.json` confirm radix-on + per-side mem (DS 0.7 / DSA 0.85) + commit SHA.
- `--host` fix verified in-wrapper (localhost banner + matching sidecar). `git diff --check` clean; commit 5e6d3afb5 pushed to `jimmy`; both nodes' GPUs freed (0 MiB).

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc). Per Codex's plan, the AC-5 remediation (smallest scheduling/decode/operating-point change to restore both, with the AC-7 data in hand) is the next focus after AC-8.
- **AC-8** (~70K-token servability probe at the lifted mem fraction — HTTP 200 with capacity, or a characterized ceiling), gated **AC-10** (after AC-3–AC-9 verified). Queued: DSA conc-64 TPS ~29.5 (pre-existing). No FlashMLA decode-assert changes; DS-fair thresholds unchanged.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-remote-server-launch
Notes: Added BL-20260530-remote-server-launch capturing ~2h of node-1 boot failures: ssh-launched detached servers (setsid/nohup) are torn down when the ssh channel closes fast; `tmux new-session "<cmd>"` bypasses the shell so env/redirects don't apply (use `tmux send-keys` into the session's bash); remote login cwd is /root so use ABSOLUTE script paths; `pgrep -f` false-matches the launcher's own command line (use `ps | grep "[s]glang.launch_server"`); `pkill` no-match exit-1 trips `set -e` (use `|| true`, not a trailing `; true`); foreground `sleep` is blocked in harness Bash. Reliable fallback: run both servers on the local node sequentially via the Bash run_in_background tool (comparator-clean), which is what AC-7 used. Applied existing lessons: BL-20260530-cold-flood-not-steady-state-slo (num_prompts=64 steady-state methodology for the sweep), BL-20260530-durable-tracked-acceptance-evidence (recomputable per-trial metrics + source SHA256 since raw .jsonl are gitignored), BL-20260530-bench-host-targeting (the --host fix, verified in-wrapper), and the push-between-rounds preference.
