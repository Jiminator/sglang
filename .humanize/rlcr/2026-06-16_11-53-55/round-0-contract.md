# Round 0 Contract

## Mainline Objective (ONE)
Re-establish the GLM-5.1-FP8 serving op-point on the fresh 8×H200 node (milestone M-A) — the gate for
every downstream measurement. Concretely: kick off the loop ledger, repoint the serve scripts to
GLM-5.1-FP8, run the pre-sweep methodology review, regenerate the channel mask with a committed
provenance record, land the DEC-1 validator content-hash authorization change, mint+authorize radix-on
without the dev override, and reconfirm capacity + DSA-native non-regression on this node.

## Target ACs (focused)
- **AC-0** (0.1 mask+provenance, 0.2 radix-on minted+authorized via content-hash fixture, 0.3 capacity
  bs cap ≥64 @ mem 0.8) — PRIMARY.
- **AC-7** (DSA-native un-regressed — mandatory because the DEC-1 validator change is a shared surface).

## Standing discipline this round (every round)
- **AC-8**: `queue.md` current; evidence pre-flight (artifact exists + tracked + claim matches
  POST-commit); `results.md` rewrite-over-append; frozen references never re-run; `git push` at the
  round boundary.
- **AC-UX.2 (partial)**: the serve-script MODEL_PATH/CHANNEL_MASK_PATH repoint lands here as a
  correctness precondition; the rest of AC-UX (runbook, comment/help-text sweep) is M-C (out of scope this round).

## Blocking side issues in scope
- None known yet. (If calibrate.py fails the FP8 dry-run placement gate, that becomes blocking and is
  resolved in-round.)

## Queued / out of scope this round
- task7 tax guard (AC-4), task8 locked sweep (AC-2/AC-3/AC-9), task9 headline report — milestone M-B.
- task10 full UX pass + runbook (AC-UX.1/.3, the rest of .2) — milestone M-C.
- task11 close-out — final round.
- The 128k-ISL second op-point — permanently OUT OF SCOPE.

## Round success criteria
1. `development/loop11b/queue.md` populated as the single source of truth; committed.
2. Serve scripts default to the GLM-5.1-FP8 snapshot + the chosen mask path; committed.
3. Pre-sweep methodology review captured (codex output integrated into queue/results, not raw).
4. `--dry-run-blocks 1` placement preflight PASSES on the FP8 checkpoint; full calibrate writes
   `glm51-fp8-channel-mask-*.safetensors`; `provenance.json` records both hashes + command + env;
   recall-comparability vs the frozen baseline checked (or a served-fp8 baseline recorded).
5. DEC-1 validator content-hash change lands; the AC-7 DSA regression runs clean in the same round.
6. Radix-on minted (override → DEC-12 edge probes → `write_radix_fixture_state`) and a no-override boot
   authorizes; `/server_info` shows the locked key set; negative-control + legacy schemas refused.
7. Capacity reproduces: derived decode-bs cap ≥ 64 @ mem 0.8, CUDA-graph capture OK, no TokenLabelTable.
8. DSA-native boots/captures un-regressed vs its own node reference.

A documented partial (e.g., mask + validator land but capacity slips to the next round) is acceptable
and reported honestly; nothing is claimed beyond committed evidence.
