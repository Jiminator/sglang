# Round 3 Summary — launcher signature_dtype + real-mask NIAH non-regression (AC-3.1)

## Mainline objective (round contract)
Run the real-mask NIAH non-regression on TP=8 hardware — boot DS with the compact int8 table on the Loop-5 mask and prove int8 does not regress DS needle recall vs the fp16 Loop-5 DS baseline. This was the last outstanding AC-3.1 evidence item (it gates AC-4). Blocking prerequisite: the launcher couldn't select the compact table.

## Correction to a prior-round premise
The RLCR box is **node 0 with 8× H200** (plus a reachable node 1) — V3.2 TP=8 serving is feasible here. My R1/R2 "2-GPU dev box" claim came from a capped `nvidia-smi -L | head -2`; the resulting "Explicitly Deferred" classification for the NIAH was wrong, and Codex was right to reject it. (Captured as a BitLesson — see below.)

## Blocking fix landed first (Codex R2 review)
`serve_double_sparsity.sh` built `DS_CONFIG` without `signature_dtype`, so the documented `bash serve_double_sparsity.sh` silently booted the **fp16** table (config default) — any compact-table hardware run would have validated full precision. Fixed (commit `5d8e47fb3`): added `SIGNATURE_DTYPE` env (default fp16), included `"signature_dtype": "${SIGNATURE_DTYPE}"` in `DS_CONFIG`, echoed it in the launch log, plus a behavioral test (stub `python3` captures `--double-sparsity-config`): default → fp16, `SIGNATURE_DTYPE=int8` → int8 and parses as a valid config.

## Real-mask NIAH non-regression — PASS (commit `8a05b1688`)
Setup (real TP=8 hardware):
- **DS-int8** on node 0:30000 — `SIGNATURE_DTYPE=int8`, mem 0.6, Loop-5 mask (`7b3207cae888`). Boot proof: `token_label_table: 0.87 GB/rank ... dtype=torch.int8 scales=float16` (the **0.5625× = 1.55→0.87 GB** reduction confirmed on hardware) + `double_sparsity_config='{...,"signature_dtype": "int8"}'`; decode coherent (" Paris.").
- **DSA (live reference)** on node 1:30001 (cross-node), mem 0.85.
- `test_double_sparsity_v32.py -k niah`, `AC12_NIAH_NUM_PROMPTS=20`, DS=node0 / DSA=node1 → **2 passed, 2 skipped, 5 subtests passed** (308 s).

| length | int8 DS (now) | fp16 Loop-5 DS | live DSA | int8 ≥ fp16? |
|---:|---:|---:|---:|:--:|
| 1024 (within budget) | 100% | 100% | 100% | ✅ |
| 1536 (within budget) | 100% | 100% | 100% | ✅ |
| 4K | 85% | 75% | 100% | ✅ (+10pp) |
| 16K | 5% | 5% | 100% | ✅ (=) |
| 64K | 0% (unservable) | 0% (unservable) | 100% | ✅ (=) |

**Verdict: PASS** — int8 matches or exceeds fp16 DS recall at every length (no regression; the +10pp at 4K is within the ±5pp/needle granularity at 20 prompts), no new unservable error (64K is an admission limit identical for both at mem-0.6), and the live DSA reference is 100% everywhere (paired cross-node setup sound). Combined with the decode-scoring microbench (TPS-neutral), the compact path is recall-neutral **and** TPS-neutral. fp16 baseline: `runs/20260528_dsv32_mvp/ac12_results/`.

## Files changed
- `development/serve_double_sparsity.sh` (+SIGNATURE_DTYPE), DS test file (+2 launcher tests) — commit `5d8e47fb3`.
- `runs/20260530_dsv32_loop6/real_mask_niah_nonregression.md` + `real_mask_niah_int8/` (5 `ac12_niah_*.json` + pytest log + int8 boot proof) — commit `8a05b1688`.
- `.humanize/bitlesson.md` (+1 lesson), goal-tracker, round-3 contract/summary (gitignored loop state).

## Validation
- Launcher tests: 2 passed (default fp16, int8 selectable through the real script).
- DS-int8 booted clean on TP=8 (after clearing a 4-day-old stale `sglang::router` that had grabbed port 30000); served coherently.
- Real-mask NIAH paired run: PASS, int8 ≥ fp16 at every length; artifacts copied under `runs/`.
- Servers killed; both nodes' GPUs freed.

## Remaining items
**AC-3 is now fully evidenced** (synthetic top-k overlap@2048≥0.99 + scale-sidecar proofs + decode-scoring microbench + real-mask NIAH + DSA-default). Nothing in AC-3 is deferred.
- **AC-4 (next mainline):** mem-fraction sweep `0.6→0.8` with `SIGNATURE_DTYPE=int8`, full NVML/torch-residual HBM accounting, `/get_server_info`, sustained long `/generate` with no OOM / no monotonic growth. Then AC-5 client-SLO (with admission-vs-prefill attribution), AC-6 hardware opt-in/DSA-default, AC-7/AC-8/AC-9, then gated AC-10.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-verify-hardware-before-deferring
Notes: Added a lesson capturing the R1→R3 mistake: a capped `nvidia-smi -L | head -2` was read as the GPU inventory, leading to a wrong "2-GPU, can't serve V3.2" deferral of the real-mask NIAH that review had to reject. The lesson: probe the FULL hardware inventory (`nvidia-smi --query-gpu=...`, CLUSTER.md, weights/rank×tp ≤ HBM/rank) before declaring a hardware step infeasible or deferring it; never let a display cap become a capacity claim. (Also surfaced but already covered by existing router-kill lessons: a 4-day-old stale `sglang::router` answered `/health` 200 with `no_available_workers`, masking my server's port-bind failure — diagnosed and cleared by killing the stale router and rebooting.)
