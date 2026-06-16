# loop11b results — authoritative current-state ledger (rewrite-over-append)

Finish loop 11's M4 verdict on a fresh 8×H200. One TP=8 server at a time; never set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving. Honest-verdict posture: a FAIL on the
throughput SLO is a complete, reportable result.

## Milestone status

| milestone | status |
|-----------|--------|
| **M-A op-point re-establishment** (AC-0, AC-5, AC-6, AC-7) | ✅ **COMPLETE** |
| M-B M4 close (AC-4 tax guard, AC-2/AC-3 locked sweep, AC-9) | ⏳ next |
| M-C productionize (AC-UX) | pending |
| close-out (AC-8) | pending |

---

## M-A — op-point re-established (fresh 8×H200, GLM-5.1-FP8 TP=8, fp8_e4m3, page 64)

### Channel mask (AC-0.1) — ✅
- **Recipe error caught + fixed.** The plan's AC-0.1 recipe carried the DeepSeek-V3.2 values
  (`--dtype bfloat16 --label-dim 16`); that mask (content `a4be98c4`) served **−5.2pp** recall at L4096
  vs the frozen baseline. The GLM-native recipe is loop8 `task5_calibration_recipe.md` + DEC-3:
  **`--dtype fp8_e4m3 --label-dim 32`** (label_dim 32 ∝ qk_nope_head_dim=192, 2× channels).
- **Mask (corrected):** `content_sha256=35155ac46ad7…`, fullfile `b223811f…`, tensors
  `channel_selection int32 [78,64,32]` + `channel_weights float32 [78,64,32]`, head_dim 192, page 64,
  seed 42, 256 Pile-val docs (corpus sha `46d72075…`, same as loop8). FP8 `--dry-run-blocks 1` placement
  gate PASSED (float8_present, sharded cuda:0–7, hooks on all 78 layers). provenance.json committed.
- **Serves** (smoke: "Paris. The city is located on the River Seine…").

### Recall (AC-5) — ✅
- **vs frozen baseline** (`loop9/runs/20260610_m0/recall_baseline.json`, matched 6240 samples/length, num=20):
  served radix-OFF L1024 100.0%, **L4096 58.045% (baseline 58.045 — EXACT)**, L16384 36.36% (base 36.04,
  +0.32pp), overall 64.80% (base 64.70) — all within ±0.5pp.
- **radix on-vs-off equivalence** (num=60, 18720 samples/length): L1024 0.0, **L4096 −0.283**, L16384 +0.123,
  overall −0.053pp — all within ±0.5pp → `recall_equivalence_passed`. (num=20 was noisy by a hair, as expected.)

### Radix-on authorization (AC-0.2, DEC-1) — ✅
- DEC-1 validator change landed (`5ac86f5cf`): `radix_fixture_config_fingerprint` pins
  `channel_mask_content_sha256` (tensor-content) instead of path + full-file SHA; `_sha256_file` removed;
  +2 portability unit tests; 13/13 fixture tests green.
- **DEC-12 mint gates (ld32 mask), all PASS:**
  - GATE A recall equivalence (above).
  - GATE B cross-rank selection identity: radix_on AND radix_off — all 8 TP ranks byte-identical; +
    no-dense-fallback `num_violations=0`.
  - GATE C production-reuse edge (num=144): boundary vs cold **−0.38pp** CI95[−0.70,−0.06], partial@~2752
    (~63% hit) **−0.012pp** CI95[−0.29,+0.26], evict **0.0pp** (clean recompute) — all within ±0.5pp;
    nearfull (~98%, OUT-OF-CONTRACT) **+1.5692pp** CI95[1.28,1.87], recorded value-affecting (matches loop11
    R27), NOT a gate input.
- **Fixture minted** → `serve_double_sparsity_radix_fixture.json` (schema v2, all 5 booleans true, DEC-1
  fingerprint `channel_mask_content_sha256=35155ac4…`).
- **No-override boot AUTHORIZED live:** DS boots radix-on via the fixture with NO dev override (validator
  "fixture recorded as PASSED"); `/server_info` shows the fixture artifact set + `disable_radix_cache=False`;
  real radix hit (warm cached 1088/1135). **DEC-1 path portability:** same-content mask at a SECOND path,
  same fixture → still authorizes.

### Capacity + AC-7 (AC-0.3, AC-7) — ✅
- **DS table-free radix-on @ mem 0.8** (production config, fixture-authorized): `token_capacity=504640`
  (= loop11 reference), `effective_max_running_requests_per_dp=64`, derived decode-bs cap ≈109 ≥ 64,
  CUDA-graph capture OK on all 8 ranks (153 s), **no TokenLabelTable**.
- **DSA-native un-regressed @ mem 0.8:** `token_capacity=410560` (= loop11 reference exactly),
  `enable_double_sparsity=False` — the DEC-1 shared-surface change did not touch the shipped default.

### DS concept intact (AC-6) — ✅
- Served path = offline channel mask → absorbed-latent table-free selection → top-k → sparse MLA decode;
  no dense fallback (`no_dense_fallback_passed`), no DSA-indexer substitution.

**M-A verdict: the GLM-5.1-FP8 table-free DS op-point is fully re-established on the fresh node, radix-on
authorized via the DEC-1 content-hash fixture, capacity reproduced, DSA-native un-regressed.**

Evidence: `runs/20260616_ma/` (provenance.json, capacity_ds_evidence.md, mint/ probes + verdicts,
server_info snapshots). Reproducers: `run_calibrate.sh`, `build_corpus.py`, `mint/{env,gate_*,mint_fixture,
verify_no_override,dsa_capacity_probe}`.

---

## M-B / M-C — pending (next)
The per-step tax guard (AC-4), the locked DS-vs-DSA sweep (AC-2/AC-3/AC-9, 2 op-points), the headline report,
the UX pass, and close-out follow. Methodology + the seven sweep-tooling side issues (SI-1..7) are recorded
in `queue.md`.
