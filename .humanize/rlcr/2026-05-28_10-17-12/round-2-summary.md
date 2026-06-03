# Round 2 Summary — First DS boot on V3.2 FP8 (M2 Phase A)

## Round objective (round-2-contract.md)
First DS boot smoke on DeepSeek-V3.2 FP8 with the calibrated mask, and capture the
evidence the boot yields (AC-1 boot, AC-0 hardware capture probe, AC-1.1 sparsity,
AC-6 CUDA-graph status).

## Outcome
The DS server now BOOTS and SERVES on V3.2 FP8 (TP=8) — the DS serving path had
never run on hardware, and this round fixed the chain of latent blockers that the
first boot surfaced. AC-1 boot + AC-6 are met; AC-0-hardware, AC-1.1, and AC-Q are
blocked by two newly-discovered DS-runtime issues (decode degeneration + DS
meta_info not surfacing).

## Code fixes (commit `34b243b07`; regressions in `44a12d5d1` evidence)
Four latent blockers, each found by a successive boot attempt and fixed:
1. **validator.py** — `is_deepseek_nsa` → `is_deepseek_dsa` (model_config renamed it).
   Stale name → ImportError at server startup. +regression (capability symbol guard).
2. **deepseek_v2.py:1516** — DS-enablement branch `self.use_nsa` → `self.use_dsa`
   (set from `is_deepseek_dsa(config)`). Stale attr → AttributeError at model
   construction. +regression (source guard).
3. **deepseek_v2.py:1979** — move bound `_ds_channel_selection` to the label-table
   device. It loaded on CPU; the KV-write hook gathers GPU-resident K_nope with it
   as the index → `RuntimeError: index on cpu, tensors on cuda` in the MHA_ONE_SHOT
   write path during warmup.
4. **serve_double_sparsity.sh** — add `--mem-fraction-static` (default 0.6). Stock
   0.897 OOMs at boot (no room for the ~31 GiB DS TokenLabelTable); 0.7 boots but
   OOMs during generation; 0.6 boots + serves stably (~38 GB runtime headroom).

## Evidence (`runs/20260528_dsv32_mvp/`)
- `ds_boot_*.log` — successive boots; the final (mem_fraction 0.6) reaches
  "The server is fired up and ready to roll!".
- `ds_boot_knobs_AC1.json` — AC-1 knobs from server_args: model_path = cluster path
  (not HF-id), tp_size=8, kv_cache_dtype=fp8_e4m3, page_size=64,
  enable_double_sparsity=True, disable_radix_cache=True, disable_cuda_graph=False,
  disable_piecewise_cuda_graph=True, disable_overlap_schedule=True,
  attention_backend=dsa. `/get_model_info` independently confirmed the cluster path.
- `ds_generate_probe.json` — `/generate` "The capital of France is" → `" Paris."`
  (correct prefill) then a repeated token (decode degeneration).

## AC status this round
- **AC-1 (boot smoke): MET (core).** Boots single-node TP=8 with the cluster
  MODEL_PATH (DEC-6) + the mask; validator accepts (radix-off); all 8 GPUs; KV cache
  + DS TokenLabelTable allocated; DS `bind_runtime_data completed` on all ranks;
  `/generate` returns non-empty text; knobs confirmed. (`/get_server_info` endpoint
  is flaky in this build — empty response — so knobs are from server_args +
  `/get_model_info`.)
- **AC-6 (regular CUDA-graph status): MET.** Regular CUDA-graph capture completed all
  52 batch sizes at boot (capture success), distinct from `--disable-piecewise-cuda-graph`
  (piecewise disabled, logged). Recorded.
- **AC-0 hardware capture probe (task2): BLOCKED** — `meta_info` has no
  `double_sparsity_radix_capture` (see Blocking #B).
- **AC-1.1 (genuine sparsity, task6): BLOCKED** — no `double_sparsity` meta to read,
  and decode output is degenerate (see Blocking #A, #B).

## UPDATE (same round, continued) — #16 decode degeneration FIXED; new #C/#18 found
After the localization below, #16 was fully fixed and validated on hardware, and a
distinct deeper bug (#C/#18) was discovered:
- **#16 FIXED (commits `2af5f4e65` + `8375b76a5`).** Two decode-path bugs: (1)
  `req_to_token` None during decode → wrong selection domain + skipped
  `logical_to_physical`; fixed via ForwardContext resolution. (2) decode tokens never
  label-written — `kv_b_proj` not on `attn_mqa` AND `head_width` derived from
  `attn_mqa.v_head_dim` (512 vs real 128); fixed by attaching `kv_b_proj` to `attn_mqa`
  and deriving `head_width` from the projection output. Validated: short-prompt decode
  coherent ("Paris. The capital of the United States is Washington, D.C. ..."),
  `selected_tokens` grows with seq (was frozen at prompt_len), `dense_fallback=0`.
  253 DS unit tests pass.
- **#C/#18 NEW (OPEN, task #18).** With #16 fixed, DS serves coherently for
  seq<top_k=2048 but ANY request with seq>top_k crashes with `cudaErrorIllegalAddress`
  in `_select_topk_indices` (the genuine-sparse path). Bisected: 1376/1933 tok OK,
  2316/~3500 tok crash. Blocks AC-1.1 (needs seq>top_k) + real-shape benchmarks.
  Next round: compute-sanitizer to localize the OOB kernel, then fix. Evidence:
  `runs/20260528_dsv32_mvp/sparse_path_oob_finding.md`.

## New blocking issues — investigated + localized this round (fix is round 3)
- **#A: DS decode degenerates (DS-specific selection over-count).** Decisively
  localized with three on-hardware experiments:
  1. **DSA baseline control** (`serve_native_nsa.sh`, same model + `dsa` backend +
     fp8 KV + flashmla_kv, NO Double Sparsity) → coherent output (" Paris. 法国的首都
     是巴黎。 The capital of Italy is Rome. ..."). So the V3.2 dsa serving stack is
     correct; the bug is **DS-specific**.
  2. **DS eager** (`--disable-cuda-graph`) degenerates identically → **not
     CUDA-graph-related**; the bug is the core DS selection.
  3. The surfaced `double_sparsity` meta shows `valid_lengths` EXCEEDING `seq_len`
     (negative `sparsity_rate`; `selected_tokens=19` when seq≈12, and `=5` when
     seq≈28) — DS over-/mis-selects. **Single-step decode over a clean prefill is
     correct (" Rome"); multi-step corrupts.** Localized to
     `retrieve_topk_via_labels` logical selection returning more valid tokens than
     the sequence length → wrong/duplicate physical slots → garbage decode
     attention. Round-3 fix plan in `decode_degeneration_diagnosis.md`. Task #16.
- **#B: DS per_request_summary not surfacing in `/generate` meta_info (graph mode).**
  Root cause found: `_publish_ds_request_summary` is intentionally skipped during
  CUDA-graph capture/replay (deepseek_v2.py:2326-2329, host `.item()`/`.tolist()`
  are illegal under capture), and decode runs under CUDA graph. In **eager** mode the
  `double_sparsity` summary DOES surface (used above to diagnose #A). The AC-0
  hardware capture probe (graph mode) needs a capture-safe publish path or an
  eager-mode probe. Task #17.

## Commits (round 2)
- `34b243b07` — the four DS serving-path fixes (validator rename, use_dsa, channel
  device, mem-fraction) + rename regressions.
- `44a12d5d1` — AC-1 boot knobs + DS `/generate` probe evidence.
- `610b65c15` — `EXTRA_SERVER_ARGS` launcher passthrough + decode-degeneration
  control/diagnosis artifacts (DSA baseline + DS eager probes).

## Operational notes
- DS server booted on **port 30010** (port 30000 held by a pre-existing orphaned
  `sglang::router` the agent is not authorized to kill; downstream tasks take
  configurable URLs).
- All servers shut down after probing; GPUs free.

## Tests
- `python -m pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
  → **253 passed** (was 251; +2 rename guards).

## Remaining / next round
Resolve #A (decode degeneration) and #B (meta surfacing) — without them AC-1.1, the
AC-0 hardware probe, and AC-Q cannot be demonstrated. #A is the priority (it questions
DS output correctness on V3.2). Then proceed to M2 Phase B/C (smoke benchmarks,
comparator, paired quality smoke).

## Goal Tracker Update Request
### Requested Changes
- Mark **AC-1 (boot smoke)** met (core) and **AC-6** met, with the evidence above.
- Add **Blocking Side Issues #A (DS decode degeneration)** and **#B (DS meta_info not
  surfacing)**; mark **AC-0 hardware probe (task2)** and **AC-1.1 (task6)** blocked on them.
- Record the four serving-path fixes + the mem_fraction default (0.6) in the Plan
  Evolution Log (round 2). No immutable-section or AC-definition changes.
### Justification
The boot (M2 Phase A core) is achieved and committed; the two findings are genuine
DS-runtime correctness/transport bugs that block the remaining AC-1 evidence and must
be resolved before benchmarks/quality. Documenting them as blocking issues keeps the
mainline honest and sets the next round's focus.

## FINAL STATE (end of round 2) — DS-on V3.2 works end-to-end (TIER-1 smoke core)
After the model cutover, the round continued to a working DS-on MVP:
- **#16 decode degeneration FIXED** (commits `2af5f4e65`, `8375b76a5`): coherent decode,
  decode tokens labelled, selection tracks the sequence.
- **#18 long-prompt OOB FIXED** (commit `eba4c640e`) via the user's domain insight that
  classic DS is **dense-prefill / sparse-decode**: keep dense MHA prefill for DS, run DS
  selection only at decode. A 2272-token (>top_k) prompt now serves; **AC-1.1 satisfied**
  (sparsity_rate=0.105, selected_tokens=2048, dense_fallback=0; `ac1_1_genuine_sparsity.json`).
- **Radix-capture CUDA-graph crash FIXED** (commit `8e9138af6`): capture+graph boots cleanly.
- **Critical review of loops 1-5** delivered (`REVIEW_loop4_loop5_precutover.md`): all
  defects were loop4-integration / loop5-calibrate "never ran on hardware" bugs; the
  loops 1-3 foundations are sound and now hardware-validated.

TIER-1 smoke status: AC-4 (mask), AC-1 (boot), AC-1.1 (genuine sparsity), AC-6 (CUDA-graph
status), AC-0 (producer code+regression) — DONE. Remaining TIER-1: AC-8/9 smoke benchmarks,
AC-8/9 comparator, AC-Q paired quality smoke. Remaining TIER-2: AC-10 radix flip (+ the
radix-capture /generate surfacing, now a queued TIER-2 item), AC-1b, AC-11, AC-12.

## BitLesson Delta
- Action: add
- Lesson ID(s): BL-20260528-dsv32-ds-serving-boot-chain, BL-20260528-dsv32-ds-decode-degeneration, BL-20260528-dsv32-ds-dense-prefill-sparse-decode, BL-20260528-ds-radix-capture-cuda-graph-safe
- Notes: Added a lesson capturing the V3.2 DS-serving boot-blocker chain (NSA→DSA
  rename stragglers in validator.py + deepseek_v2.py; the channel_selection CPU/CUDA
  device-placement bug in the KV-write hook; and the DS mem_fraction_static headroom
  for the TokenLabelTable + CUDA graphs — 0.897 OOMs at boot, 0.7 OOMs in generation,
  0.6 serves). `bitlesson-selector` for the boot returned
  {BL-20260527-ds-metadata-via-forward-context, BL-20260527-shell-json-into-python-source};
  both were applied (server JSON read via `json.load(sys.stdin)`, not source-spliced;
  the metadata-via-ForwardContext lesson informs the #B meta-surfacing investigation).
