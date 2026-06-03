# Round 10 Summary — AC-6 (opt-in / DSA-default product) + AC-9 (real-token within-budget) on hardware

## Mainline objective (round contract)
Per Codex's R9 directive (AC-5 evidence verified/resolved; move to hardware): land
the DS-opt-in / DSA-default product property (AC-6) and the real-token within-budget
gate (AC-9) on hardware, via one cross-node bring-up (DS int8 @ 0.7 on node 0,
DSA-default on node 1). The AC-5 directional verdict + open strict-SLO blocker stay
tracked but are not this round's objective.

## AC-9 — within-budget gate from real `usage.prompt_tokens` (commits d6e884aa9 code, daad92923 evidence)
- Harness edit: `_generate` returns `(text, prompt_tokens)` from `usage.prompt_tokens`
  (chat) / `meta_info.prompt_tokens` (generate); threaded through `_GenAttempt` →
  `_summarize_prompt_tokens` (max-over-served `input_tokens` + `usage_missing`
  fail-closed signal) → `_run_niah` → `_niah_record`. `within_budget` now computed
  from real `input_tokens` (→ fail-closed `None` if usage missing); records
  `input_tokens`, `dsa_input_tokens`, and the old `within_budget_wordcount_proxy`.
  `test_niah_within_budget` asserts the premise from real tokens. Renamed the
  misleading `length_tokens` → `length_words`. **DS-fair gate definition UNCHANGED**
  (INDEX_TOPK=2048, 5 pp tolerance, 1024/1536-word lengths). Dry-run verified the
  parsing + fail-closed logic before hardware.
- Live re-run (DS node0 + DSA node1) **PASSED** (1 passed, 2 subtests, 26.5 s):
  1024 words → `input_tokens=1128`; 1536 words → `1678`; both `within_budget=True`,
  `usage_missing=False`; DS recall 100% vs DSA 100% (Δ0.0 pp). The real-token
  `within_budget` **matches** the word-count proxy at both lengths ⇒ **the proxy was
  safe** (recorded per-length). Artifacts: `ac9_real_token_within_budget.md`,
  `ac9_within_budget/ac12_niah_{1024,1536}_*.json`.

## AC-6 — DS opt-in; DSA stays the default (DEC-2 "Both")
- **The opt-in flag toggles the compact DS path** (`ac6_product_proof/get_server_info_keys.json` + boot logs):
  - DS opt-in (node 0): `enable_double_sparsity=True`, `double_sparsity_config={…,"signature_dtype":"int8"}`,
    `token_label_table: 6.48 GB/rank … dtype=torch.int8 scales=float16` on all 8 TP ranks.
  - DSA-default (node 1): `enable_double_sparsity=False`, `double_sparsity_config=None`,
    **0** `token_label_table` lines, full **910784**-token KV pool — **no DS table allocated**.
  - Identical Option B operating point (fp8 KV, page 64, flashmla_kv prefill+decode, overlap/piecewise disabled).
- **DSA-default admits full nominal concurrency** and serves cleanly: achieved 16.00 / 32.00 / [64]
  at conc 16/32/64, completed 64/64 each, **errors 0** (`ac6_product_proof/dsa_default_slo.txt`).
- **"Meets the SLO unchanged":** the authoritative DSA steady-state SLO is the established
  Loop-5 baseline (P99 TTFT 0.73 / 1.37 / 2.04 s, ≥30 TPS) at this identical operating point;
  this fresh boot reproduces that operating point exactly. The fresh `WARMUP=0` confirmation run
  is **cold-ramp-dominated** (DSA P99 TTFT 22.5 s / TPS 16.9 at conc 16) — the **same flood
  artifact AC-5 documented for DS** (min TTFT 1.6 s; tight median≈p99); under identical
  `WARMUP=0` methodology DSA is **not** faster than DS (AC-5 conc-16 12.8 s), confirming the
  inflation is the cold ramp, not DS-specific. A clean all-trials steady-state DSA sweep is AC-7.

## Result
AC-9 met (code + live rerun, proxy shown safe). AC-6 product property met (opt-in toggle +
DSA no-table + full admission + clean serving). The **strict-SLO miss remains the open
mainline blocker** (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc) — unchanged.

## Files Changed
- `test/manual/test_double_sparsity_v32.py` (AC-9 harness; commit d6e884aa9).
- `runs/20260530_dsv32_loop6/`: `ac9_real_token_within_budget.md`, `ac9_within_budget/` (daad92923);
  `ac6_optin_dsa_default_product.md`, `ac6_product_proof/` (get_server_info ×2 + keys, boot excerpts, dsa_default_slo.txt).
- `.humanize/bitlesson.md` (+1 lesson `cold-flood-not-steady-state-slo`), goal-tracker, round-10 contract/summary (gitignored loop state).

## Validation
- Cross-node bring-up: DS int8@0.7 node0 (`token_label_table 6.48 GB/rank int8`, no OOM) + DSA-default node1 (no table, pool 910784).
- AC-9 gate PASSED on hardware; `within_budget` from real tokens, proxy safe.
- AC-6 toggle/no-table proven from `/get_server_info` + boot logs; DSA full admission (achieved==nominal), errors 0.
- AC-9 code dry-run (mock responses) confirmed usage capture + fail-closed before hardware; `git diff --check` clean; commits pushed to `jimmy`.
- Servers killed and GPUs freed after capture.

## Remaining Items
- **Open mainline blocker:** strict client SLO still fails (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc).
- **AC-7** (3-trial DS+DSA lifted-point re-sweep, 120/600 s — also gives the clean steady-state DSA SLO),
  **AC-8** (~70K-token servability probe), gated **AC-10**. No FlashMLA decode-assert changes (AC-3.3).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260530-cold-flood-not-steady-state-slo
Notes: Added a lesson from the AC-6 SLO confirmation: a `WARMUP=0` / `request_rate=inf` + `max_concurrency` flood run inflates TTFT/TPS for the NATIVE baseline (DSA) too (DSA conc-16 P99 TTFT 22.5 s / TPS 16.9 vs the established steady-state 0.73 s) because the cold ramp floods `max_concurrency` simultaneous 4096-prefills → prefill/decode contention. Such a run validates ADMISSION (achieved==nominal) + clean SERVING (errors 0) but NOT steady-state latency; the SLO number must come from a proper-warmup baseline at the identical operating point. Tell: tight median≈p99 well above a small `min`. Cross-checking the native baseline under identical methodology also retro-validates the AC-5 directional caveat (its WARMUP=0 run over-states DS TTFT). Reinforces BL-20260530-admission-restore-tps-tradeoff. Applied existing lessons: BL-20260530-durable-tracked-acceptance-evidence (tracked .json/.txt proofs + get_server_info under runs/) and the `pkill -f 'sglang::router'` router-kill gotcha for the cross-node bring-up.
