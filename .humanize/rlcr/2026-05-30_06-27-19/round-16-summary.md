# Round 16 Summary — AC-8 64K servability PASS at the lifted DS operating point

## Mainline objective (round contract)
Codex R15-review Required-Plan **step 1**: complete AC-8 — at the lifted DS int8 / `mem_fraction_static=0.7`
/ radix-on operating point, demonstrate that a ~70K-token `/generate` is now **ADMITTED (HTTP 200)** rather
than the Loop-5 mem-0.6 `HTTP 400 "Input length (69970) exceeds the maximum allowed (53050)"`, recording the
served `max_total_num_tokens`, real `prompt_tokens`, and no OOM/instability — or a characterized ceiling.
AC-5 strict-SLO remediation (Required-Plan step 2) and gated AC-10 were explicitly out of scope this round.

## What landed (hardware round; commit below)
**AC-8 PASS (servability).** Single-node TP=8, node-0 localhost:
1. **Lifted operating point booted + proven** (identical to AC-4/AC-5/AC-7): DS int8 @ mem 0.7, radix-on via
   the config-bound fixture `ds_radix_fixture_state_int8.json` (sha `f3b67943`, both M3-B fixtures PASSED),
   `disable_radix_cache=False`, int8 `token_label_table` **6.48 GB/rank on all 8 ranks**
   (`dtype=torch.int8 scales=float16`), `max_total_num_tokens=396096`, `context_len=163840`,
   `chunked_prefill_size=8192`. Proven from `get_server_info_{before,after}.json` + boot log.
2. **Named AC-8 deliverable** `development/loop6/probe_64k.json` (the plan's one scaffolding exemption):
   deterministic varied prose (seed 20260531) + a one-line question, raw `/generate`, `max_new_tokens=16`,
   `temperature=0`, `text_sha256=652e4f51…`. Local tokenizer estimate **70759 tokens** == the server-reported
   `prompt_tokens=70759` (exact provenance). `70759 > 53056` (Loop-5 pool), `>= 69970` (Loop-5 64K reference),
   `< 396096` (lifted pool) — so it exercises the same admission length-check that 400'd at mem-0.6.
3. **Probe result** (`ac8_probe_response.json`): **HTTP 200**, served `max_total_num_tokens=396096`, generated
   16 tokens (`finish_reason=length`), latency 11.95 s, **server alive before AND after**. Server log:
   chunk-prefill 8×8192+5248 = 70759 (matches `prompt_tokens`), `#queue-req:0` (admitted immediately),
   token usage 0.02→0.18 of the pool, **0 OOM / CUDA-error lines** in the whole boot+serve log.

## Result
AC-8 PASS — the Loop-5 64K **HTTP-400 admission ceiling is removed** at the lifted operating point: a
~70K-token `/generate` now serves cleanly (HTTP 200, no OOM, server stable) with `max_total_num_tokens=396096`.
No characterized ceiling is needed (the prompt fits with large margin — 70759 of a 396096 pool, 18% token
usage). This is a **servability/admission** result; 64K **recall** accuracy is bounded by the kernel-locked
`top_k=2048` and remains a Tier-2/AC-10 concern, unchanged here. The raw-`/generate` output is a degenerate
continuation (no chat template) — irrelevant to admission, noted explicitly.

## Files Changed
- `development/loop6/probe_64k.json` — named ~70K-token AC-8 probe payload (NEW; the one allowed exemption).
- `runs/20260530_dsv32_loop6/ac8_servability/` (NEW): `ac8_64k_servability.md` (report + Loop-5 contrast),
  `ac8_probe.py` (reproducible driver — reads payload, asserts sha, captures before/after server-info, sends
  raw `/generate` catching rejections as recordable results), `ac8_probe_response.json`,
  `get_server_info_{before,after}.json`, `server_log_excerpt.txt` (chunked-prefill window + 0-OOM scan).
- `.humanize/bitlesson.md` — updated `BL-20260528-dsv32-hf-calibration-load` with the tokenizer-only corollary.
- goal-tracker (R16 Plan Evolution row; task9/AC-8 → done-pending-verification), round-16 contract/summary
  (gitignored loop state).

## Validation
- Boot config + `get_server_info`: `mem_fraction_static=0.7`, `signature_dtype=int8`, `disable_radix_cache=False`,
  `enable_double_sparsity=True`, `max_total_num_tokens=396096`, int8 table 6.48 GB/rank ×8.
- Probe HTTP 200; `prompt_tokens=70759` == local tokenizer estimate; served pool 396096; 16 tokens generated;
  server `/get_server_info` 200 before and after; 0 OOM lines (`grep -icE "out of memory|OOM|CUDA error|..."` = 0).
- Loop-5 contrast embedded: mem-0.6 pool 53056 → HTTP 400 for 69970 tokens vs mem-0.7 pool 396096 → HTTP 200
  for 70759 tokens (lifted-mem retry, not a silent re-record — AC-8 negative test satisfied).
- GPUs freed at round end (all 8 at 0 MiB, no live `launch_server`). `git diff --check` run before commit.

## Remaining Items
- **Open mainline blocker:** AC-5 DS strict client SLO (conc-32/64 TTFT > 22 s; per-req TPS < 30 every conc) —
  Codex Required-Plan **step 2**, the next mainline after AC-8 is review-clean: the smallest scheduling/decode/
  operating-point change to restore both `<22 s` and `≥30 TPS/req`, then a full re-run with exact arrays + a
  fail-closed verifier.
- **Gated AC-10** (Tier-2 adjustable-`top_k` kernel) — only after AC-3..AC-9 are all verified.
- **Cross-node wrapper smoke** — future-gated (this round was single-node localhost; no cross-node artifact).
- **DSA-default conc-64 TPS ~29.4** — queued pre-existing DSA limit (R12 user decision).
- No FlashMLA decode-assert changes; DS-fair AC-12 gate unchanged.

## BitLesson Delta

Action: update
Lesson ID(s): BL-20260528-dsv32-hf-calibration-load
Notes: Added a tokenizer-only corollary (clause e + Source Rounds += loop6 R16): to size a long-context probe to
an exact token count offline, `AutoTokenizer.from_pretrained` fails the SAME way as AutoConfig on the unregistered
`deepseek_v32` model type, so load `PreTrainedTokenizerFast(tokenizer_file=.../tokenizer.json)` directly (no config),
and cross-check the local count against the live server's authoritative `meta_info.prompt_tokens` (R16: 70759 == 70759).
Applied existing lessons without new failure modes: BL-20260528-dsv32-ds-serving-boot-chain + BL-20260529-ds-radix-flip-config-bound-artifact
(int8/mem-0.7/radix-on boot), BL-20260529-ds-longcontext-needle-recall-vs-topk (framed AC-8 as servability, not recall),
BL-20260529-gate-record-artifact-before-raise (probe driver catches HTTP rejections as recordable results),
BL-20260530-remote-server-launch (background boot + `ps | grep "[s]glang.launch_server"` + `pkill || true`; foreground
`sleep` blocked), BL-20260530-durable-tracked-acceptance-evidence (tracked `.json`/`.txt`/`.md`, exact sha provenance).
No new standalone lesson — clean execution of a well-understood hardware probe.
