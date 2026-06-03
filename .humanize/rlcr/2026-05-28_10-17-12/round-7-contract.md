# Round 7 Contract

## Mainline Objective
**Finish the #H diagnosis with reviewable raw evidence of DS decode SELECTION on the exact
failing AC-Q prompts, then act on what it shows:** if DS drops context on a short
(seq ≤ top_k) prompt — less than full-context selection, or any dense fallback — that is a DS
selection/label bug; fix it (+ narrowest regression) and rerun AC-Q to `all_pass=true`. Only
if the metadata proves full-context selection with `dense_fallback == 0` AND the raw controls
prove the loop is purely temperature-0 greedy degeneration may I file ONE concrete
measurement-change request for approval (no harness default changed until approved).

Round 6 was STALLED because the eager==graph control only ruled out a CUDA-graph bug; it did
NOT capture DS `meta_info["double_sparsity"]` (selected_tokens vs seq_len, dense_fallback) and
so did NOT exonerate the selection/label path, and the concise/sampling controls were
markdown-only (not reviewable JSON).

## Target ACs (≤ 2)
- **AC-Q** — the four-gate paired quality smoke. This round's success is: a DS selection/label
  bug fixed + AC-Q rerun to `all_pass=true`, OR (if selection is proven healthy) reviewable
  raw evidence + one measurement-change request filed for approval, with AC-Q honestly
  recorded as still NOT MET pending approval.

## Blocking issues in scope
- **#H — DS selection/label path not yet exonerated for the short AC-Q failure.** Required:
  1. Capture reviewable RAW JSON for the exact chat-formatted failing prompts with DS
     `meta_info["double_sparsity"]` (or, if the API does not surface it on short prompts,
     targeted server-side instrumentation logging `selected_tokens`, `seq_len`,
     `sparsity_rate`, `dense_fallback` per decode step). First find out WHY
     `double_sparsity` was `None` on short prompts (read the publish path:
     `deepseek_v2.py::_publish_ds_request_summary`, `metrics.py`, the scheduler summary hop).
  2. Healthy shape for seq ≤ top_k: full-context selection (`selected_tokens == seq_len`/all),
     `sparsity_rate == 0` (nothing dropped), `dense_fallback == 0`. If that does NOT hold →
     DS selection/label bug → fix it before any measurement discussion.
  3. Save Codex's exact artifact set: DS graph temp-0 `17*23`, DS eager temp-0 `17*23`, DS
     graph temp-0 primes, DSA temp-0 `17*23`, DS concise `17*23`, DS concise primes, DS
     temp-0.5 `17*23` — each with request body, server info, generated text, and DS metadata
     where applicable.

## Queued / explicitly out of scope this round
- **#I** — RESOLVED + verified (exact-fixture validation). No further work.
- **#F** — DS KV-pool/effective-concurrency at mem 0.6 (blocks AC-11 TTFT only).
- **TIER-2** — task11 AC-10, task12 AC-1b, task13 AC-11, task14 AC-12, task15 bundle. Do NOT
  start until AC-Q passes under the immutable gate or an approved measurement change.
- Stale `calibrate.py` operator recipe docstring.

## Round success criteria
1. Reviewable raw JSON artifacts saved under `runs/20260528_dsv32_mvp/` for the exact failing
   prompts, including DS selection metadata (or an instrumentation log) that shows
   `selected_tokens`/`seq_len`/`dense_fallback` on the seq ≤ top_k decode. The cause of the
   `double_sparsity == None` observation is explained from the code path.
2. A definitive verdict backed by that evidence: DS selection is full-context (no bug) OR DS
   drops context (bug).
3. If bug: fix the DS selection/label/summary path with the narrowest regression; reboot DS;
   rerun the sequential `capture`→`compare` AC-Q workflow; `all_pass=true`.
   If no bug: the raw controls (concise→correct, sampling→escape, eager==graph) are committed
   as JSON, and ONE measurement-change request is filed for approval; AC-Q stays NOT MET
   pending approval. No harness threshold/prompt/decoding default changed unilaterally.
4. DS unit suite stays green; commit + push each step. Goal tracker updated (task9, #H);
   `round-7-summary.md` with a BitLesson Delta. No immutable-section changes.

## Known risks / notes
- `double_sparsity` meta was `None` via both `/generate` and `/v1/chat/completions` on short
  prompts; surfacing per-decode selection may require reading the publish gate and/or adding
  a temporary, env-gated instrumentation log (remove or keep env-gated; do not leave a hot
  host-sync in the CUDA-graph decode path — see BL-20260528-ds-radix-capture-cuda-graph-safe).
- Operational: don't kill the pre-existing port-30000 router; use a free port; verify
  `nvidia-smi` clear before each boot; standalone `pkill`/`commit`/`push`.
