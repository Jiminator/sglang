# Round 5 Contract

## Mainline Objective
**Complete task9 / AC-Q — the paired quality smoke, run single-node sequentially.** Fix the
harness so it captures the DSA reference outputs first (with only the DSA server up), then —
after DSA is shut down and DS is booted — loads that reference artifact and runs DS against
it, evaluating and recording all four AC-Q gates. Then actually run it on 8x H200 and save
the artifact under `runs/20260528_dsv32_mvp/`.

## Target ACs (≤ 2)
- **AC-Q** — `test_dsv32_quality_smoke.py` compares DS-on vs DSA on 20 deterministic prompts
  (+5 NIAH-mini), single-node sequential (DSA reference captured first, then DS). Four gates:
  `prefix_match_rate >= 0.80`, `mean_rouge_l >= 0.85`, `niah_mini_recall >= 4/5`,
  `first_8_tokens_divergence == 0`. Any single gate below threshold fails AC-Q.

## Blocking issues in scope
- **#G — the AC-Q harness requires two simultaneous TP=8 servers.**
  `test/manual/test_dsv32_quality_smoke.py:231-234` skips unless BOTH `DS_BASE_URL` and
  `DSA_BASE_URL` are set, and `_run_paired` (`:250-260`) interleaves DSA→DS per prompt.
  Two TP=8 servers cannot co-reside on one 8-GPU node (DEC-2). Fix: add a DSA-reference
  **capture** mode (writes the 20+5 DSA outputs to a JSON artifact) and a DS **compare**
  mode (loads that artifact while DS is running, generates DS outputs, evaluates + records
  all four gates). Keep the legacy simultaneous mode for environments that can run both.

## Queued / explicitly out of scope this round
- **#F — DS KV-pool / effective-concurrency limit at mem 0.6.** Does NOT affect AC-Q (the
  quality smoke issues prompts one at a time with short outputs, so a single request fits the
  KV pool). #F blocks the AC-11 directional TTFT comparison only; resolve before task13.
- **TIER-2:** task11 AC-10 radix flip, task12 AC-1b chunked-prefill probe, task13 AC-11
  sweep, task14 AC-12 full quality, task15 evidence bundle.
- Stale `calibrate.py` operator recipe docstring (queued cleanup).
- Raw `*.jsonl` storage/transfer policy for the final bundle (task15 concern).

## Round success criteria
1. The quality-smoke harness supports single-node sequential operation:
   - a **capture** path (DSA URL only → writes a DSA-reference JSON: prompts, DSA texts,
     NIAH texts, DSA commit), and
   - a **compare** path (DS URL + the reference JSON → generates DS outputs, computes the
     four gates, writes the AC-Q artifact, and fails loudly if any gate is below threshold).
   - The legacy simultaneous (`DS_BASE_URL`+`DSA_BASE_URL`) mode still works.
2. A CPU/mock regression verifies the capture→compare wiring (round-trip the reference
   artifact through the gate evaluator) without needing live servers, so the mechanism is
   provably correct independent of the hardware run. Full DS unit suite stays green.
3. Hardware: boot DSA (radix-off, cluster path), capture 20+5 references → artifact; shut
   down DSA; boot DS (radix-off, cluster path), run compare → AC-Q artifact under
   `runs/20260528_dsv32_mvp/` recording all four gate values + pass/fail. Report honestly
   (a gate miss is an AC-Q failure + documented follow-up, not hidden).
4. Commit + push (per the between-rounds preference). Goal tracker updated (task9 status,
   #G resolved); `round-5-summary.md` written with a BitLesson Delta. No immutable-section
   changes.

## Known risks / notes
- DS sparse-decode could diverge from DSA on some prompts → a gate could fail. That is a
  real AC-Q result; record it and file a follow-up rather than masking it. First make the
  mechanism correct (criteria 1-2), then run and report (criterion 3).
- Operational: do not kill pre-existing processes I did not create (port 30000 router); use
  a free port; verify `nvidia-smi` clear before each boot; run `pkill`/`commit`/`push` as
  standalone commands (their exit codes abort compound commands).
