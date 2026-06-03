# Round 7 Contract

## Mainline Objective (exactly one)
**Make the AC-5 acceptance evidence durable and the attribution clean**, so task6/AC-5 is verifiable. The R6 run is real and the TTFT movement is verified; this round closes the two evidence/attribution blockers Codex's R6 review found (no re-run — the local JSONLs + server request-time-stat log are present). It is an evidence/correction round (no production code).

## Target ACs (1–2)
- **AC-5** (`coding`, hardware-derived) — complete the durable evidence + corrected attribution addendum.

## Blocking Side Issues in Scope (Codex R6 review)
1. **AC-5 benchmark evidence not recomputable from tracked files.** The raw `.jsonl` are gitignored (`*.jsonl`), so percentiles can't be independently recomputed from committed artifacts. Fix: add a tracked `ac5_evidence_addendum.txt` with, per conc: completed count, duration, error count (all-empty proof), input/output length distributions, and the TTFT/TPOT/ITL percentile sources (the exact arrays or their recomputable summary) + the sidecar path. Derived from the local JSONLs.
2. **Attribution not clean enough to claim the spine "validated by measured attribution."** The R6 excerpt has an **invalid negative `queue_duration` row** and `N=959≠960`, and `forward_duration` (completion-time prefill+decode) was misused as a first-token "prefill floor." Fix: reprocess the **full** server request-time-stat log **per concurrency**, recording expected rows vs parsed rows, invalid/negative/health rows + the filtering policy, queue-duration p50/p95/p99 per conc, TTFT p99 + residual, and a **corrected measured-vs-inferred** statement (admission-wait = `queue_duration` measured; first-token prefill = `TTFT − queue_duration` inferred; `forward_duration` is completion-time, not first-token). Add the server **decode-batch** excerpt as tracked evidence for the TPS root cause.
3. **Report language.** Soften "spine validated by measured attribution" → **"directional characterization"** until the attribution is clean; keep the **strict SLO miss explicit** (conc-32/64 TTFT > 22 s; per-req TPS < 30 at every conc).

## Queued / Out of Scope (but explicitly NOT downgraded)
- **Strict-SLO failure remains a visible mainline SLO blocker** (per Codex): conc-32/64 TTFT and the per-request-TPS-vs-admission tradeoff block the ultimate done-criterion; they stay tracked as blocking, not queued.
- **AC-6 hardware proof, AC-7 (3-trial DS+DSA re-sweep), AC-8 (~70K probe), AC-9 (within-budget harness edit), gated AC-10** — later rounds. No FlashMLA decode-assert changes (AC-3.3).

## Round Success Criteria
1. Tracked `ac5_evidence_addendum.txt` (per conc: completed/duration/errors/ISL+OSL distributions/TTFT+TPOT+ITL percentiles + source) — AC-5 numbers recomputable from committed files without the gitignored `.jsonl`.
2. Tracked `attribution_per_conc.txt`: per conc queue-duration p50/p95/p99, expected-vs-parsed rows, invalid/negative/health rows + filtering policy, TTFT p99 + residual, corrected measured-vs-inferred note; + a tracked `decode_batch_excerpt.txt` (the `#running-req` / gen-throughput lines backing the ~14 tok/s/req TPS root cause).
3. `client_slo_report.md` updated: "directional characterization" wording, corrected prefill-floor framing, strict SLO miss kept explicit, references the addenda.
4. `git diff --check` clean; commit + push; `round-7-summary.md` with BitLesson Delta; tracker (task6 + the strict-SLO blocker).

## Out-of-Scope Guards
- No re-run (use the present local files). No production code change. Directional verdict unchanged; this round only makes it durable + the attribution honest.
- Do not weaken the strict SLO or mark the loop done; the SLO miss stays a mainline blocker.
