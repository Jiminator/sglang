# Round 16 Contract

## Mainline Objective (exactly one)
**AC-8 — lifted ~70K-token `/generate` servability probe.** At the lifted DS operating point
(int8 compact table @ `mem_fraction_static=0.7`, radix-on), demonstrate that a ~70K-token raw
`/generate` is now **ADMITTED (HTTP 200)** — no longer the Loop-5 mem-0.6 `HTTP 400 "Input length
(69970) exceeds the maximum allowed (53050)"` — recording the served `max_total_num_tokens`, the
real `prompt_tokens`, and **no OOM / no instability**. If ~70K still does not fit, publish a
**characterized new admission ceiling** with the server-side reason (acceptable per the Lower Bound
+ DEC-9 soft handling). This is the Codex R15-review Required-Plan step 1 ("complete AC-8 next").

## Target AC(s)
- **AC-8 (task9)** — 64K servability; soft (DEC-9). `coding` / hardware-run / owner claude.

## Blocking Side Issues (truly block this objective)
- **None.** AC-8 is a single-node, node-0-localhost DS probe (only the DS int8 server is needed; no
  DSA, no cross-node). The lifted operating point (mem 0.7, `max_total_num_tokens=396096`, int8
  table, radix-on) is already verified by AC-4/AC-5/AC-7; this round reuses it. The cross-node
  wrapper-smoke issue does NOT block AC-8 (no cross-node benchmark artifact is produced).

## Queued / Explicitly Out of Scope This Round
- **AC-5 strict-SLO remediation** — the open mainline blocker and Codex Required-Plan **step 2**;
  it comes AFTER AC-8 is review-clean. Do **not** start it this round (keeps the mainline single).
- **Gated AC-10** (Tier-2 adjustable-`top_k` kernel) — only after AC-3..AC-9 are all verified.
- **DSA-default conc-64 TPS ~29.4** — pre-existing DSA/H200 limit (R12 user decision), queued.
- **Cross-node wrapper smoke** — future-gated; not exercised this round (single-node probe).

## Concrete Success Criteria
1. DS int8 server boots at `mem_fraction_static=0.7`, **radix-on** — proven from `/get_server_info`
   (`signature_dtype=int8`, `disable_radix_cache=false`, fixture artifact set, int8 `token_label_table`
   on all 8 ranks) — i.e. the identical lifted operating point as AC-4/AC-5/AC-7.
2. A ~70K-token probe payload committed as the **named AC-8 deliverable** `development/loop6/probe_64k.json`
   (raw `/generate` text + recorded token count + provenance).
3. The probe `/generate` returns **HTTP 200** with: the real `meta_info.prompt_tokens` (~70K, > the
   Loop-5 mem-0.6 pool 53056), the served `max_total_num_tokens=396096` recorded, **0 OOM lines** in
   the server log, server **alive** before+after (`/get_server_info` 200). The HTTP path is the SAME
   admission check that returned 400 at mem-0.6 — so a 200 here is the lifted-mem retry, not a
   silent re-record (AC-8 negative test). If it 400s/OOMs instead → record the **characterized
   ceiling** with the verbatim server reason (still a valid soft outcome).
4. Durable **tracked** evidence under `runs/20260530_dsv32_loop6/ac8_servability/` as `.txt`/`.json`/`.md`
   (never gitignored `.log`/`.jsonl`/`.csv`): the request, the response (status + prompt_tokens +
   served max_total + output snippet), `/get_server_info`, a server-log excerpt (admit + 0 OOM), and
   `ac8_64k_servability.md` contrasting Loop-5 mem-0.6 HTTP-400 vs the lifted mem-0.7 result.
5. GPUs freed (servers killed) at round end; commit + push to `jimmy`; goal-tracker mutable section
   + round-16-summary.md + BitLesson Delta updated.

## Applicable BitLessons (selected; confirm per-task via bitlesson-selector)
- `BL-20260530-remote-server-launch` — boot via Bash `run_in_background`; verify with
  `ps -eo pid,stat,cmd | grep "[s]glang.launch_server"` + GPU climb (NOT `pgrep -f`); `pkill ... || true`;
  never chain foreground `sleep`.
- `BL-20260529-ds-radix-flip-config-bound-artifact` + `BL-20260528-dsv32-ds-serving-boot-chain` —
  boot DS int8 radix-on (`SIGNATURE_DTYPE=int8`, `RADIX_FIXTURE_ARTIFACT`, `MEM_FRACTION_STATIC=0.7`).
- `BL-20260529-ds-longcontext-needle-recall-vs-topk` + `BL-20260529-gate-record-artifact-before-raise`
  — AC-8 tests **servability/admission** (does the ~70K prompt no longer 400), NOT recall; capture
  the HTTP status + body as a recordable result (a rejection is a characterized ceiling, not a crash).
- `BL-20260530-durable-tracked-acceptance-evidence` — embed tracked `.txt`/`.json`/`.md`; confirm with
  `git status --ignored` + `git diff --check`.

## Out-of-bounds reminders
No FlashMLA decode-assert changes (AC-3.3). No new serve/bench scaffolding — reuse `serve_double_sparsity.sh`
(the ~70K probe payload is the one named exemption). No plan-process tokens (AC-/DEC-/Tier/Option B/Round N)
in code/comments. Do not change the DS-fair AC-12 gate. Must not exit by lying / editing loop state / cancel.
