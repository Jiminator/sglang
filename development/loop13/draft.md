# Loop 13 Draft — Find the cause of the DS-vs-DSA accuracy degradation

> Written 2026-06-20. This is a **diagnosis loop, not a feature/fix loop.** The goal is to
> explain *why* table-free Double Sparsity (DS) on GLM-5.1-FP8 is far less accurate than the native
> DSA indexer, and to localize the cause to either (a) the algorithm itself not transferring to
> GLM/MLA, or (b) a specific performance optimization we added during loops 6–12. We are not
> required to fix it this loop — the deliverable is a **root cause with evidence**, plus the reusable
> reference selector built to find it.
>
> Run everything on the **dev clone** `/sgl-workspace/sglang` (branch `dev/double-sparsity-standalone`)
> — the default editable `import sglang`. This is the implementation under investigation. (The
> `double-sparsity-v2` clone is a separate, refactored codebase; out of scope here.)
>
> Feed this through `gen-plan` once scope is confirmed.

---

## What this is (and is NOT) — read first

**This is a measurement-driven root-cause investigation.** No new selection algorithm, no perf
work, no SLO chasing. We build the *simplest correct* version of DS to establish an accuracy
ceiling, then bisect our own optimization history against that ceiling. Every claim must be a
GSM8K number from a live 8×H200 run, not a theory.

**The deliverable is a verdict, not a patch.** Either "the channel-selection algorithm does not
transfer to GLM-5.1 MLA (ceiling is bad — recovering accuracy needs research)" or "the algorithm is
fine (ceiling ≈ DSA); optimization X regressed it (here is the commit and the accuracy it costs)."
Fixing X is a *follow-up* loop.

---

## Background — the measured regression (Loop-12 session, 2026-06-20)

GSM8K, GLM-5.1-FP8, 8×H200 TP=8, same base/backend, only the token-selector differs. Greedy
(temp 0), completion API (no thinking-mode, no shot/question leakage). DS ran `--disable-radix-cache`
(the dev clone gates DS+radix; radix is output-neutral at temp 0). Both regimes single-request
reproducible.

| GSM8K (dev clone) | DSA (native) | DS (current table-free) |
|---|---|---|
| 5-shot / 200 — **dense** (~763 tok, seq < top_k 2048) | 0.970 | **0.625** (serial control 0.700) |
| 24-shot / 150 — **sparse** (~4.2k tok, seq > 2048) | 0.953 | **0.000** |

Failure mode (verified single-request, `finish_reason=length`): at long context DS **degenerates
into garbage** (`"...the the8�a00 the the … RRRRRRR0R0QRQRQ…"`) and runs to the token cap. DSA scored
0.953 on the *exact same* 24-shot prompts, so the prompt/model are fine — only the DS selector
corrupts generation. The server returns 200 throughout: the corruption is **silent** (the selector
picks a bad top-2048 set → the model attends to wrong KV). DS is also degraded in the **dense**
regime (~0.63–0.70 vs 0.97) where it selects *all* tokens and should equal plain attention — so the
problem is not purely the sparse cap.

Evidence: `development/loop12/gsm8k_evidence/` (the `*_short.out` / `*_long.out` run logs and the
single-request probe outputs).

---

## Hypotheses to discriminate (the whole point of the loop)

**H0 — the algorithm doesn't transfer.** Channel-importance selection (offline `mean|Q·K|` heavy
channels → approximate `Q_label·K_labelᵀ` → top-k → full attention) is fundamentally weaker than
GLM's *learned* DSA indexer on MLA. If true, even a perfect, slow, exact implementation is far below
DSA, and perf is irrelevant.

**H1 — a perf optimization regressed it.** The exact algorithm matches DSA, but one of the speed
optimizations we layered in loops 6–12 corrupts the selection. Leading suspects, with the history
that makes each plausible:

- **Raw-dot scoring (`scorer_norm` locked to `"off"`).** Loop 7 measured that a **cosine scorer
  took 16K NIAH recall from 5% → 40%** (`e2674f4f4`, `599d7cc99`). The table-free rewrite (Loop 11)
  then **removed cosine**: `config.py` now hard-locks `scorer_norm="off"` because the absorbed-latent
  identity `score = max_h v_h·c_kv` only holds for the raw dot — "direction-only norms would operate
  on a materialized per-head signature the selector never builds." So going table-free may have
  *reverted to the known-bad scorer*. **Top suspect.**
- **Table-free absorbed-latent fp8 scoring (Loop 11, `01e3ff238` deletes `TokenLabelTable`).** Scores
  are computed from the resident fp8 MLA latent with per-128-block dequant in-register, instead of a
  materialized signature. fp8 precision + the absorbed approximation could wreck score ordering.
- **bf16 score-reduce across TP (Loop 9, `c877d7fa1`).** The per-head scores are all-reduced in bf16;
  precision loss in the reduction can flip the top-k.
- **Approximate sequence-aware radix top-k replacing exact `torch.topk` (Loop 9, `859c8ee2c`).** A
  blocked/deterministic top-k that may not return the exact top-2048.
- **Selector-width ladder / compact W=5120 (Loop 10, `9956c240e`, `6c92240b9`).** Scores only a
  covering width; a 24-shot prompt (~4.2k) sits under the 5120 bucket, but the bucketing/keying logic
  is worth ruling out.
- **int8-symmetric signature compaction (Loop 6, `84d3410b9`).** Likely gone post-table-free, but
  confirm it isn't on any path.
- **Cross-head `head_agg="max"` shared selection.** MLA forces one token set shared across heads
  (the paper selects per-head). This may itself cost accuracy independent of the above.

**H2 — the offline channel mask is bad for GLM-5.1.** The calibrated mask (`mean|Q·K|` on the noPE
reconstruction) may simply not capture GLM's important channels. This sits under the H0 branch — the
reference selector reuses the existing mask, so if the ceiling is bad, mask quality is the next thing
to test (recalibrate, or vary `label_dim`).

---

## Execution harness — clone safety (do NOT launch the v2 clone)

**The hazard, stated plainly.** The default `import sglang` (the editable install) is the **dev clone**
`/sgl-workspace/sglang`. The **v2 clone** `/sgl-workspace/double-sparisty-v2/sglang` is a *different,
refactored* DS codebase. In the Loop-12 session we accidentally served the v2 clone by passing
`PYTHONPATH=$V2/python`, which silently invalidated the first DSA/DS numbers. **Every server in this
loop must run from the dev clone, with no `PYTHONPATH` override.**

**Ready-made scripts (shipped in `development/loop13/`, guard-enforced).** Use these; do not hand-roll
launch commands. Run `serve.sh` / `teardown.sh` **backgrounded** (they poll with `sleep`).

| Script | Purpose |
|---|---|
| `_env.sh` | Sourced guard + env. **Refuses to proceed** unless `import sglang` resolves to the dev clone and `PYTHONPATH` is clean; also blocks `expandable_segments`. Exports `MODEL`/`MASK`/`HOST`/`PORT`/`EVID`/`PIDFILE`. |
| `serve.sh <dsa\|ds>` | Boots GLM-5.1-FP8 from the dev clone (**no `PYTHONPATH`**), writes `$PIDFILE`, waits for `/health`. `ds` adds `--disable-radix-cache` + the dev-ABI DS config. |
| `probe_ds_active.sh` | Long-context request → asserts `selected < total`, `dense_fallback == 0` (DS genuinely active). |
| `run_gsm8k.sh <label>` | Dense (5-shot/200) + sparse (24-shot/150) GSM8K via the dev clone's `run_eval --api completion`. |
| `teardown.sh` | Kills only `$PIDFILE`, waits all 8 GPUs to ~0 MiB. Never blanket-kills. |

The guard is the load-bearing part — it has been verified to **stop** (FATAL, rc=1) the instant
`PYTHONPATH` points at the v2 clone, and to pass only when `import sglang` is the dev clone.

**Canonical launch commands** (what `serve.sh` runs — note **no `PYTHONPATH`**):

```bash
# DSA (native indexer, DS off) — the accuracy target
python3 -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port 30000 \
  --tp-size 8 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.8 \
  --max-running-requests 64 --cuda-graph-max-bs 64 --page-size 64 \
  --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule --disable-piecewise-cuda-graph \
  --random-seed 42 --trust-remote-code

# DS (current table-free) — add radix-off + the dev-clone-ABI config (lifted fields accepted)
#   ...same flags as above, PLUS:
#   --disable-radix-cache --enable-double-sparsity \
#   --double-sparsity-config '{"top_k":2048,"page_size":64,"channel_mask_path":"'"$MASK"'","device_buffer_size":4096,"scorer_norm":"off","head_agg":"max","anchor_mode":"off","anchor_budget":0,"enable_lifted_budget_decode":false,"lifted_budget_top_k":0}'
```

**Standard per-arm sequence** (one TP=8 server at a time):

```bash
cd /sgl-workspace/sglang/development/loop13
bash serve.sh dsa            # (backgrounded) boot + wait READY
bash run_gsm8k.sh dsa        # (backgrounded) dense + sparse
bash teardown.sh             # (backgrounded) kill + wait GPU idle
# then repeat with `ds` (and, once built, the reference-selector arm)
bash serve.sh ds && bash probe_ds_active.sh && bash run_gsm8k.sh ds && bash teardown.sh
```

> The **reference (naive) selector** arm (Phase A) does not exist yet — it is what the loop builds.
> `serve.sh` must be extended with a `ref` mode (a config flag / env that selects the naive path)
> once that selector lands, reusing the same guard and the same `run_gsm8k.sh`.

---

## Approach

### Phase A — Establish the accuracy ceiling (the naive-but-correct port)

Build the **most performance-naive, algorithmically-faithful** DS selector for GLM-5.1-FP8 and
measure its GSM8K accuracy. "Naive" = strip *every* perf optimization at once; correctness over speed
(eager, `--disable-cuda-graph` allowed, pure-torch selector acceptable — it can be 100× slow, it only
runs the eval). Reference the traditional algorithm in
`development/past_implementations/study/{00-survey,06-proposed-architecture,07-mvp-proposed-architecture}.md`
(repo A = the paper-faithful `DoubleSparse`).

The reference selector, per decode step:
1. **Reuse the existing offline channel mask** (the `mean|Q·K|` heavy-channel calibration is the
   paper-faithful part and is already done — `/cluster-storage/models/glm51-fp8-channel-mask-loop12.safetensors`).
   Do not re-derive it in Phase A.
2. Materialize a **real per-head `K_label`** by reconstructing K-noPE from the latent in **bf16/fp32**
   (no fp8 dequant-in-register) and gathering the mask channels → `[S, H, r]`.
3. Build `Q_label` (query projected onto the same heavy channels).
4. Score **exactly** in fp32: `Q_label · K_labelᵀ` per head. Test **both** raw-dot and **cosine**
   normalization (the Loop-7 lever — cosine needs the materialized per-head signature this path has,
   which table-free lacks).
5. **Exact full-width `torch.topk`** (no radix approximation, no selector-width bucketing, no bf16
   reduce — reduce scores in fp32).
6. Full attention over the selected indices (the existing FlashMLA sparse decode path is fine — it's
   the *selection*, not the attention, under investigation).

Then measure, on the **same** GSM8K configs already validated this session
(`run_eval --eval-name gsm8k --api completion`, temp 0, **5-shot/200 dense** + **24-shot/150 sparse**,
plus a serial control), three arms on the same server build:
- **DSA** (native indexer) — the target.
- **Naive-DS** (reference selector, raw-dot) and **Naive-DS (cosine)**.
- **Current production DS** (the table-free path) — to confirm the regression reproduces.

**Decision gate (the loop's fork):**
- **Ceiling is BAD** (naive-DS, best of raw/cosine, still far below DSA — e.g. long-context still
  collapses): conclude **H0/H2** — the algorithm (or the mask) doesn't transfer. Recovering accuracy
  is a research problem, not an optimization rollback. Document, optionally probe mask quality
  (recalibrate / vary `label_dim` / per-head vs shared selection), and **stop** — do not chase perf.
- **Ceiling is GOOD** (naive-DS ≈ DSA, especially long-context > 0): conclude **H1** — a perf
  optimization regressed it. Proceed to Phase B.

### Phase B — Bisect the optimization history (only if ceiling is good)

Walk forward from the naive selector toward the current production path, **re-enabling one
optimization at a time**, measuring GSM8K (dense + sparse) at each step, until the accuracy drops.
The first toggle that drops it is the culprit (there may be more than one). Order by suspicion:

1. raw-dot vs cosine scorer (if cosine was the ceiling-maker, this alone may explain it),
2. fp8 absorbed-latent scoring vs materialized bf16 `K_label`,
3. bf16 vs fp32 score-reduce,
4. approximate radix top-k vs exact `torch.topk`,
5. selector-width ladder / W=5120 bucketing,
6. `head_agg` shared-vs-per-head.

Prefer **config/flag toggles** where they already exist (`scorer_norm`, `head_agg`,
`selector_width_buckets`, `score_reduce_dtype`, the `recall_oracle` / `score_capture` diagnostics in
`config.py`) over reverting commits; fall back to `git`-stepping the loop commits
(loop6→loop12, list in `git log -- python/sglang/srt/layers/attention/double_sparsity/`) where no
toggle exists. Reuse `selection_recall_oracle.py` and the score-capture instruments as secondary
diagnostics (recall@2048 and score-flip dumps) to corroborate each GSM8K delta.

---

## Acceptance criteria (rough — gen-plan will formalize as AC-X)

- **AC-1** A reference ("naive") DS selector serves GLM-5.1-FP8 with all perf optimizations off
  (materialized fp32 `K_label`, exact full-width top-k, fp32 reduce, eager OK), DS genuinely active at
  long context (`selected < total`, `dense_fallback == 0`), both raw-dot and cosine selectable.
- **AC-2** GSM8K measured for DSA, naive-DS (raw + cosine), and current production DS on the validated
  configs (5-shot/200 dense, 24-shot/150 sparse, temp 0, completion API), with the production
  regression reproduced (sanity that the harness/build are sound).
- **AC-3** The decision gate is recorded with an explicit numeric threshold for "ceiling good vs bad"
  (e.g. naive-DS long-context within N points of DSA), and the loop branches accordingly.
- **AC-4** (conditional, ceiling good) The optimization that introduces the regression is identified
  by toggle/bisection, with the GSM8K accuracy each optimization costs and the commit(s) responsible.
- **AC-5** A root-cause writeup in `development/loop13/` with the evidence table (per-arm GSM8K +
  recall-oracle corroboration), the verdict (H0/H1/H2), and a recommendation (research vs targeted
  fix) — explicitly *not* a fix.

---

## Constraints (carry forward, do not relitigate)

- **One TP=8 server at a time.** Tear down, wait for all 8 GPUs to return to ~0 MiB before the next
  boot. Track the launched PID; never blanket-kill GPU PIDs or `pkill -f` a parent-matching pattern.
- **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for serving** (breaks custom all-reduce
  at TP=8; calibration-only, separate process).
- **Run the dev clone** (`/sgl-workspace/sglang`, default `import sglang`), **never** the v2 clone, and
  **never** set `PYTHONPATH` to a clone. Always launch via the `development/loop13/` scripts whose
  `_env.sh` guard enforces this (see "Execution harness — clone safety"). Hand-rolled
  `python -m sglang.launch_server` is allowed only after the same guard passes.
- **DS+radix:** the dev clone gates it; use `--disable-radix-cache` (output-neutral at temp 0). Keep
  DSA and DS otherwise byte-identical (model, base, dsa backends, page 64, fp8 KV, seed 42).
- **GSM8K harness is settled:** `python -m sglang.test.run_eval --eval-name gsm8k --api completion`
  (completion path bypasses GLM's thinking template and uses leakage-free shot/question split). Do
  not switch to the chat path.
- Perf is irrelevant this loop — do not optimize the reference selector; slow-but-correct is the point.

---

## References

- **Measured regression + harness:** `development/loop12/gsm8k_evidence/`,
  `development/loop12/RUN_AND_EVALUATE.md`, `development/loop12/V2_PERFORMANCE.md`.
- **Traditional algorithm (the ceiling target):**
  `development/past_implementations/study/00-survey.md` (§1 paper concept, §4 vocabulary, §5 the three
  "double" definitions), `06-proposed-architecture.md`, `07-mvp-proposed-architecture.md`,
  `08-current-system-architecture.md` (as-built; §3 int8 table, §5 over-scan + kernel ABI, §6 recall).
- **Optimization history (the bisection list):** `git log --oneline -- python/sglang/srt/layers/attention/double_sparsity/`
  — Loop 6 `84d3410b9` (int8 table), `ece26eb52` (over-scan), `2715b7382` (tie-break); Loop 7
  `599d7cc99`/`e2674f4f4` (cosine scorer, recall 5%→40%); Loop 8 `4e49d8416`/`43709a761` (GLM-5.1 port
  + calibration); Loop 9 `c877d7fa1` (bf16 reduce), `859c8ee2c` (approx radix top-k); Loop 10
  `6c92240b9`/`9956c240e` (selector-width ladder, W=5120); Loop 11 `776f3e613`→`01e3ff238` (table-free
  absorbed-latent, delete TokenLabelTable).
- **Current selector code:** `selection_kernel.py`, `absorbed_latent.py`, `absorbed_latent_kernel.py`,
  `config.py` (the `scorer_norm`/`head_agg`/`selector_width_buckets` knobs + diagnostics),
  `selection_recall_oracle.py` (recall corroboration).

## Open decisions for the user / gen-plan

1. **How naive is naive?** Purest is a stand-alone pure-torch reference selector (most trustworthy,
   slowest); cheaper is reusing the absorbed path in bf16 with exact top-k + cosine. Recommend the
   pure-torch reference for AC-1 so the ceiling is unimpeachable.
2. **Ceiling threshold.** What long-context GSM8K gap from DSA counts as "algorithm transfers"
   (e.g. within 5 points) vs "doesn't transfer"?
3. **Scope of the fork's BAD branch.** If the ceiling is bad, do we spend this loop probing mask
   quality (recalibrate / `label_dim` / per-head selection), or stop at the verdict and open a
   research loop?
4. **Is per-head selection feasible on MLA at all,** or is shared-across-heads a hard constraint we
   must accept (and therefore part of the ceiling, not a bug)?
