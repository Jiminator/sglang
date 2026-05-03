# sgl-eval migration plan

Scoping doc for moving every accuracy benchmark sglang currently offers
into sgl-eval. Companion to `README.md` (architecture) and `CLAUDE.md`
(editing rules). Not a checklist — a map of the messes plus a strategy
for cleaning them up.

## Why this exists

sglang's accuracy-eval surface today is fragmented across **at least
six** independent harnesses, plus shell-outs to **three** external
evaluation frameworks. The same benchmark (gsm8k, mmlu) can be scored
multiple different ways depending on which CI test you happen to be
looking at. The only unified path that scales today is shelling out to
NeMo-Skills, which means a `uv venv` + `pip install
git+https://github.com/NVIDIA/NeMo-Skills.git` on every CI run
(`accuracy_test_runner.py:_run_nemo_skills_eval` lines 164-221).

sgl-eval's job is to be the lightweight, vendored alternative: one
unified harness, one set of graders (vendored verbatim from NS so
scores match), no minutes-per-CI-run installation.

## Roadmap coverage

This plan addresses sgl-eval's `README.md` roadmap as follows:

| roadmap bullet | covered? | where in this doc |
|---|---|---|
| **1. Replace the accuracy-eval surface in `sgl-project/sglang`.** "sgl-eval aims to be the single client SGLang's CI calls." | ✅ This is the entire purpose of the document. | `## The fragmentation, mapped` enumerates what needs replacing; `## Per-benchmark deep dive` describes every implementation; `## Effort tiers and migration sequencing` lays out the order. |
| **2. More benchmarks within `math` and `multichoice`** (MATH-500, AIME26, MMLU-Pro, GPQA-extended). | ✅ Covered for the sglang-relevant subset. | `## Per-benchmark deep dive` covers MATH (Hendrycks), MMLU-Pro. **Tier 1** sequencing recommends MATH, MATH-500, MMLU-Pro as first-PR targets. AIME26 / GPQA-extended are not in sglang today, so out of scope of *this* migration but trivially register-as-row when needed. |
| **3. New metrics types** — `long_context`, `code`, `instruction_following`, `multimodal`, `agentic`. | ⚠️ Partially covered. | `long_context` (LongBench V2): per-benchmark deep dive. `code` (HumanEval): per-benchmark deep dive + Tier 2 sequencing. `multimodal` (MMMU, MMMU-Pro): per-benchmark deep dive + Tier 2. `instruction_following` (IFEval) and `agentic` (BFCL, Tau-Bench) are not in sglang today and not analyzed here — they're future-state additions, not migrations. |
| **4. More vendor sources** beyond NeMo-Skills (`lm-evaluation-harness`, `lmms-eval`, `openai/simple-evals`). | ✅ Analyzed. | `## The fragmentation, mapped` shows lm-eval-harness and lmms-eval as existing in-tree shell-outs. mgsm migration explicitly proposes `lm-evaluation-harness` as the natural *first* second-vendor-source candidate. mmmu migration proposes `lmms-eval` for basic MMMU. `### The vendor-rule pressure point` consolidates the strict-vs-roadmap-vs-soft decision and recommends "roadmap-as-written." `openai/simple-evals` is the codebase the existing sglang `simple_eval_*.py` files are adapted from, so it's referenced throughout but not proposed as a *new* vendor source — its content is already inlined. |
| **5. LLM-as-judge benchmarks** (Arena-Hard, MTBench). | ❌ Out of scope of this migration. | sglang doesn't ship Arena-Hard or MTBench accuracy eval (the `benchmark/llm_judge/` and `benchmark/mtbench/` legacy scripts are *not really accuracy benchmarks* — they generate judgments rather than score them). Closest in-scope item is the `simple_eval_math.py` GPT-4-turbo judge — flagged in the math section as a methodology change the migration removes. **Genuine LLM-as-judge benchmarks remain a future architectural addition**, as the README says, and need a second judge endpoint + prompt-pair handling. |
| **6. Regression CI infra** (publish per-run metrics to `sgl-eval-data` repo, baselines, fail-on-regression). | ❌ Out of scope of this migration. | Mentioned at the boundary in `### CI threshold migration` (every `*_score_threshold` constant in `test/registered/` is calibrated against a specific sglang harness and will need to shift). The "what to do about the threshold drift" question (Open Question #6) is the on-ramp to building this infra, but the infra itself is downstream of the migration finishing. |

**Reading guide:** if you came to this doc from a specific roadmap
bullet, the "where in this doc" column tells you what to read. If
you're reviewing the doc cold, the rest of the sections proceed
top-down: fragmentation → per-benchmark detail → source coverage →
sequencing → cross-cutting concerns → open questions.

## The fragmentation, mapped

Six in-tree harnesses:

| # | path | shape |
|---|---|---|
| 1 | `python/sglang/test/run_eval.py` + `simple_eval_*.py` | modern, simple-evals-style; the one that's growing |
| 2 | `python/sglang/test/few_shot_gsm8k.py` | deprecated wrapper, but **still imported by ~10 Ascend NPU CI tests** |
| 3 | `python/sglang/test/few_shot_gsm8k_engine.py` | engine-API variant of #2; same status |
| 4 | `python/sglang/eval/llama3_eval.py` | Meta-Llama-3-Instruct-evals dataset, runs mmlu/mmlu_cot/mmlu_pro/gsm8k against Meta's official eval set |
| 5 | `python/sglang/eval/loogle_eval.py` | LooGLE long-context QA, BERTScore-based |
| 6 | `sgl-model-gateway/e2e_test/infra/run_eval.py` + `simple_eval_*.py` | **copy** of #1 for the gateway's own e2e tests (currently MMLU-only) |

Three external-harness shell-outs:

| harness | callsite | what it covers | weight |
|---|---|---|---|
| **NeMo-Skills** (`ns eval`) | `accuracy_test_runner.py:_run_nemo_skills_eval` | mmmu-pro today; conceptually anything | creates uv venv on first call, `pip install git+nemo_skills` (~minutes) |
| **lmms-eval** | `kits/mmmu_vlm_kit.py` | `mmmu_val`; CI default for VLM model tests | `python -m lmms_eval` subprocess; needs `lmms_eval` already installed |
| **lm-eval-harness** | `kits/lm_eval_kit.py` | gsm8k via yaml configs (extensible) | `import lm_eval` in-process |

Plus the legacy `benchmark/<name>/bench_sglang.py` standalone scripts
(out of scope by user direction — flagged at the end).

The rest of this doc walks every benchmark, lists every place it
currently lives, names the divergences, and proposes the migration.

---

## Per-benchmark deep dive

### gsm8k — six implementations, three datasets

This is the worst case. Three distinct upstream data sources, six
distinct in-tree code paths, and four distinct ways the model is
actually queried (chat-completions HTTP, completions HTTP, sglang
runtime HTTP, sglang in-process engine). Each implementation exists
because someone needed something the previous one couldn't do.

| path | dataset source | prompt | grader | callers |
|---|---|---|---|---|
| `simple_eval_gsm8k.py` (modern, `--eval-name gsm8k`) | `openai/grade-school-math` `test.jsonl` (raw GitHub) | 5-shot, `Question: ... \nAnswer: ...`, completion API by default in `kits/eval_accuracy_kit.py:GSM8KMixin` | regex `-?\d+\.?\d*`, last-number, exact match | `kits/eval_accuracy_kit.py:GSM8KMixin`, half a dozen `test/registered/{piecewise_cuda_graph,scheduler,hicache,...}` tests |
| `few_shot_gsm8k.py` (deprecated wrapper) | same | same, sent to sglang `/generate` | regex `\d+` (no negatives, no decimals) — divergent | ~10 `test/registered/ascend/...` NPU tests |
| `few_shot_gsm8k_engine.py` (deprecated, engine-mode) | same | same, sent to in-process `sgl.Engine` | same as #2 | one or two ascend / engine tests |
| `llama3_eval.py` `--task gsm8k` | `meta-llama/Llama-3.1-405B-Instruct-evals` HF | Meta's `input_final_prompts` field | `"The final answer is X"` regex, compare against Meta's `input_correct_responses` | manual; not wired to CI |
| `kits/lm_eval_kit.py` (gsm8k yaml) | lm-eval-harness's own gsm8k task | 5-shot per yaml; `apply_chat_template` configurable | lm-eval `exact_match,strict-match` and `flexible-extract` | `test/registered/{models,amd/accuracy/...}` configs |
| `benchmark/gsm8k/bench_sglang.py` (legacy, out of scope) | grade-school-math **OR** `madrylab/gsm8k-platinum` HF (only this one supports platinum) | sgl.function few-shot, regex `\d+` | exact match | manual |

**Implementation 1: `python/sglang/test/simple_eval_gsm8k.py` — the modern, run_eval-routed path.**
Dispatched by `python -m sglang.test.run_eval --eval-name gsm8k` (`run_eval.py:172`). Loads grade-school-math test.jsonl from raw GitHub; constructs a 5-shot prompt with the simple-evals `Question:/Answer:` template; sends it through `simple_eval_common.py:ChatCompletionSampler` or `CompletionSampler`. Notably, `kits/eval_accuracy_kit.py:GSM8KMixin` (the CI consumer) calls this with `api="completion"`, so under the hood gsm8k actually hits `/v1/completions` and uses the stop tokens `["Question", "Assistant:", "<|separator|>"]` to short-circuit at the next few-shot boundary. Concurrency: Python `multiprocessing.pool.ThreadPool`, bounded by `--num-threads` (default 1024 in CI). Answer extraction regex is `-?\d+\.?\d*` (handles negatives and decimals). Score = exact match on the last extracted number. **This is what most CI tests under `test/registered/{piecewise_cuda_graph,scheduler,hicache,breakable_cuda_graph,...}` use.**

**Implementation 2: `python/sglang/test/few_shot_gsm8k.py` — the deprecated wrapper around sglang's native runtime endpoint.**
Predates `simple_eval_gsm8k.py`. Dispatched as `python -m sglang.test.few_shot_gsm8k`. Same data source, same 5-shot template, but talks to **sglang's own `/generate` endpoint via `RuntimeEndpoint`** (not OpenAI-compatible) using a `@sgl.function` and `run_batch`. Because the endpoint is sglang-native, the prompt is sent as raw text without any chat template — equivalent to running an OpenAI completion against an sglang server. The answer regex is `\d+` (no negatives, no decimals — divergent from #1, will mark "$-5" as 5). Status: the file emits `DeprecationWarning` and tells callers to use #1, but ~10 `test/registered/ascend/...` NPU tests still import it directly because they were written before #1 existed and nobody has moved them. **The behavioral difference vs #1 in practice: this one bypasses the OpenAI-compatible layer and tests the sglang native generate path, which is meaningful if you're actually trying to validate that path. The grader is also strictly less robust.**

**Implementation 3: `python/sglang/test/few_shot_gsm8k_engine.py` — the deprecated wrapper around the in-process engine.**
Same data, same prompt, same regex as #2. Difference: instead of an HTTP server, it instantiates `sgl.Engine(model_path=...)` in-process and calls `engine.async_generate` directly. No server launch, no port, no HTTP. Concurrency via `asyncio.gather`. Used by exactly one CI test (`test/registered/core/test_srt_engine.py:test_5_gsm8k`) and a few Ascend NPU spec-decoding tests that need to drive the engine without a server. Also marked deprecated. **Functional reason this exists separately from #2: lets you accuracy-check a model without standing up a server — useful for engine-only paths and for tests that compare engine vs server behavior.**

**Implementation 4: `python/sglang/eval/llama3_eval.py --task gsm8k` — Meta's official Llama-3 evaluation reproduction.**
Loads `meta-llama/Llama-3.1-405B-Instruct-evals` from HuggingFace (or 70B/8B variants depending on `--model-size`). The prompt is **whatever Meta baked into `input_final_prompts`** in that dataset — it's not a few-shot template, it's the exact byte sequence Meta used to publish their official numbers. The script prepends `<|begin_of_text|>` and sends to `/v1/completions` async. Answer extraction is `re.search(r"The final answer is (.+)\.?")` — matches Meta's prompt template. Scoring compares against the dataset's `input_correct_responses` field (a list of accepted answer strings). **This is not really gsm8k as a benchmark; it's a regression-snapshot tool that reproduces Meta's eval methodology to validate "does my serving stack reproduce Meta's published Llama-3 gsm8k number." Not wired to any CI.**

**Implementation 5: `python/sglang/test/kits/lm_eval_kit.py` + `test/lm_eval_configs/*.yaml` — the lm-evaluation-harness in-process path.**
A test mixin (`LMEvalMixin`) that imports `lm_eval` and calls `lm_eval.simple_evaluate(...)` directly. The benchmark, dataset, prompt, grader, and metrics all come from lm-eval-harness — sglang only configures the `model="local-completions"` backend pointing at `base_url + "/v1/completions"`. Each consumer test ships a YAML (`test/lm_eval_configs/Qwen3.5-397B-A17B.yaml`, etc.) declaring `tasks: [gsm8k]`, `num_fewshot: 5`, and expected `value:` for both `exact_match,strict-match` and `exact_match,flexible-extract` metrics. Currently used by `test/registered/models/test_nvidia_nemotron_3_nano.py` and `test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py`. **Functional difference vs #1: the prompt template, few-shot exemplars, dataset filtering, and the `flexible-extract` metric are all lm-eval's, not simple-evals'. Numbers are calibrated against the broader lm-eval ecosystem's published baselines.**

**Implementation 6: `benchmark/gsm8k/bench_sglang.py` — legacy standalone script (out of scope per user direction).**
Standalone script in `benchmark/gsm8k/`. Talks to sglang via `select_sglang_backend` + `@sgl.function` (similar to #2). Has a `--platinum` flag that switches the dataset from grade-school-math to `madrylab/gsm8k-platinum` (a relabeled version with corrected mislabels). **This is the only path that supports gsm8k-platinum.** Same `\d+` regex as #2. Status: legacy, not in CI; user explicitly said legacy is out of scope.

**NS coverage (already vendored in sgl-eval):**
`_vendored/.../dataset/gsm8k/{__init__,prepare}.py`. Loads grade-school-math test.jsonl, normalizes to `{problem, expected_answer, reference_solution}`, applies a hand-curated `fixes` dict for known-mislabeled rows, hands off to `MathEvaluator` (extracts `\boxed{}` content, scores via `math_equal`). Already in `_TABLE`.

**The four distinct query paths matter for migration**: any sgl-eval client calls *only* OpenAI-compatible endpoints (chat completions or completions). #1 and #4 already fit that model. #2 and #3 use the sglang-native generate endpoint and the in-process engine respectively — sgl-eval has no equivalent and would not directly replace those. The Ascend NPU tests using #2 would either need to be moved to call sgl-eval's `gsm8k` (which means launching an OpenAI-compatible server) or kept on `few_shot_gsm8k.py` as native-runtime regression tests.

**Migration:**
1. The modern `simple_eval_gsm8k.py` path = sgl-eval's `gsm8k`. Done. Score parity needs an A/B run because:
   - sglang uses 5-shot completion-API with `Question:/Answer:` template; NS uses 0-shot chat with `\boxed{}` answer.
   - sglang's last-number regex vs NS's symbolic `math_equal` will diverge on edge cases (model says "$42" vs "42").
2. **Deprecation path**: `few_shot_gsm8k.py` (#2) and `few_shot_gsm8k_engine.py` (#3) are already marked deprecated. Their ~10 Ascend NPU callers need to be migrated to `run_eval` (or to sgl-eval directly). Sequencing matters: those NPU tests are owned by a different team, so this is a coordination cost not a code cost.
3. **`llama3_eval.py`**: a Meta-specific path that uses Meta's pinned eval set and Meta's "is this answer in `input_correct_responses`" comparison. **Not equivalent to standard gsm8k.** Either keep it as a separate sgl-eval benchmark (`gsm8k-meta` or similar), drop it, or treat it as a regression-snapshot tool rather than an accuracy eval. Ask owner.
4. **`kits/lm_eval_kit.py`**: this is the only path that exposes lm-eval's `flexible-extract` metric, which some downstream tests depend on for thresholding. Migrating away from it requires confirming sgl-eval's grader matches what those configs expect, or shipping a `flexible-extract`-equivalent metric in sgl-eval.
5. **gsm8k-platinum**: `madrylab/gsm8k-platinum` is a relabeled variant of the GSM8K test set (corrected mislabels). Only the legacy script supports it. NS covers this implicitly via its `fixes` dict, but if exact platinum parity matters, register a second sgl-eval entry that loads the platinum HF dataset.

---

### mmlu — five implementations, three datasets

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_mmlu.py` (`--eval-name mmlu`) | `openaipublic.blob.core.windows.net/simple-evals/mmlu.csv` (OpenAI's flat CSV) | "Answer the following multiple choice question..." chat-style | regex `(?i)Answer\s*:\s*([A-D])` |
| `llama3_eval.py` `--task mmlu` / `mmlu_cot` | `meta-llama/Llama-3.1-405B-Instruct-evals` | Meta's prompt + `<|begin_of_text|>` prefix | `mmlu`: 1-token completion; `mmlu_cot`: regex `The (best|correct) answer is X` |
| `kits/lm_eval_kit.py` (potential, no yaml today) | lm-eval-harness's mmlu | configurable | lm-eval's match metrics |
| `benchmark/mmlu/bench_sglang.py` (legacy) | `https://people.eecs.berkeley.edu/~hendrycks/data.tar` (Hendrycks tarball, 60 subjects) | k-shot dev set, completion API, `max_new_tokens=1` (loglik-style first-token answer) | `pred[:1] == label` |
| `sgl-model-gateway/e2e_test/infra/simple_eval_mmlu.py` | `openaipublic.blob.core.windows.net/simple-evals/mmlu.csv` | same as #1 | same as #1 |

**Implementation 1: `python/sglang/test/simple_eval_mmlu.py` — modern, run_eval-routed.**
Dispatched by `python -m sglang.test.run_eval --eval-name mmlu`. Loads OpenAI's flat 14k-row CSV (`openaipublic.blob.core.windows.net/simple-evals/mmlu.csv`). For each row, builds a chat-style prompt: `"Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"` (`simple_eval_common.py:244-253`). Sends through `ChatCompletionSampler` to `/v1/chat/completions`. Answer extraction: `re.search(r"(?i)Answer\s*:\s*([A-D])")`. Per-row score: 1 if extracted letter equals the gold letter else 0. Per-subject metrics tracked via `subject2category` dict (stem / humanities / social_sciences / other). **This is what `kits/eval_accuracy_kit.py:MMLUMixin` calls, which is the CI default for MMLU thresholds.**

**Implementation 2: `python/sglang/eval/llama3_eval.py --task mmlu | mmlu_cot` — Meta's pinned eval-set reproduction.**
Same script as gsm8k #4 above, different `--task`. Loads `meta-llama/Llama-3.1-405B-Instruct-evals` for `--task mmlu` (uses Meta's official prompts and `input_correct_responses` answer keys) or `--task mmlu_cot` (uses a chain-of-thought variant from the same dataset). The `mmlu` task runs with `max_tokens=1` since Meta's prompt is loglik-style (the model outputs a single letter token). The `mmlu_cot` task runs with `max_tokens=1024` and extracts via `r"The best answer is (.+)\.?"` (with three regex fallbacks — see `llama3_eval.py:150-169`). **This is the only sglang path that supports CoT-style MMLU. Not wired to CI; manual tool for reproducing Meta's published numbers.**

**Implementation 3: `python/sglang/test/kits/lm_eval_kit.py` (potential, no current MMLU yaml).**
Same `LMEvalMixin` mechanism as gsm8k #5. No `mmlu` yaml ships in `test/lm_eval_configs/` today, but the mechanism is generic — adding `tasks: [mmlu]` to a yaml would make it active. Listed for completeness because if a future test needs lm-eval's MMLU numbers, this is how it would land.

**Implementation 4: `benchmark/mmlu/bench_sglang.py` — legacy standalone script (out of scope).**
Downloads the Hendrycks tarball (`https://people.eecs.berkeley.edu/~hendrycks/data.tar`) which contains the 60-subject canonical MMLU CSVs. For each subject, loads the dev split as k-shot examples (default k=5, automatically reduced to fit a 1536-token limit measured with `tiktoken.encoding_for_model("gpt-3.5-turbo")`), then evaluates the test split. Uses `@sgl.function` with `max_new_tokens=1` — i.e. the model produces exactly one letter token, no chain-of-thought, no "Answer:" prefix. Score = `pred[:1] == label`. Per-subject accuracy printed and weighted-averaged. **Methodologically very different from #1: this is loglik-style, while #1 is chat-style with reasoning. Same questions, different scores.**

**Implementation 5: `sgl-model-gateway/e2e_test/infra/simple_eval_mmlu.py` — gateway's own copy of #1.**
A literal copy (with light edits for the gateway's logging conventions) of `python/sglang/test/simple_eval_mmlu.py`. Lives in the `sgl-model-gateway` subproject because that subproject's e2e tests can't import from `sglang.test` cleanly. Loads the same simple-evals MMLU CSV, uses the same `ChatCompletionSampler`. Functionally equivalent to #1; exists only because of the import-path constraint. Used by `sgl-model-gateway/e2e_test/router/test_mmlu.py` and `test_pd_mmlu.py`. **Once sgl-eval is pip-installable, this duplicate disappears entirely — the gateway just imports `sgl_eval`.**

NS coverage (already vendored): `dataset/mmlu/__init__.py` says `METRICS_TYPE = "multichoice"`, prompt `eval/aai/mcq-4choices-boxed`. `prepare.py` downloads from the same Hendrycks tarball, processes all 60 subjects. Already in `_TABLE`.

**Migration:**
1. `simple_eval_mmlu.py` ≅ sgl-eval `mmlu`. Score-parity check: sgl-eval uses NS's full Hendrycks 60-subject set vs sglang's flat 14k-row CSV — different sample populations. Confirm whether the unified value should be the Hendrycks set (more standard) or the simple-evals CSV (what sglang's CI is calibrated to).
2. The legacy `bench_sglang.py` is loglik-style (`max_new_tokens=1`); fundamentally different methodology from chat-eval. Not in scope, but worth noting it exists because some old comparisons still cite its numbers.
3. `sgl-model-gateway`'s copy is a literal duplicate of `simple_eval_mmlu.py` adapted for that subproject. After sgl-eval lands as a pip install, the gateway should `import sgl_eval` instead of carrying its own copy. Mechanical cleanup.
4. `mmlu_cot` (Meta's chain-of-thought MMLU variant) is only in `llama3_eval.py`. NS does not have `mmlu_cot`. Decision: register sgl-eval `mmlu` (NS-style) and `mmlu_cot` (separate, would need a vendored or sgl-eval-authored prompt config), or drop `mmlu_cot`.

---

### gpqa — two implementations, two seed regimes

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_gpqa.py` (`--eval-name gpqa`) | `openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv` | multichoice, 4-way permutation seeded with `random.Random(0)` per call | regex `(?i)Answer\s*:\s*([A-D])` |
| `accuracy_test_runner.py` (registered tests) | same as above (delegates to run_eval) | same | same |

**Implementation 1: `python/sglang/test/simple_eval_gpqa.py` — modern, run_eval-routed.**
Loads `simple-evals/gpqa_diamond.csv` (the "diamond" subset, 198 PhD-level questions). For each row, packs the correct answer plus three incorrect answers into a 4-way list, **permutes them with `random.Random(0).sample(range(4), 4)` per row**, then renders with the same shared `QUERY_TEMPLATE_MULTICHOICE` as MMLU. Answer extraction is the same `(?i)Answer\s*:\s*([A-D])` regex. Supports `n_repeats` for variance estimation. Score = 1 if extracted letter matches the correct-answer's permuted slot. **The seed-0 permutation is the canonical simple-evals behavior; shifting to a different seed changes scores measurably because some models are positionally biased.**

**Implementation 2: `python/sglang/test/accuracy_test_runner.py` (`run_accuracy_test` with `dataset="gpqa"`) — same path, different dispatch.**
Not a separate implementation — same `simple_eval_gpqa.py` underneath. Listed only to flag the dispatch surface: `accuracy_test_runner.run_accuracy_test(model, params, ...)` is the function called by big-model CI tests like `test/registered/8-gpu-models/test_deepseek_v32.py`. It launches the server, calls `run_eval(args)` with `eval_name="gpqa"`, and validates `score >= baseline_accuracy`. Uses `gpt_oss_common.py:_run_one_eval` patterns (`temperature=0.1`, `max_tokens=4096`, `reasoning_effort` passthrough) when called from `gpt_oss_common`. **Functionally identical to #1; called out because the migration needs to update both the direct `run_eval` callers and the `accuracy_test_runner` callers.**

The only sglang gpqa entry. Used by `gpt_oss_common.py:_run_one_eval` (line 124) with 198 examples, GPT-OSS-style reasoning_effort.

NS coverage (vendored): `dataset/gpqa/__init__.py` `METRICS_TYPE = "multichoice"`, prompt `eval/aai/mcq-4choices-boxed`. `prepare.py` loads `Idavidrein/gpqa` HF (the canonical source, not the simple-evals CSV), with `random_seed` kwarg for choice permutation. sgl-eval registers it with `random_seed=42`. Already in `_TABLE`.

**Migration:**
1. **Score-parity risk: choice-permutation seed.** sglang permutes per-call with `random.Random(0)`; sgl-eval (via NS prepare) permutes once at dataset-prep time with seed 42. Different orderings → different "A/B/C/D" labels for the same question → numerically different scores. Need an A/B run before declaring drop-in.
2. **Dataset-source risk.** The simple-evals CSV is a frozen snapshot; the HF `Idavidrein/gpqa` dataset can be updated. The 198 questions in `gpqa_diamond` *should* be identical, but verify counts.
3. `n_repeats` is supported on both sides (sglang via `--repeat`, sgl-eval via registry `default_n_repeats=8`). sgl-eval's pass@k / majority@k aggregation is richer than sglang's mean.

---

### aime25 — two implementations

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_aime25.py` (`--eval-name aime25`) | `opencompass/AIME2025` HF (`AIME2025-I` + `AIME2025-II`, 30 problems) | "Solve the following AIME problem... Answer: $ANSWER (integer 000-999)" | regex extract `Answer:`, `int(float(s))`, range-check 0-999, exact match |
| sgl-eval `aime25` (already registered) | bundled JSONL in `_vendored/.../dataset/aime25/test.txt` (30 problems) | `prompts/math.yaml` with `\boxed{}` | NS `extract_answer` (boxed-first) + `math_equal` |

**Implementation 1: `python/sglang/test/simple_eval_aime25.py` — only sglang path.**
Loads `opencompass/AIME2025` from HuggingFace, concatenating the `AIME2025-I` and `AIME2025-II` configs (15 + 15 = 30 problems). Renders each through the AIME-specific template (`simple_eval_aime25.py:25-33`) which tells the model: "Note: AIME answers are always integers from 000 to 999... Remember to put your answer on its own line after 'Answer:', and express your answer as an integer from 000 to 999." Sends through `ChatCompletionSampler`. Answer extraction: `re.search(r"(?i)Answer\s*:\s*([^\n]+)")`. Both the extracted answer and the gold answer are normalized through `normalize_aime_answer` (`int(float(s))`, range-check 0-999, fall back to raw string). Score = 1 if normalized strings match. Reports `chars` metric (length of response) per row. **Single-shot at default temperature=0.0; AIME is a 30-problem benchmark and people normally repeat-sample to get stable numbers, but this implementation doesn't bake that in — caller must use `--repeat`.**

**Implementation 2 (already in sgl-eval): registry entry `aime25`.**
For comparison: sgl-eval ships AIME25 as bundled JSONL in `_vendored/.../dataset/aime25/test.txt` (30 problems with `id`, `problem`, `expected_answer`, `reference_solution` fields). Prompt is `_vendored/prompts/math.yaml`: `"Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.\n\n{problem}"`. Grader is NS's `extract_answer` (looks for `\boxed{}` first, falls back to regex) + `math_equal` (symbolic comparison via `math_verify`). Defaults: `temperature=0.0`, `default_n_repeats=16`, `thinking=True`. **Same 30 problems but completely different prompt and grader — score parity needs verification.**

**Migration:**
1. Same 30 problems, but **prompts and graders differ.** sglang asks for `Answer: $ANSWER`; sgl-eval asks for `\boxed{}`. Models that boxed but didn't write "Answer:" will score differently.
2. **Default n_repeats:** sgl-eval has `default_n_repeats=16` and `thinking=True`. sglang's run is single-shot at default `temperature=0.0`. The "correct" comparison is an open question — sgl-eval's defaults match the AIME25 leaderboard convention better, but it's a methodology change.
3. **aime24:** in sgl-eval's registry, not in sglang's `run_eval`. No-op for migration, but flag if anyone wants it via sglang.

---

### math (Hendrycks MATH) — one implementation, judge-based

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_math.py` (`--eval-name math`) | `openaipublic.blob.core.windows.net/simple-evals/math_test.csv` | "Solve the following math problem... Answer: $ANSWER" | LLM-as-judge: `gpt-4-turbo` rates equivalence of extracted vs gold |

**Implementation 1: `python/sglang/test/simple_eval_math.py` — only sglang path.**
Loads OpenAI's frozen MATH CSV (`simple-evals/math_test.csv`, 5000 rows from the original Hendrycks et al. test split). Prompt template (`simple_eval_math.py:26-32`): "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes)... you do not need to use a `\boxed` command." Sends through `ChatCompletionSampler`. Answer extraction: `re.search(r"(?i)Answer\s*:\s*([^\n]+)")` — captures everything after "Answer:" on that line. **Scoring is the unique part:** instead of symbolic comparison, it spawns a *second* sampler — `equality_checker = ChatCompletionSampler(model="gpt-4-turbo")` (hardcoded in `run_eval.py:116`) — and asks GPT-4-turbo whether the extracted answer is equivalent to the gold answer using the `EQUALITY_TEMPLATE` (a few-shot prompt full of trivial-simplification examples, `simple_eval_common.py:259-317`). Score = 1 if the judge says "Yes". **This means running `--eval-name math` requires a working OpenAI API key with paid GPT-4-turbo access.** It's a hidden CI dependency that doesn't appear in any other simple_eval. Not currently wired to any registered CI test, presumably for that reason; manual-only.

NS coverage (NOT vendored yet, but exists upstream):
- `nemo_skills/dataset/hendrycks_math/` → `METRICS_TYPE = "math"`, prompt `generic/math`. Same shape as gsm8k from sgl-eval's perspective — pure symbolic grading, no judge.
- Also `nemo_skills/dataset/math-500/` (the GPT-4 verified subset, often used as the canonical post-2024 MATH benchmark) and `nemo_skills/dataset/minerva_math/`.

**Migration (Tier 1 — easiest in this whole doc):**
1. Vendor `dataset/hendrycks_math/{__init__,prepare}.py` via SOURCES.yaml. Add one row to `_TABLE`. Done.
2. **Methodology change to call out:** sglang relies on a paid LLM judge; sgl-eval would use NS's symbolic grader. Numbers will differ. The judge-based eval is what OpenAI's simple-evals does because it predates good open-source math graders. NS's `math_equal` (built on `math_verify` + `latex2sympy2_extended`) is the modern replacement. **The migration removes the `OPENAI_API_KEY=real` CI dependency** — net win even before considering speed/cost.
3. Also recommend adding `math-500` as a separate registered eval since it's the more-cited 2024+ benchmark.

---

### mgsm / mgsm_en — one implementation, NOT in NS

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_mgsm.py` (`--eval-name mgsm` / `mgsm_en`) | `openaipublic.blob.core.windows.net/simple-evals/mgsm_{lang}.tsv` for 11 languages | per-language instruction template (en, bn, de, es, fr, ja, ru, sw, te, th, zh) with language-localized `Answer:`/`答案:`/etc. | regex extract last number after localized "Answer:" prefix, decimal-stripped exact match |

**Implementation 1: `python/sglang/test/simple_eval_mgsm.py` — only sglang path.**
Loads MGSM TSV files from OpenAI's simple-evals bucket, one per language (`mgsm_bn.tsv` through `mgsm_zh.tsv`). Per-language constants in the file: `LANG_TO_FPATH` (URL per language), `LANG_TO_INSTRUCTIONS` (the instruction template translated to each of 11 languages — Bengali, German, Spanish, French, Japanese, Russian, Swahili, Telugu, Thai, Chinese, English), `LANG_TO_ANSWER_PREFIX` (e.g. `"Answer"` for `en`, `"答案"` for `zh`, `"उत्तर"` for `bn`). Each row evaluated by: render the language's instruction with the question, send to `ChatCompletionSampler`, extract `re.findall(r"\d+\.?\d*", text_after_prefix)`, take the last number with trailing decimal stripped, exact-match against gold. Also reports per-language metrics and `group_latin`/`group_non_latin` aggregates (rows scored both into their language bucket and into a Latin/non-Latin script bucket). The `--eval-name mgsm_en` variant restricts to English only — invoked by `kits/eval_accuracy_kit.py:MGSMEnMixin` at threshold 0.835. **Used by `test/registered/eval/test_eval_accuracy_large.py`, `test/registered/breakable_cuda_graph/test_breakable_cuda_graph.py`, `test/registered/hicache/test_hicache_variants.py`, the deepseek/glm/mistral 8-gpu tests, and several MLA tests. ~6 active CI consumers, all with calibrated thresholds.**

NS coverage: **NS does not have mgsm.** No `nemo_skills/dataset/mgsm/`.

**Migration (this is the one that doesn't fit the vendor-from-NS pattern):**
- **Option A: Author native sgl-eval support.** Write `evals/mgsm.py` (data loader + per-language prompt + grader) as the first non-NS-sourced benchmark. Violates the "anything score-deciding is vendored" rule in `CLAUDE.md` — would be the first SE-authored grader.
- **Option B: Add a second vendor source.** lm-evaluation-harness has mgsm. The roadmap (`README.md` line 99-101) already anticipates `lm-evaluation-harness` as a future second vendor source. mgsm is the natural first benchmark to land that pattern.
- **Option C: Punt.** Drop mgsm; note that 4+ CI tests currently depend on `mgsm_en` thresholds and would need new benchmarks.

Recommendation: **Option B**, but it's a significant architectural addition (second `_vendored/<source>/` directory + parallel SOURCES.yaml + audit script update). Not first-week work.

---

### humaneval — one implementation, sandbox-required

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_humaneval.py` (`--eval-name humaneval`) | `human_eval.data.read_problems()` (the openai/human-eval pip package) | "Read the function signature and docstring, fully implement..." | extract ```python ... ``` block, **execute against test cases** via `human_eval.execution.check_correctness` |

**Implementation 1: `python/sglang/test/simple_eval_humaneval.py` — only sglang path.**
Imports the `human_eval` pip package (the openai/human-eval reference implementation). Calls `human_eval.data.read_problems()` to load all 164 HumanEval problems (each is `{task_id, prompt, entry_point, canonical_solution, test}`). For each problem, prepends the instruction `"Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"` to the problem prompt, sends to `ChatCompletionSampler`, samples it `num_samples_per_task` times (default 5). For each sample, extracts a ```python ... ``` code block via regex (falls back to the raw text), then strips the function signature so what remains is the function body. **Then executes the model's code against the problem's hidden test cases** using `human_eval.execution.check_correctness` — which spawns a subprocess with a 3-second timeout per check and returns a pass/fail. Aggregates with `human_eval.evaluation.estimate_pass_at_k` for k in `[1, 2, 5]`. The score per problem is `correct_samples / total_samples`. **Critically, this is the only implementation in the entire sglang accuracy surface that executes model output as code.** Used by `kits/eval_accuracy_kit.py:HumanEvalMixin` (thresholds 0.64 CUDA / 0.60 AMD) and `test/registered/eval/test_eval_accuracy_large.py`.

NS coverage (NOT vendored yet, exists upstream):
- `nemo_skills/dataset/human-eval/` → `METRICS_TYPE = "evalplus"` (uses the EvalPlus framework, which includes HumanEval+ extended tests).
- Prompt config `generic/codegen`.

**Migration (Tier 2 — needs new runner type):**
1. sgl-eval has no `code` runner today; both `_math.py` and `_multichoice.py` assume the grader is pure-Python and stateless. HumanEval needs to **execute model output as code** with a timeout.
2. The execution sandbox is a real architectural addition. Three options:
   - Vendor NS's evalplus integration (which itself wraps the EvalPlus pip package); SE adds a thin runner that calls it.
   - Fall back to the openai/human-eval package directly (matches sglang's current code).
   - Build a sandboxed exec runner in SE (overkill).
3. **Security caveat for the design discussion:** any "execute model output" runner needs sandboxing. NS's evalplus wraps it; SE's runner needs to inherit that posture, not roll its own.
4. Effort: medium. New runner kind (`code`), new metrics flatten (`pass@1`, `pass@k` with `k` dependent on `n_repeats`), new vendoring section.

---

### longbench_v2 — one implementation, tokenizer-coupled

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_longbench_v2.py` (`--eval-name longbench_v2`) | `THUDM/LongBench-v2` HF (or local CSV/JSON path), `train` split | "Please read the following text... The correct answer is (X)" official template | regex `The correct answer is \(([A-D])\)` (with two fallbacks) |

**Implementation 1: `python/sglang/test/simple_eval_longbench_v2.py` — only sglang path.**
Loads `THUDM/LongBench-v2` from HuggingFace by default, `train` split (or a local CSV/JSON/JSONL path passed via `--dataset-path`). Each row has a long `context` (up to ~128k tokens), a `question`, four choices (`A`/`B`/`C`/`D`), and an `answer` letter. Renders with the **official LongBench-v2 template** (which differs from MMLU/GPQA): `"Please read the following text and answer the question below.\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) ...\n(B) ...\n(C) ...\n(D) ...\n\nFormat your response as follows: \"The correct answer is (insert answer here)\"."`. Sends through `ChatCompletionSampler`. Answer extraction prefers `re.search(r"The correct answer is \(([A-D])\)")`, falls back to `r"The correct answer is ([A-D])"`, then to the standard MMLU `(?i)Answer\s*:\s*([A-D])` pattern, then to `r"answer\s+is\s*\(?([A-D])\)?"`. **Two unique knobs not in any other simple_eval:** (1) `--max-context-length` / `--min-context-length` filters rows by token count, requiring `AutoTokenizer.from_pretrained(model, trust_remote_code=True)` — meaning *the eval itself* needs to load the model under test's tokenizer; (2) per-category metrics across the six task categories (`single_document_qa`, `multi_document_qa`, `long_in_context_learning`, `long_dialogue_history`, `code_repo_understanding`, `long_structured_data`) plus `difficulty_easy`/`difficulty_hard` buckets. Used by `python/sglang/test/longbench_v2/test_longbench_v2_eval.py` and the validation scripts in that directory.

NS coverage (NOT vendored yet, exists upstream):
- `nemo_skills/dataset/longbench-v2/` → `METRICS_TYPE = "multichoice"`. Prompt config `eval/longbench/default`.

**Migration (Tier 1 — fits existing multichoice runner):**
1. Vendor `dataset/longbench-v2/` and the prompt yaml. Add registry row. The existing `_multichoice.py` runner consumes it as-is.
2. **Open question: tokenizer-based context-length filter.** sgl-eval doesn't have a benchmark that pulls a tokenizer from the model under test. Two paths:
   - Keep filtering as a CLI knob, do it in the loader after fetching the dataset (eats RAM but avoids tokenizer dependency).
   - Drop the filter; users sample subsets via `--num-examples`.
   I'd default to dropping it; the filter is a sglang-specific feature that's not in the NS dataset spec. Worth confirming the registered tests don't depend on it.
3. Per-category metrics also need to surface in sgl-eval's `format_summary`. Today the multichoice runner aggregates flat `pass@1`. Need `metrics/longbench_v2.py`-level breakdown, or accept that sgl-eval's headline number is overall and per-category is dropped.

---

### mmmu — four implementations, all VLM, none agree

This is the messiest after gsm8k.

| path | dataset source | prompt | grader |
|---|---|---|---|
| `simple_eval_mmmu_vlm.py` (`--eval-name mmmu`) | `MMMU/MMMU` HF, validation split, all subjects, top-100 by sorted id | own prompt format, `<image>`-tagged inline split, OpenAI image_url | own `_parse_multi_choice_response` (different from #2 below) + `_parse_open_response` |
| `benchmark/mmmu/bench_sglang.py` + `eval_utils.py` (legacy) | `MMMU/MMMU` HF, validation, all subjects, full set | own prompt yaml, async OpenAI image_url | own `parse_multi_choice_response` (different parser despite similar name) |
| `kits/mmmu_vlm_kit.py` (CI default for VLM tests) | shells out to `lmms-eval`'s `mmmu_val` task | lmms-eval's prompt | lmms-eval's grader (`mmmu_acc,none` from the JSON output) |
| `accuracy_test_runner.py:_run_nemo_skills_eval` (`mmmu-pro` only) | shells out to `nemo_skills.pipeline.eval`, prepares its own data | NS's `vlm/mmmu-pro.yaml` | NS's multichoice evaluator |

**Implementation 1: `python/sglang/test/simple_eval_mmmu_vlm.py` — modern, run_eval-routed.**
Dispatched by `--eval-name mmmu`. Loads each of the 30 MMMU subjects (across 6 domains like Art and Design, Business, Science, ...) from `MMMU/MMMU` HF, `validation` split, concatenates them, and **deterministically picks the top 100 rows by sorted `id`** (so different runs see the same 100 questions). For each row, base64-encodes the image into a `data:image/png;base64,...` URI, builds a prompt that's either MCQ (`question` + lettered options + "Answer the following multiple-choice question. The last line of your response should be of the following format: 'Answer: $LETTER' ...") or open ("question + Answer: "). Splits the prompt around any `<...>` image tag and constructs an OpenAI `[{"type":"text"}, {"type":"image_url"}, {"type":"text"}]` content block. Sends through `ChatCompletionSampler`. Scoring branches: MCQ rows go through `_parse_multi_choice_response` (looks for "Answer: X" with various decorations, then `(X)`, then bare `X`, then matches answer text), open rows go through `_parse_open_response` (extracts numbers and key-phrase candidates) + `_eval_open` (substring/numeric match). Reports per-subject accuracy plus "Overall-Art and Design"/"Overall-Business"/etc. domain rollups. **The only run_eval path that handles images.**

**Implementation 2: `benchmark/mmmu/bench_sglang.py` + `eval_utils.py` + `data_utils.py` — legacy standalone script.**
Out of scope per user direction, but described here because it has its own MMMU parsers that **predate and partially conflict with** #1's. Loads all 30 MMMU subjects in parallel via `ThreadPoolExecutor` against HF, processes images (writes them to `~/.cache/mmmu/images/`), constructs a prompt via `prompt_format.yaml` (a hydra-style config file in the same directory). Sends to the model with `openai.AsyncOpenAI` + `asyncio.Semaphore(concurrency)`. Has its own `parse_multi_choice_response` (in `eval_utils.py`, similar but not identical to #1's — different regex ordering, different fallback for ambiguous candidates) and its own `parse_open_response`. Score aggregation also rolls up to domain-level. **Same dataset, different prompts/parsers, different score.**

**Implementation 3: `python/sglang/test/kits/mmmu_vlm_kit.py` — CI default, lmms-eval subprocess.**
The actual MMMU path that VLM model tests in CI use. `MMMUMixin` and `MMMUMultiModelTestBase` shell out to `lmms-eval` via `subprocess.run(["python3", "-m", "lmms_eval", "--model", "openai_compatible", "--model_args", f'model_version="{model}",tp=1', "--tasks", "mmmu_val", "--batch_size", "64", ...])`. The dataset (`mmmu_val` task in lmms-eval's task registry), prompt, image handling, and grader all come from lmms-eval. After the subprocess completes, parses the JSON output for `result["results"]["mmmu_val"]["mmmu_acc,none"]` as the score. Has automatic retry logic if the MMMU parquet cache is corrupted (`_is_mmmu_parquet_corruption` / `_cleanup_mmmu_dataset_cache`). **Used by ~25 `test/registered/{vlm,ascend/vlm_models}` model tests** — this is the CI default for any VLM model test that wants an MMMU score. Uses neither the simple_eval nor the legacy benchmark path. Adds a hard dependency on `lmms_eval` already being importable in the test env, plus the lmms-eval HF dataset cache.

**Implementation 4: `python/sglang/test/accuracy_test_runner.py:_run_nemo_skills_eval` — for `mmmu-pro`, NS subprocess.**
The heavyweight one. When `AccuracyTestParams(dataset="mmmu-pro")` is passed to `run_accuracy_test`, the code (`accuracy_test_runner.py:250-444`) creates a temporary uv venv (`uv venv` + `uv pip install git+https://github.com/NVIDIA/NeMo-Skills.git`, ~30s on a warm cache), runs `nemo_skills.dataset.prepare mmmu-pro` to download VLM data (~minutes, includes large image archives), then runs `python -m nemo_skills.pipeline.eval eval --benchmarks=mmmu-pro:1 --server_type=sglang --server_address=... ++prompt_config=vlm/mmmu-pro ++max_concurrent_requests=512 ++max_samples=500` against the launched sglang server. Parses the score from output regex (`(?:accuracy|score)[:\s]+([0-9.]+)`), falls through three fallbacks if the regex misses: hunt JSON files in `eval-results/`, last-resort manually re-score JSONL by extracting `Answer:` letters. **This is the path the user named as "very heavyweight" — it's the canonical example of why sgl-eval needs to vendor NS instead of shelling out to it.**

NS coverage (partial):
- `nemo_skills/dataset/mmmu-pro/` exists upstream — `METRICS_TYPE = "multichoice"`, prompt `vlm/mmmu-pro`.
- `nemo_skills/dataset/mmmu/` does NOT exist. NS only has the Pro variant.

**Migration (Tier 3 — biggest architectural change in this doc):**
1. **VLM sampler.** sgl-eval's `ChatCompletionSampler` only handles text. MMMU needs `[{"type": "text", ...}, {"type": "image_url", "image_url": {"url": ...}}, ...]` content blocks. This is a real change to `sampler.py`.
2. **Image-bearing `Example`.** `_predictions.py` builds prediction dicts from `Example.inputs` assuming text. Needs an image field.
3. **Two benchmarks, not one:**
   - `mmmu`: NS doesn't have it → either author SE-side (violates vendor rule) or vendor from lmms-eval (third vendor source) or skip in favor of mmmu-pro.
   - `mmmu-pro`: NS has it → vendor + register, once VLM sampler exists.
4. **Decision point: kill `kits/mmmu_vlm_kit.py`'s lmms-eval shell-out.** Once sgl-eval supports VLM, the CI default for VLM model tests can replace the lmms-eval subprocess with a `sgl-eval run mmmu-pro` call. **Score parity required first** — the vendored NS scorer should match what `mmmu_acc,none` reports, but I haven't verified.
5. **Decision point: kill `accuracy_test_runner.py:_run_nemo_skills_eval`** for mmmu-pro. This is the biggest immediate value-add of the migration: the current code creates a uv venv, pip-installs nemo_skills from main, runs `ns eval` as a subprocess, and parses its output (and falls back to manually re-scoring `*.jsonl` if parsing fails — see lines 388-425). Replacing it with an in-process sgl-eval call eliminates minutes of CI startup and a fragile shell pipeline.
6. Effort: large. New runner kind (`multimodal`), sampler refactor, two new vendored datasets, possibly a second vendor source for the basic `mmmu`.

---

### mmlu_pro — only via shell-outs

Not in `simple_eval_*`. Surfaced through:

| path | dataset source | prompt | grader |
|---|---|---|---|
| `llama3_eval.py` `--task mmlu_pro` | `meta-llama/Llama-3.1-405B-Instruct-evals` (Meta's pinned mmlu_pro set) | Meta's `input_final_prompts` field, `<|begin_of_text|>` prefix | regex `The (best|correct) answer is X` (mmlu_cot extractor) |
| `accuracy_test_runner.py:_run_nemo_skills_eval` (potential) | NS's `mmlu-pro` dataset | NS's `eval/aai/mcq-10choices` prompt (10 choices, not 4) | NS multichoice evaluator |

**Implementation 1: `python/sglang/eval/llama3_eval.py --task mmlu_pro` — Meta's pinned eval-set reproduction.**
Reuses the same `llama3_eval.py` machinery as gsm8k #4 and mmlu #2. Loads `meta-llama/Llama-3.1-405B-Instruct-evals` with the `evals__mmlu_pro__details` config. Prompt is whatever Meta encoded in `input_final_prompts`. `max_tokens=2048` (longer than mmlu's 1, because mmlu_pro is CoT-style). Answer extraction reuses `get_mmlu_cot_answer` — the same `r"The (best|correct) answer is (.+)\.?"` regex with three fallbacks. Score = is the extracted answer in `input_correct_responses`. **Same caveat as the other `llama3_eval.py` paths: it's a regression-snapshot tool reproducing Meta's published number, not a generic mmlu_pro benchmark. Not in CI.**

**Implementation 2: `python/sglang/test/accuracy_test_runner.py` — *potential* NS shell-out (not currently wired).**
The `_run_nemo_skills_eval` path is keyed off `dataset in ("mmmu-pro", "mmmu_pro")` today. Adding `"mmlu-pro"` would route through the same uv-venv + `pip install nemo_skills` machinery. Listed because it's the natural escape hatch the codebase is set up for — not currently exercised, but trivially possible.

NS coverage (NOT vendored yet): `nemo_skills/dataset/mmlu-pro/` → `METRICS_TYPE = "multichoice"`, prompt `eval/aai/mcq-10choices` (10-choice variant, not 4).

**Migration (Tier 1):**
1. Vendor `dataset/mmlu-pro/` + the 10-choice prompt yaml. Add registry row.
2. Existing multichoice runner needs a small fix: today `_multichoice.py` assumes `mcq-4choices*.yaml`. The 10-choice yaml will need rendering support (probably trivial — it's a different yaml file, same shape).

---

### LooGLE — one implementation, BERTScore

| path | dataset source | prompt | grader |
|---|---|---|---|
| `python/sglang/eval/loogle_eval.py` | `bigai-nlco/LooGLE`, `longdep_qa` config, `test` split | "Please answer the question based on the long texts below..." | `BERTScorer(lang="en")` (the bert_score package) F1 score |

**Implementation 1: `python/sglang/eval/loogle_eval.py` — only sglang path.**
Standalone script, dispatched as `python -m sglang.eval.loogle_eval`. Loads `bigai-nlco/LooGLE` from HF, `longdep_qa` configuration, `test` split — long-context multi-doc QA, contexts up to ~100k tokens. For each row, builds a prompt `"Please answer the question based on the long texts below.\n{context}\nQuestion: {question}\nAnswer:"`, sends through `openai.AsyncOpenAI` with `temperature=0.0`, `max_tokens=512`. Concurrency via `asyncio.Semaphore(args.max_concurrency)`, default 144. Caches each response as a pickle file in `tmp-output-dir/response_{idx}.pkl` (skips re-runs if file exists — useful for crash recovery on the long contexts). After all responses are collected, the `analyse` step uses `bert_score.BERTScorer(lang="en", device="cuda" if available)` to compute F1 between each model response and the reference answer in batches of 64. Reports average F1 as the headline metric. **Heaviest dependency in the entire eval surface:** `bert_score` itself pulls in `transformers` + a BERT-like model that gets downloaded at scoring time. Not wired to `run_eval`, not in CI today, no consumers found in `test/registered/`. Existence is more vestigial than active.

NS coverage: **NS does not have LooGLE.** No `nemo_skills/dataset/loogle/`.

**Migration:**
- Out of CI today, low signal. Punt unless someone explicitly needs it. If kept, would need: a third vendor source (or SE-authored), plus a new `embedding` or `judge` runner type for BERTScore-style soft metrics.

---

### asr (audio benchmark, legacy script)

`benchmark/asr/bench_sglang.py` — Whisper/audio ASR via the audio API; uses HF `evaluate` for WER.

User said legacy is out of scope. **Flagging only:** if audio evals come into scope, this is yet another runner type (`audio` sampler, WER metric, no NS upstream).

---

### Other "not in run_eval" CI paths recap

- **`kits/lm_eval_kit.py`**: lm-evaluation-harness in-process. Today only gsm8k yamls under `test/lm_eval_configs/`. Conceptually extensible to anything lm-eval covers. **Migration question:** is sgl-eval's value prop "replace lm_eval_kit too" (so we have one harness) or "coexist" (keep lm_eval_kit for benchmarks SE doesn't cover)? Recommend: replace, eventually, but not until SE has parity for the metrics those configs depend on (notably `flexible-extract`).

- **`kits/mmmu_vlm_kit.py`**: lmms-eval subprocess. See MMMU section.

- **`accuracy_test_runner.py:_run_nemo_skills_eval`**: NS shell-out. See mmmu/mmmu-pro section. **This is the path the user explicitly named as heavyweight.** Eliminating it for `mmmu-pro` is the single highest-value migration item.

---

## Source-coverage scorecard

The "implementations" view above is one lens. The other — and arguably
more useful for "is this benchmark *actually* migrated?" — is to count
**upstream data sources**. Several sglang benchmarks pull from more
than one upstream (gsm8k pulls from four, mmlu from five). For sgl-eval
to be a drop-in replacement, every upstream sglang touches needs to be
reachable through sgl-eval.

By that bar: **none of the 12 benchmarks are source-complete today.**

### Strict count — different upstream URL/dataset = different source

| benchmark | upstream sources sglang pulls from | sgl-eval covers | gap |
|---|---|---|---|
| **gsm8k** | (1) raw GitHub `openai/grade-school-math` test.jsonl, (2) `madrylab/gsm8k-platinum` HF, (3) `meta-llama/Llama-3.1-405B-Instruct-evals` HF, (4) lm-eval-harness's bundled gsm8k data | **1 of 4** (#1, via NS's `prepare.py`) | platinum variant, Meta's eval set, lm-eval's variant |
| **mmlu** | (1) `openaipublic.blob.core.windows.net/simple-evals/mmlu.csv`, (2) `meta-llama/Llama-3.1-405B-Instruct-evals` mmlu config, (3) same dataset's mmlu_cot config, (4) Hendrycks tarball `https://people.eecs.berkeley.edu/~hendrycks/data.tar`, (5) lm-eval's bundled mmlu | **1 of 5** (#4, via NS) | the **CI-active source is #1**, which sgl-eval doesn't have. Meta's two and lm-eval's also uncovered |
| **gpqa** | (1) `openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv` | **0 of 1 by literal source** — sgl-eval pulls `Idavidrein/gpqa` HF (the canonical original; the CSV is OpenAI's repackaging of the same diamond subset) | same 198 questions, different upstream URL |
| **aime25** | (1) `opencompass/AIME2025` HF (`AIME2025-I` + `AIME2025-II`) | **0 of 1 by literal source** — sgl-eval ships its own bundled `test.txt` curated by NS | same 30 problems, different package; verify with checksum |
| **aime24** | (none — not in sglang) | n/a — vacuously complete (forward-compat addition) |
| **math** (Hendrycks) | (1) `openaipublic.blob.core.windows.net/simple-evals/math_test.csv` | not registered. NS upstream uses Hendrycks's original GitHub release | different package, same 5000 questions; needs registration |
| **mgsm / mgsm_en** | (1)-(11) `openaipublic.blob.core.windows.net/simple-evals/mgsm_{lang}.tsv` for 11 languages | not registered. **NS doesn't have mgsm at any source** | requires second vendor source (lm-eval-harness) or SE-authored |
| **humaneval** | (1) `human_eval.data.read_problems()` (pip package's bundled JSONL) | not registered. NS upstream uses evalplus's HumanEval | different package, same 164 problems; needs `code` runner |
| **longbench_v2** | (1) `THUDM/LongBench-v2` HF default, (2) local file path via `--dataset-path` | not registered. NS upstream **pulls the same HF dataset** | source-equivalent once registered |
| **mmmu** | (1) `MMMU/MMMU` HF (val split, used by `simple_eval_mmmu_vlm` + legacy), (2) `lmms-lab--MMMU` parquet cache (lmms-eval shellout), (3) NS's prepared `mmmu-pro` data (NS shellout) | not registered | needs VLM sampler + new runner; NS only has mmmu-pro variant |
| **mmlu_pro** | (1) `meta-llama/Llama-3.1-405B-Instruct-evals` mmlu_pro config (only path) | not registered. NS upstream pulls `TIGER-Lab/MMLU-Pro` HF | different package; needs registration |
| **LooGLE** | (1) `bigai-nlco/LooGLE` HF | not registered | low signal, defer |

### "Same-questions-different-package" relaxation

For four benchmarks (gpqa, aime25, math, longbench_v2), sgl-eval points
at a *different upstream URL* but the underlying canonical question set
should be the same. Concretely:

- **gpqa**: `Idavidrein/gpqa` (NS) is the dataset authors' own HF
  release. `gpqa_diamond.csv` (sglang) is OpenAI's flat repackaging.
  Same 198 questions; permutation seeds differ, which is the score-affecting
  divergence — not the upstream choice itself.
- **aime25**: 30 problems (15 AIME-I + 15 AIME-II). Both `opencompass/AIME2025`
  and NS's bundled `test.txt` should be the same 30. **Worth a checksum
  verification before declaring equivalent.**
- **math**: 5000 rows of MATH test split. simple-evals CSV is OpenAI's
  repackaging; NS's `hendrycks_math/prepare.py` pulls Hendrycks's GitHub
  original.
- **longbench_v2**: **both pull from `THUDM/LongBench-v2` HF.** This is
  the only benchmark in the gap list that's *literally* source-equivalent —
  registering it in sgl-eval immediately closes the gap.

If "same-questions-yes" counts as covered:
- gsm8k: still **1 of 4** (the variant sources are genuinely different
  question sets — gsm8k-platinum has corrected mislabels; Meta's eval
  set is a fixed subset; lm-eval's prompt template selects different
  exemplars).
- mmlu: **0 of 5** of the *CI-relevant* sources covered. The
  simple-evals CSV (`mmlu.csv`) and Hendrycks tarball both contain the
  ~14k-row test split, but CI thresholds are calibrated against the
  CSV and not all rows are bit-identical between the two packages.
  Verify before swapping.
- gpqa: **same-questions-yes (1/1)** — but score will shift due to
  permutation seed difference.
- aime25: **likely same-questions (1/1)** — verify checksum.
- longbench_v2: **literally same source (1/1)** — once registered.
- everything else: 0 covered.

### Bottom line

**gsm8k is the only benchmark where sglang genuinely pulls from
multiple distinct upstream data sources.** sgl-eval covers the
dominant one (raw grade-school-math) but misses the three variant
sources (platinum, Meta-evals reproduction, lm-eval). Every other
benchmark is single-source in sglang or different-packaging-of-the-
same-source.

The implication for migration sequencing: gsm8k is in the registry but
*not source-complete* — declaring it "done" requires deciding what to
do about the four variant paths. Options:

1. **Drop the variants.** If the lm-eval gsm8k yaml configs and the
   `llama3_eval` Meta-evals reproduction aren't load-bearing for any
   ongoing decision, just retire them.
2. **Register them as separate sgl-eval entries.** `gsm8k-platinum`,
   `gsm8k-meta`, etc. — each is a one-row `_TABLE` addition once the
   loader knows about the alternate dataset. The platinum case is
   trivial (one HF dataset call); Meta and lm-eval are heavier because
   they import each ecosystem's own prompt machinery.
3. **Punt on the variants.** Migrate the dominant path now, leave the
   variants on the existing sglang harness until someone needs them.

Recommend option 3 for gsm8k specifically, then revisit if the variant
sources turn out to be load-bearing.

---

## Effort tiers and migration sequencing

### Tier 1 — paste into registry (NS already has it, fits existing runner)

These are one-PR migrations. Vendor the NS dataset module + any prompt yaml, add a row to `_TABLE`, run `pytest` to confirm grader parity, ship.

| benchmark | NS source | runner | rough scope |
|---|---|---|---|
| **Hendrycks MATH** | `dataset/hendrycks_math/` + `prompts/generic/math.yaml` (already vendored) | math | 1 SOURCES.yaml entry, 1 `_TABLE` row, score-parity check |
| **MATH-500** | `dataset/math-500/` (same prompt) | math | same |
| **MMLU-Pro** | `dataset/mmlu-pro/` + `prompts/eval/aai/mcq-10choices.yaml` (new) | multichoice | needs verifying multichoice runner handles 10-choice yaml |
| **LongBench-v2** | `dataset/longbench-v2/` + `prompts/eval/longbench/default.yaml` (new) | multichoice | + decision on tokenizer-based context filter (recommend drop) |
| **AIME24** (already in SE registry) | already done | math | n/a |

### Tier 2 — needs new runner type (NS has dataset, runner is the work)

| benchmark | NS source | new runner | rough scope |
|---|---|---|---|
| **HumanEval** | `dataset/human-eval/`, `METRICS_TYPE=evalplus` | `code` | sandboxed exec, pass@k aggregation. Wrap NS's evalplus integration or call openai/human-eval directly |
| **MMMU-Pro** | `dataset/mmmu-pro/`, multichoice + VLM prompt | `multimodal` | image-aware sampler, image-bearing Example, image data loader |

### Tier 3 — NS doesn't have it (needs second vendor source or SE-authored)

| benchmark | options |
|---|---|
| **MGSM** | (A) author SE-side, (B) vendor from lm-eval-harness — recommend B; first benchmark of the second vendor source |
| **MMMU** (basic, validation split) | (A) author SE-side, (B) vendor from lmms-eval; consider whether mmmu-pro is sufficient and we drop basic mmmu |
| **mmlu_cot** (Meta's CoT variant) | only in `llama3_eval.py`; either author or drop |
| **gsm8k-meta / mmlu-meta / mmlu_pro-meta** (the `llama3_eval.py` paths) | likely drop or treat as snapshot tools, not as accuracy evals |
| **gsm8k-platinum** | trivial second-source: the data is one HF call; could be a new registry row that points at `madrylab/gsm8k-platinum` and reuses the existing math runner |
| **LooGLE** | low signal; defer |

### Sequencing recommendation

1. **First wave (week 1-ish, low risk):** MATH, MATH-500, MMLU-Pro, LongBench-v2. All Tier 1. Each is a paste job. Ship them and prove the migration cadence.
2. **Second wave: deprecation trail.** Migrate the `simple_eval_*` callers in CI to call sgl-eval directly. Especially the `kits/eval_accuracy_kit.py` mixins — those are the broadest blast radius. *No new sglang code* references `simple_eval_*` after this wave.
3. **Third wave: kill the heavyweight shell-out.** Implement `multimodal` runner + VLM sampler + vendor mmmu-pro. Replace `accuracy_test_runner.py:_run_nemo_skills_eval` with an in-process sgl-eval call. Then replace `kits/mmmu_vlm_kit.py`'s lmms-eval subprocess with sgl-eval as well. **This is the biggest user-visible win** — CI gets minutes faster, no `pip install nemo_skills` per run.
4. **Fourth wave: HumanEval.** Code runner + sandbox; design carefully.
5. **Fifth wave: MGSM (or whichever second-vendor-source benchmark goes first).** Establishes the lm-eval-harness vendoring pattern.

---

## Cross-cutting concerns

### Score parity verification

Every migration row in this doc has a "scores will differ slightly" hazard because sgl-eval's grader (NS) and sglang's grader (per-benchmark ad-hoc) are different code paths. Before any benchmark is *deprecated* on the sglang side, do an A/B run of the same model against both harnesses and document the delta in `MIGRATION_PLAN.md` (or a per-benchmark notes file). Acceptance bar should be something like "<1% absolute difference, with each model having the same relative ranking" rather than bit-exactness.

### CI threshold migration

Every `*_score_threshold` / `baseline_accuracy` constant in `test/registered/` is calibrated against *some specific harness*. When the harness changes underneath, those thresholds may need to shift. List of CI consumers to audit (from grep):

- `kits/eval_accuracy_kit.py`: `gsm8k_accuracy_thres`, `mmlu_score_threshold`, `humaneval_score_threshold(_amd)`, `mgsm_en_score_threshold`.
- `accuracy_test_runner.py:AccuracyTestParams.baseline_accuracy` — every `test/registered/...` test that constructs one.
- `kits/lm_eval_kit.py` consumers: yamls in `test/lm_eval_configs/` carry `value:` per metric.

### Deprecation paths to actually delete

After migration:

- `python/sglang/test/few_shot_gsm8k.py` + `few_shot_gsm8k_engine.py` — already marked deprecated; just need their NPU-test consumers to migrate.
- `python/sglang/test/simple_eval_*.py` — the whole file family.
- `python/sglang/test/run_eval.py` — replaced by `sgl-eval run`.
- `python/sglang/eval/llama3_eval.py` — keep iff the Meta-evals snapshot is genuinely a different benchmark (likely yes, but reframe as a regression tool, not an accuracy eval).
- `python/sglang/eval/loogle_eval.py` — drop unless someone owns it.
- `sgl-model-gateway/e2e_test/infra/simple_eval_{common,mmlu}.py` — gateway switches to importing sgl-eval.
- `python/sglang/test/kits/{eval_accuracy_kit,mmmu_vlm_kit,lm_eval_kit}.py` — collapse into thin wrappers around `sgl-eval` (or delete if CI moves to calling `sgl-eval run` directly).

### The vendor-rule pressure point

Three benchmarks (MGSM, basic MMMU, mmlu_cot) don't exist in NS. sgl-eval's core architectural principle is "anything that decides a score is vendored verbatim from NS." The migration forces a decision:

- **Strict:** drop those benchmarks. sgl-eval covers what NS covers.
- **Roadmap-as-written:** accept second/third vendor sources (`lm-evaluation-harness`, `lmms-eval`, `simple-evals`) as already anticipated in `README.md` line 99-101 and `_vendored/<source>/` per-source.
- **Soft:** allow SE-authored graders for benchmarks no upstream offers, with explicit annotation in `CLAUDE.md`.

Recommend roadmap-as-written. The `_vendored/lm_evaluation_harness/` slice would be small (mgsm prep + grader is a few hundred lines), and it unblocks future expansion.

---

## Open questions for the team

1. **Score parity tolerance.** What's the acceptable delta when migrating, e.g. "gsm8k goes from 87.0% to 86.7% because the regex changed"? Bit-exact is unattainable without preserving every grader; a documented tolerance avoids per-PR re-litigation.
2. **`kits/lm_eval_kit.py` keep-or-replace.** Some customers explicitly want lm-eval's `flexible-extract` numbers for legacy comparability. Does sgl-eval try to match that, or is the answer "lm_eval_kit stays for now"?
3. **`llama3_eval.py` keep-or-drop.** Is Meta's pinned-eval-set methodology a benchmark we want to support long-term, or was it always a one-off?
4. **VLM scope in v1.** mmmu-pro is the only NS VLM eval. Is it OK to ship sgl-eval's VLM support with just mmmu-pro and no basic mmmu, or does that block the migration of `kits/mmmu_vlm_kit.py`?
5. **mgsm: NS-second-source vs SE-authored.** Worth the architectural addition of a second `_vendored/<source>/`, or simpler to author one benchmark out-of-pattern?
6. **CI threshold rebaseline plan.** Who owns rerunning the registered models against sgl-eval and updating the thresholds? Is this a one-time exercise per migration wave, or per-PR?

---

## Appendix: legacy `benchmark/<name>/` scripts (out of scope, for reference)

User direction: not in scope for this migration. Listed here only because they exist and someone might find this doc later.

Accuracy-flavored:
- `benchmark/gsm8k/`, `benchmark/mmlu/`, `benchmark/mmmu/`, `benchmark/hellaswag/`, `benchmark/boolq/`, `benchmark/ceval/`, `benchmark/reasoning_benchmark/` (LIMO), `benchmark/asr/`.

Not really accuracy:
- `benchmark/{llm_judge,mtbench,multi_document_qa,multi_turn_chat,json_schema,line_retrieval,tree_of_thought_*,multi_chain_reasoning,tip_suggestion,react,dspy,generative_agents,long_json_decode,prefill_only,gpt_oss,deepseek_v3}`.

If legacy comes back into scope, the additions to plan:
- **hellaswag**, **boolq**, **ceval**: NS doesn't have any of them. All would need a second vendor source (lm-eval-harness has hellaswag and boolq) or be authored SE-side.
- **reasoning_benchmark / LIMO**: own `answer_extraction.py` + `eval_utils.math_equal` (different from NS's). Likely subsumed by NS's `math` runner with the right prompt.
- **asr**: audio runner type, never going to be a one-line registry add.
