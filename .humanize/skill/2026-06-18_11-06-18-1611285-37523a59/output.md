AGREE:
- Additive minimal closure is the right porting method.
- `validator.py` must ship; `calibrate.py`, capture modules, oracle/comparator/dev tests should not.
- Keeping `meta_info["double_sparsity"]` as the DS-active proof is reasonable; DEC-2 can be closed.
- Adding the missed `logits_processor.py` dataclass field and retargeting CUDA graph hunks are correctly called out.

DISAGREE:
- M2/M3 ordering is wrong. Pruning before re-applying upstream hunks can reintroduce dev refs. Apply/retarget hunks and prune dev-only branches per touched file, then run global closure checks.
- AC-2 is both too broad and too narrow: generic `capture` grep will false-fail on legitimate CUDA graph capture, while import-only checks miss attr reads like `selection_capture`.
- AC-5 is not testable while DEC-4/DEC-6 are unresolved. Radix-on plus no fixture/mask artifact makes server boot/perf acceptance undefined.
- AC-6 is not testable as written. “Same scheduler step” needs an explicit injection and call-order/state assertion.
- AC-7 is under-specified. “Written parity band” and “p50 decode TPS” need an exact formula and numeric threshold.
- AC-8 is too strict at symbol granularity. Runtime helpers may be valid without direct serve-path references.

REQUIRED_CHANGES:
- Pin the base commit, or explicitly decide to refresh from latest main. Do not leave “current main” ambiguous.
- Reorder milestones: inventory -> copy pure new files -> apply retargeted hunks with pruning -> static/import/unit closure -> DS boot -> abort test -> perf -> final sweep -> push.
- Replace AC-2/AC-3 with precise sweeps for `score_capture`, `selection_capture`, `latent_capture`, `radix_fixture_capture`, `calibrate`, `oracle_artifact_sink`, `selection_recall_oracle`, and excluded paths.
- Decide artifact policy before M5: committed sanitized mask or external immutable path with recorded SHA. Do not keep `calibrate.py` to solve this.
- Split radix acceptance: fixture artifact, override-based perf run, or radix-off. Also add a negative test that radix-on without fixture/override fails closed.
- Make AC-5 command-level: exact model revision, env, server args, response path for `meta_info`, CUDA graph enabled log/assertion, and dense fallback rejection.
- Make AC-6 command/test-level: how the DS per-request error is injected and what state/call order proves `set_finish_with_abort` plus `update_finish_state`.
- Define perf metric exactly and numerically: formula, tolerance around 26.9 TPS / 25.1s P99 TTFT, prompt count, seed, warmup, output artifact.
- Add a kernel/package closure gate for copied Triton/sgl-kernel/deep_gemm/flash_mla dependencies, or explicitly prove no new dependency is introduced.
- Resolve `lifted_budget`: keep code plus test, or drop both. The current “lower bound minus lifted_budget” conflicts with the stated test floor.

OPTIONAL_IMPROVEMENTS:
- Collapse DEC-3 to stock `bench_serving` plus a small wrapper if the wrapper defines the parity metric.
- Run slim validator/runtime tests before expensive 8-GPU boot/perf.
- Save perf command, server args, commit SHA, GPU info, and JSON output as eval evidence.
- Use `git diff --name-only <base>...HEAD` for diff hygiene instead of informal branch inspection.

UNRESOLVED:
- Channel mask provenance: self-contained branch artifact vs external artifact path.
- Radix-on semantics: fixture artifact vs `SGLANG_DS_RADIX_OVERRIDE=1` vs relaxing AC-5 to radix-off.
- Numeric perf tolerance: needs human agreement because one trial plus GPU variance can otherwise create arbitrary pass/fail.
- Base target: exact `105e095e0` vs refreshed latest main.
