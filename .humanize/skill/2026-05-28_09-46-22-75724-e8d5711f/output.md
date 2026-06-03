Edited [development/loop5/plan.md](/sgl-workspace/sglang/development/loop5/plan.md:14) with `Codex —` comment blocks only. I did not touch the Original Design Draft section.

**Verdicts**
- Root blocker: agree — AC-0 capture bug is env-gated radix evidence, not calibration/mask generation.
- AC-10 text: partially agree — avoid analyze drift, but validator/launcher wiring is real plumbing, not just a flag delete.
- task2 dependency: partially agree — graph allows calibration before task2, but calibration does not need AC-0; task5 already gates both.
- task3 calibration load: disagree — actual load is bare `AutoModelForCausalLM.from_pretrained(..., torch_dtype=..., device_map={"": "cuda"...})`; real risk is HF FP8 sharded load/upcast and `model.device`.
- task8 label: agree — benchmark scripts produce JSONL artifacts; comparator consumes them.
- task10 design: partially agree — collapse/require artifact, but no-env radix flip needs a ServerArgs/launcher/artifact contract.
- task14 CUDA graph: agree — regular CUDA graph capture happens at model-runner boot independent of radix; replay status follows from smoke request logs/metrics.

**New Critiques Added**
- Key feasibility: require a one-block sharded calibration dry run logging dtypes/devices.
- AC-1: launcher still defaults to HF id, so task5 must pin/override cluster `MODEL_PATH`.
- AC-6: distinguish regular CUDA graph from disabled piecewise CUDA graph.
- DEC-5: narrow away env override for final evidence.
