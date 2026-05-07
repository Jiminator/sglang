# Nightly CI Regression Report: `test_encoder_dp.py` GLM-4.1V-9B-Thinking 0.42 Failure

**Reference failing job:** https://github.com/sgl-project/sglang/actions/runs/24971499389/job/73115280649
**Test:** `test/registered/vlm/test_encoder_dp.py::TestVLMEncoderDP::test_vlm_mmmu_benchmark`
**Suite:** `nightly-4-gpu`
**Workflow:** `.github/workflows/nightly-test-nvidia.yml` (job `nightly-test-general-4-gpu-h100`)
**Reported assertion:** `Model zai-org/GLM-4.1V-9B-Thinking accuracy (0.4200) below expected threshold (0.6800)`

## Verdict

**The regression is still present at HEAD.** It is **not** a model accuracy regression — it is a CI test-isolation bug. The encoder_dp test reads a stale `lmms-eval` result file from `test_epd_disaggregation.py`, which now runs ahead of it in the same nightly suite.

## Root Cause

Two changes combined to produce the failure:

1. **PR #23518 / commit `267c2c0849` (2026-04-22)** moved `test_epd_disaggregation.py` from `stage-c-test-4-gpu-h100` into the `nightly-4-gpu` suite, where `test_encoder_dp.py` already lives. Diff:
   ```diff
   -register_cuda_ci(est_time=97, suite="stage-c-test-4-gpu-h100")
   +register_cuda_ci(est_time=97, suite="nightly-4-gpu", nightly=True)
   ```
2. **CI runner provisioning change between 2026-04-16 and 2026-04-20** moved the 4-gpu-h100 nightly job off ephemeral Docker containers (machine names `ae7f65c23069`, `8eb62ec92095`, `ea80be1a94ea`) onto persistent bare hosts (`host-85-234-79-31`, `host-10-240-159-123`). The bare hosts do not wipe `./logs/` between tests in the same job.

After these two changes, `test_epd_disaggregation.py::test_epd_disaggregation_multi_encoder_mmmu` runs first, evaluates `Qwen/Qwen2.5-VL-3B-Instruct` at MMMU `limit=50`, and writes its result JSON to `./logs/epd_multi_encoder_mmmu/__Qwen__Qwen2.5-VL-3B-Instruct__/.../results_*.json` (accuracy ≈ 0.42 on the 50-sample subset). When `test_encoder_dp.py` runs immediately after, this code at `test/registered/vlm/test_encoder_dp.py:168` picks up that stale file:

```python
result_files = glob.glob(f"{output_path}/**/*.json", recursive=True)
# output_path == "./logs"  →  recursive glob includes ./logs/epd_multi_encoder_mmmu/...
result_file_path = result_files[0]
```

The test then asserts the stale 0.4200 against whichever model's threshold was randomly selected, so it appears as a failure of `Qwen2.5-VL-72B`, `Qwen3-VL-32B`, `InternVL2_5-8B`, or `GLM-4.1V-9B-Thinking` — all reporting the same impossible-to-coincidentally-match 0.4200.

## Smoking Gun

Failing reference job (run 24971499389), the JSON encoder_dp reads contains:

```
'model_args': 'model_version="Qwen/Qwen2.5-VL-3B-Instruct",tp=1'
'limit': 50.0, 'effective': 50
'mmmu_acc,none': 0.42
```

`Qwen2.5-VL-3B-Instruct` is **not** in encoder_dp's `MODELS` list (`test/registered/vlm/test_encoder_dp.py:23-28`). It is the model used by the EPD disaggregation test. Encoder_dp does not pass `--limit`, so its own eval would run on all 900 samples, not 50.

## Pass/Fail Timeline (encoder_dp-relevant evidence only)

| Date | Run | SHA | Machine | Selected model | Accuracy | Verdict |
|------|-----|-----|---------|----------------|----------|---------|
| 2026-04-08 | 24111611527 | dd73e9a6 | ae7f65c23069 (docker) | Qwen2.5-VL-72B | 0.6278 | PASS |
| 2026-04-09 | 24166163578 | 2c4e113d | 8eb62ec92095 (docker) | Qwen3-VL-32B | 0.6056 | PASS |
| 2026-04-10 | 24220618561 | 5638d40f | 8eb62ec92095 (docker) | Qwen2.5-VL-72B | 0.6278 | PASS |
| 2026-04-11 | 24270600673 | 0011d2ae | 8eb62ec92095 (docker) | Qwen3-VL-32B | 0.6133 | PASS |
| 2026-04-13 | 24320635302 | 37fc47c6 | ae7f65c23069 (docker) | **GLM-4.1V** | **0.2389** | **FAIL (different signature, pre-regression)** |
| 2026-04-14 | 24374813228 | c456cba7 | ae7f65c23069 (docker) | Qwen2.5-VL-72B | 0.6289 | PASS |
| 2026-04-16 | 24486059047 | a4cf2ea1 | ea80be1a94ea (docker) | Qwen2.5-VL-72B | 0.6289 | PASS — last clean encoder_dp pass in window |
| Apr 17–19 | — | — | — | — | — | runs cancelled |
| 2026-04-20 | 24643380928 | 1cff871c | host-85-234-79-31 (bare) | InternVL2_5-8B | (not logged before crash) | inconclusive |
| 2026-04-21 | 24698143162 | 712b01d8 | host-10-240-159-123 | Qwen2.5-VL-72B | 0.6278 | PASS (job failure was elsewhere) |
| 2026-04-22 | 24754179764 | 1408d974 | host-10-240-159-123 | Qwen2.5-VL-72B | 0.6278 | PASS |
| **2026-04-22 20:59 PT** | — | **267c2c08** | — | — | — | **EPD merged into nightly-4-gpu** |
| 2026-04-23 | 24810524606 | c689f774 | host-85-234-79-31 | Qwen2.5-VL-72B | 0.6278 | PASS (commit not yet in HEAD) |
| 2026-04-25 | 24918423411 | a4facdf3 | host-10-240-159-123 | GLM, then InternVL | (no value before crash) | first run after EPD merge |
| 2026-04-27 | 24971499389 | 977830e9 | host-85-234-79-31 | Qwen3-VL-32B, then **GLM-4.1V** | **0.4200** | **FAIL — reference job, stale-file signature** |
| 2026-04-28 | 25027858196 | 4a04a981 | host-10-240-159-123 | InternVL2_5-8B | 0.4200 | FAIL (stale) |
| 2026-04-29 | 25085535219 | 14b4e6fa | host-85-234-79-31 | Qwen3-VL-32B, Qwen2.5-VL-72B | 0.4200 | FAIL (stale) |
| 2026-04-30 | 25141769035 | 3553fd03 | host-10-240-159-123 | Qwen2.5-VL-72B, InternVL | 0.4200 | FAIL (stale) |
| 2026-05-04 | 25295900861 | c611a3fb | host-10-240-159-123 | InternVL2_5-8B | 0.4200 | FAIL (stale) |
| 2026-05-05 | 25351892972 | 2f7d99b7 | host-85-234-79-31 | InternVL, then **GLM-4.1V** | **0.4200** | **FAIL (stale)** |
| 2026-05-06 | 25410559507 | b91b05ae | host-85-234-79-31 | Qwen2.5-VL-72B, then **GLM-4.1V** | **0.4200** | **FAIL (stale) — most recent failed run** |
| 2026-05-07 | 25469734855 | 2e642ea1 | — | — | — | cancelled |

## Confidence

- That the current Apr 25 → present failures are a stale-`./logs` test-isolation bug, not a model regression: **High.** Identical `0.4200` across four different models, `model_version="Qwen2.5-VL-3B-Instruct"` in the read JSON (not in encoder_dp's list), `limit=50` matching EPD not encoder_dp.
- That `267c2c0849` is the introducer in conjunction with the runner change: **High.** Date alignment is exact; before the merge encoder_dp was passing on bare hosts (Apr 21–23 reading 0.6278); the first nightly to include the merged commit (Apr 25) regressed.

## Caveats

- Random model selection in `test_encoder_dp.py` (line 248-249) means most scheduled runs don't pick GLM-4.1V specifically. The diagnosis is anchored on the cross-model 0.4200 signature, which is independent of which model was picked.
- I could not find the explicit commit that flipped the 4-gpu-h100 nightly runner pool from Docker to bare host. It's a self-hosted runner provisioning change, inferred from machine-name patterns. The encoder_dp.py code itself contained the latent stale-glob bug all along; PR #23518 just made it observable on the bare-host runners.
- The 2026-04-13 GLM-4.1V failure (0.2389) is a **separate, pre-existing GLM-specific issue** unrelated to this stale-file regression. With only one GLM data point inside docker in this window, I cannot confirm whether GLM-4.1V ever cleared 0.68 in this 4-gpu-h100 nightly suite.

## Recommendations

The fix is on the test side, not on any model. Two minimally invasive options:

### Option 1 — Use a per-test temp output dir (recommended)

```python
# test/registered/vlm/test_encoder_dp.py
import tempfile, shutil

class TestVLMEncoderDP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ...
        cls.output_dir = tempfile.mkdtemp(prefix="encoder_dp_logs_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir, ignore_errors=True)

    def test_vlm_mmmu_benchmark(self):
        ...
        for model in models_to_test:
            self._run_vlm_mmmu_test(model, self.output_dir)
```

This eliminates cross-test contamination entirely.

### Option 2 — Scope the glob to the model's own subdirectory

`lmms-eval` writes results to `<output_path>/<sanitized_model_name>/...`, so:

```python
sanitized = model.model.replace("/", "__")
result_files = glob.glob(f"{output_path}/**/*{sanitized}*/*.json", recursive=True)
```

This is a 2-line fix but is fragile (sanitization rules can change in lmms-eval).

### Apply the same fix to test_epd_disaggregation.py

The same defect exists in `test/registered/distributed/test_epd_disaggregation.py:798` and `:1201` — both use the same `glob.glob(f"{output_path}/**/*.json", recursive=True)` pattern with shared `./logs/...` paths. While EPD currently runs first (so it isn't affected today), a future test ordering change could expose the same bug in reverse.

### After fixing

Once the stale-file bug is removed, re-run with the targeted reproduction to confirm GLM-4.1V actually clears 0.68 (it has not been observed to do so in this nightly window — that is a separate question worth answering once the stale-file confound is gone):

```bash
PYTHONPATH=test/registered/vlm python3 -c '
import sys, unittest
from types import SimpleNamespace
import test_encoder_dp as m
m.MODELS = [SimpleNamespace(model="zai-org/GLM-4.1V-9B-Thinking", mmmu_accuracy=0.68)]
suite = unittest.TestLoader().loadTestsFromName("TestVLMEncoderDP.test_vlm_mmmu_benchmark", m)
sys.exit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)'
```
