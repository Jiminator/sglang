"""Exercise the SOUND fail-closed table-free radix fixture gate end-to-end.

Validates, against a real ServerArgs-shaped object bound to THIS boot's config
(GLM-5.1-FP8 / TP8 / page64 / fp8_e4m3 / the calibrated channel mask), that
`apply_radix_fixture_artifact`:

  1. ACCEPTS a correctly-formed table-free state file (schema
     ds_radix_fixture_state_tablefree_v1, every probe field the JSON bool true,
     config fingerprint matching this boot) -> records _double_sparsity_radix_fixture_passed.
  2. REJECTS the legacy label-capture schema (the deleted TokenLabelTable fixture).
  3. REJECTS a state with a probe field that is NOT the JSON bool true (e.g. the
     string "true", or false, or missing) -> the `is True` fail-closed.
  4. REJECTS a state recorded for a DIFFERENT config (e.g. tp_size mismatch /
     a different channel-mask SHA).
  5. REJECTS a missing artifact path.

Run on HEAD 49a401a72 (the SOUND gate). Writes gate_demo_result.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/sgl-workspace/sglang")
from sglang.srt.layers.attention.double_sparsity.validator import (  # noqa: E402
    RADIX_FIXTURE_STATE_SCHEMA,
    RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA,
    apply_radix_fixture_artifact,
    radix_fixture_config_fingerprint,
    write_radix_fixture_state,
)

MASK = "/models/glm51-fp8-channel-mask-s256.safetensors"
GLM = (
    "/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/"
    "f396cf805182f4ca10fa675e1a99815b3ca384db"
)
DS_CONFIG = json.dumps(
    {
        "top_k": 2048,
        "page_size": 64,
        "channel_mask_path": MASK,
        "device_buffer_size": 4096,
        "scorer_norm": "off",
        "head_agg": "max",
        "anchor_mode": "off",
        "anchor_budget": 0,
        "recall_oracle": False,
        "enable_lifted_budget_decode": False,
        "lifted_budget_top_k": 0,
    }
)
HERE = os.path.dirname(os.path.abspath(__file__))


class FakeArgs:
    """Minimal ServerArgs-shaped object: only the attrs the gate reads."""

    def __init__(self, **over):
        self.enable_double_sparsity = True
        self.disable_radix_cache = False  # radix ON (needs authorization)
        self.double_sparsity_radix_fixture_artifact = None
        self.double_sparsity_config = DS_CONFIG
        self.model_path = GLM
        self.tp_size = 8
        self.page_size = 64
        self.kv_cache_dtype = "fp8_e4m3"
        self._double_sparsity_radix_fixture_passed = False
        for k, v in over.items():
            setattr(self, k, v)


def _try(args):
    try:
        apply_radix_fixture_artifact(args)
        return getattr(args, "_double_sparsity_radix_fixture_passed", False), None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main():
    results = {}

    # 1. Correctly-formed table-free artifact (all three probe fields True,
    #    config fingerprint matches this boot) -> ACCEPT.
    good = os.path.join(HERE, "tablefree_fixture_state_GOOD.json")
    args = FakeArgs()
    write_radix_fixture_state(
        good,
        server_args=args,
        cold_warm_selection_equivalence_passed=True,
        recall_radixon_passed=True,
        edge_probe_passed=True,
    )
    args.double_sparsity_radix_fixture_artifact = good
    ok, err = _try(args)
    results["1_good_tablefree_accepts"] = {"authorized": ok, "error": err}

    # 2. Legacy label-capture schema -> REJECT.
    legacy = os.path.join(HERE, "_legacy_label_state.json")
    state = json.load(open(good))
    state["schema"] = RADIX_FIXTURE_STATE_SCHEMA
    json.dump(state, open(legacy, "w"))
    args = FakeArgs(double_sparsity_radix_fixture_artifact=legacy)
    ok, err = _try(args)
    results["2_legacy_schema_rejects"] = {"authorized": ok, "error": err}

    # 3a. Probe field is the STRING "true" (truthy non-bool) -> REJECT (`is True`).
    strtrue = os.path.join(HERE, "_strtrue_state.json")
    state = json.load(open(good))
    state["edge_probe_passed"] = "true"
    json.dump(state, open(strtrue, "w"))
    args = FakeArgs(double_sparsity_radix_fixture_artifact=strtrue)
    ok, err = _try(args)
    results["3a_string_true_rejects"] = {"authorized": ok, "error": err}

    # 3b. Probe field false -> REJECT.
    falsest = os.path.join(HERE, "_false_state.json")
    state = json.load(open(good))
    state["recall_radixon_passed"] = False
    json.dump(state, open(falsest, "w"))
    args = FakeArgs(double_sparsity_radix_fixture_artifact=falsest)
    ok, err = _try(args)
    results["3b_false_field_rejects"] = {"authorized": ok, "error": err}

    # 4. Config mismatch (tp_size recorded 8, boot tp_size 4) -> REJECT.
    args = FakeArgs(
        tp_size=4, double_sparsity_radix_fixture_artifact=good
    )
    ok, err = _try(args)
    results["4_config_mismatch_rejects"] = {"authorized": ok, "error": err}

    # 5. Missing artifact path -> REJECT.
    args = FakeArgs(
        double_sparsity_radix_fixture_artifact=os.path.join(HERE, "_nope.json")
    )
    ok, err = _try(args)
    results["5_missing_artifact_rejects"] = {"authorized": ok, "error": err}

    # Cleanup the throwaway negatives (keep GOOD as the deliverable artifact).
    for f in (legacy, strtrue, falsest):
        try:
            os.remove(f)
        except OSError:
            pass

    verdict = (
        results["1_good_tablefree_accepts"]["authorized"] is True
        and all(
            results[k]["authorized"] is False
            for k in results
            if k != "1_good_tablefree_accepts"
        )
    )
    out = {
        "gate_demo_verdict": "PASS" if verdict else "FAIL",
        "good_artifact": good,
        "good_artifact_fingerprint": radix_fixture_config_fingerprint(FakeArgs()),
        "schemas": {
            "tablefree": RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA,
            "legacy_rejected": RADIX_FIXTURE_STATE_SCHEMA,
        },
        "cases": results,
    }
    with open(os.path.join(HERE, "gate_demo_result.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
