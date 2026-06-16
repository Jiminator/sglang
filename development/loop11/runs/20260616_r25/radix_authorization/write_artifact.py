"""R25 — write the table-free radix-fixture state file (schema v2) IFF all 5 probes
passed. Calls validator.write_radix_fixture_state with the EXACT serving ServerArgs
for this op-point so the config fingerprint matches the no-override boot.

This script REFUSES to write a passing artifact unless every one of the 5 value-
equivalence probe verdicts (read from the probe verdict JSONs) is a real PASS. It
re-derives each pass flag from the raw probe verdicts — it does not trust a summary.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.layers.attention.double_sparsity import validator  # noqa: E402

GLM = "/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db"
DS_CFG = (
    '{"top_k": 2048, "page_size": 64, '
    '"channel_mask_path": "/models/glm51-fp8-channel-mask-s256.safetensors", '
    '"device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", '
    '"anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, '
    '"lifted_budget_top_k": 0}'
)


def _build_server_args() -> ServerArgs:
    # Construct the EXACT serving ServerArgs for this op-point. We only need the
    # fields radix_fixture_config_fingerprint reads (model_path, tp_size, page_size,
    # kv_cache_dtype, double_sparsity_config -> channel_mask_path), but build a full
    # ServerArgs so the fingerprint is computed exactly as it will be at boot.
    return ServerArgs(
        model_path=GLM,
        tp_size=8,
        kv_cache_dtype="fp8_e4m3",
        page_size=64,
        mem_fraction_static=0.8,
        max_running_requests=64,
        trust_remote_code=True,
        enable_double_sparsity=True,
        double_sparsity_config=DS_CFG,
        dsa_prefill_backend="flashmla_kv",
        dsa_decode_backend="flashmla_kv",
        random_seed=20260607,
    )


def _probe_passes(verdict_dir):
    """Return (flags dict, problems list) derived from the raw probe verdicts."""
    problems = []
    flags = {}

    # Probe 1 recall_equivalence
    with open(f"{verdict_dir}/p1_recall_verdict.json") as fh:
        p1 = json.load(fh)
    flags["recall_equivalence_passed"] = p1.get("verdict") == "PASS"
    if not flags["recall_equivalence_passed"]:
        problems.append(f"probe1 recall: verdict={p1.get('verdict')} problems={p1.get('problems')}")

    # Probe 2 cross_rank_selection_identity (both paths)
    with open(f"{verdict_dir}/p2_cross_rank/radix_on_verdict.json") as fh:
        p2on = json.load(fh)
    with open(f"{verdict_dir}/p2_cross_rank/radix_off_verdict.json") as fh:
        p2off = json.load(fh)
    flags["cross_rank_selection_identity_passed"] = (
        p2on.get("status") == "PASS" and p2off.get("status") == "PASS"
    )
    if not flags["cross_rank_selection_identity_passed"]:
        problems.append(
            f"probe2 cross-rank: on={p2on.get('status')} off={p2off.get('status')}"
        )

    # Probe 3 edge_probe
    with open(f"{verdict_dir}/p3_edge_verdict.json") as fh:
        p3 = json.load(fh)
    flags["edge_probe_passed"] = p3.get("status") == "PASS"
    if not flags["edge_probe_passed"]:
        problems.append(f"probe3 edge: status={p3.get('status')} problems={p3.get('problems')}")

    # Probe 4 no_dense_fallback
    with open(f"{verdict_dir}/p4_no_dense_fallback_verdict.json") as fh:
        p4 = json.load(fh)
    flags["no_dense_fallback_passed"] = p4.get("status") == "PASS"
    if not flags["no_dense_fallback_passed"]:
        problems.append(f"probe4 no_dense_fallback: status={p4.get('status')} reason={p4.get('reason')}")

    # Probe 5 cold_warm_flips_value_neutral_documented
    with open(f"{verdict_dir}/p5_cold_warm_flips_verdict.json") as fh:
        p5 = json.load(fh)
    flags["cold_warm_flips_value_neutral_documented"] = p5.get("documented") is True
    if not flags["cold_warm_flips_value_neutral_documented"]:
        problems.append(f"probe5 cold_warm_flips: documented={p5.get('documented')}")

    return flags, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    flags, problems = _probe_passes(args.verdict_dir)
    print("[write_artifact] derived probe flags:")
    for k, v in flags.items():
        print(f"  {k} = {v}")
    if problems:
        print("[write_artifact] REFUSING to write a passing artifact — probe(s) failed:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if not all(flags.values()):
        print("[write_artifact] REFUSING: not every probe flag is True.")
        return 2

    sa = _build_server_args()
    state = validator.write_radix_fixture_state(
        args.out,
        server_args=sa,
        recall_equivalence_passed=True,
        cross_rank_selection_identity_passed=True,
        edge_probe_passed=True,
        no_dense_fallback_passed=True,
        cold_warm_flips_value_neutral_documented=True,
    )
    print(f"[write_artifact] wrote v2 artifact -> {args.out}")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
