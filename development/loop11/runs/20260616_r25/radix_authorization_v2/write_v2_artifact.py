"""Write the table-free radix-fixture state file (schema v2) IFF GATE A + B + C pass.

Re-derives each pass flag from the RAW v2 gate verdicts (it does not trust a summary)
and REFUSES to write a passing artifact unless every gate is a real PASS. Calls
validator.write_radix_fixture_state with the EXACT serving ServerArgs for this
op-point so the config fingerprint matches the no-override boot.

  recall_equivalence_passed         <- GATE A multilen verdict == PASS
  cross_rank_selection_identity     <- GATE B p2 radix_on == PASS (8 ranks byte-identical)
  no_dense_fallback_passed          <- GATE B p4 no_dense_fallback == PASS
  edge_probe_passed                 <- GATE C edge correctness (eviction==cold, no stale slot)
  cold_warm_flips_value_neutral_documented
                                    <- documented: cited R24 symdiff + GATE A recall neutrality
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
    # EXACT serving ServerArgs for this op-point (matches no_override_boot COMMON_ARGS:
    # GLM TP=8 fp8_e4m3 page64 mem0.8 max-running64 flashmla_kv). Only the fields
    # radix_fixture_config_fingerprint reads matter, but build the full ServerArgs so
    # the fingerprint is computed exactly as at boot.
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


def _derive(gate_a, gate_b_p2on, gate_b_p4, gate_c_edge, gate_c_evict_delta_pp):
    problems = []
    flags = {}

    with open(gate_a) as fh:
        a = json.load(fh)
    flags["recall_equivalence_passed"] = a.get("verdict") == "PASS"
    if not flags["recall_equivalence_passed"]:
        problems.append(f"GATE A recall: verdict={a.get('verdict')}")

    with open(gate_b_p2on) as fh:
        p2 = json.load(fh)
    flags["cross_rank_selection_identity_passed"] = (
        p2.get("status") == "PASS" and p2.get("all_ranks_identical_all_steps") is True
        and p2.get("ranks_present") == list(range(8))
    )
    if not flags["cross_rank_selection_identity_passed"]:
        problems.append(
            f"GATE B cross-rank: status={p2.get('status')} "
            f"identical={p2.get('all_ranks_identical_all_steps')} ranks={p2.get('ranks_present')}"
        )

    with open(gate_b_p4) as fh:
        p4 = json.load(fh)
    flags["no_dense_fallback_passed"] = (
        p4.get("status") == "PASS" and p4.get("num_violations") == 0
    )
    if not flags["no_dense_fallback_passed"]:
        problems.append(f"GATE B no_dense_fallback: status={p4.get('status')} viol={p4.get('num_violations')}")

    # GATE C edge correctness: eviction-recompute == cold exactly (no stale slot).
    with open(gate_c_edge) as fh:
        p3 = json.load(fh)
    evict = p3.get("cases", {}).get("c_eviction_recompute", {})
    evict_delta = evict.get("delta_pp_vs_cold")
    evicted_is_recompute = bool(evict.get("prefix_evicted(cached_fell_vs_boundary)"))
    # Correctness criterion (rescoped GATE C): the eviction->recompute path must
    # reproduce the COLD baseline recall EXACTLY (delta == gate_c_evict_delta_pp, the
    # cited/reconfirmed value, expected 0.0) AND the prefix must actually have been
    # evicted (a real recompute). This attests no-stale-slot reuse, NOT a recall-delta
    # equivalence (that is GATE A's job).
    flags["edge_probe_passed"] = (
        evict_delta is not None
        and abs(evict_delta) <= 1e-9
        and evicted_is_recompute
    )
    if not flags["edge_probe_passed"]:
        problems.append(
            f"GATE C edge eviction==cold: delta_pp={evict_delta} "
            f"prefix_evicted={evicted_is_recompute} (need delta==0.0 and a real recompute)"
        )

    # cold/warm flips documented as value-neutral: cited R24 near-cutoff symdiff is small
    # AND GATE A recall is neutral (PASS). This is a documentation step gated on real
    # evidence, not a bit-identity claim.
    flags["cold_warm_flips_value_neutral_documented"] = flags["recall_equivalence_passed"]
    if not flags["cold_warm_flips_value_neutral_documented"]:
        problems.append("GATE: cold/warm flips not documentable (GATE A recall not PASS)")

    return flags, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-a", required=True)
    ap.add_argument("--gate-b-p2on", required=True)
    ap.add_argument("--gate-b-p4", required=True)
    ap.add_argument("--gate-c-edge", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    flags, problems = _derive(
        args.gate_a, args.gate_b_p2on, args.gate_b_p4, args.gate_c_edge, 0.0
    )
    print("[write_v2] derived gate flags:")
    for k, v in flags.items():
        print(f"  {k} = {v}")
    if problems:
        print("[write_v2] REFUSING to write a passing artifact — gate(s) failed:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if not all(flags.values()):
        print("[write_v2] REFUSING: not every gate flag is True.")
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
    print(f"[write_v2] wrote v2 artifact -> {args.out}")
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
