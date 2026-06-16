"""Mint the table-free radix-fixture state file (schema v2) IFF GATE A + B + C pass.

loop11b port of loop-11 R27 write_v2_artifact.py: same FAIL-CLOSED logic (re-derives each
pass flag from the RAW gate verdicts and REFUSES — returns non-zero, writes nothing —
unless every gate is a real PASS), but the ServerArgs it builds points double_sparsity_config
at the REGENERATED channel mask so radix_fixture_config_fingerprint pins
channel_mask_content_sha256 == a4be98c4c4989ea828b6ac128968af72336994b04cc1b6086408dbb208aa800d
(DEC-1: the fingerprint pins the tensor-CONTENT sha, not the file path / full-file sha).
Calls validator.write_radix_fixture_state with the EXACT serving ServerArgs for this op-point
so the config fingerprint matches the no-override boot.

  recall_equivalence_passed         <- GATE A recall verdict == PASS
  cross_rank_selection_identity     <- GATE B p2 radix_on == PASS (8 ranks byte-identical)
  no_dense_fallback_passed          <- GATE B p4 no_dense_fallback == PASS
  edge_probe_passed                 <- GATE C edge verdict status == PASS and NO problems
                                       (robust-n boundary/partial/eviction all clean within
                                        +/-0.5pp of cold + eviction==cold no-stale-slot)
  cold_warm_flips_value_neutral_documented
                                    <- documented: cited R24 symdiff + GATE A recall neutrality

After writing, prints the resulting fixture JSON AND the computed config fingerprint
(validator.radix_fixture_config_fingerprint(sa)) so the operator can confirm
channel_mask_content_sha256 == a4be98c4… before the no-override boot.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.layers.attention.double_sparsity import validator  # noqa: E402

GLM = "/cluster-storage/models/models--zai-org--GLM-5.1-FP8/snapshots/f396cf805182f4ca10fa675e1a99815b3ca384db"
# REGENERATED loop11b channel mask (loop8 DEC-3 recipe: fp8_e4m3, label_dim 32). DEC-1 pins
# the tensor-content sha, so the fingerprint computed below must be 35155ac4…; the path here
# is the durable cluster-storage copy.
MASK = "/cluster-storage/models/glm51-fp8-channel-mask-s256.safetensors"
EXPECTED_MASK_CONTENT_SHA256 = (
    "35155ac46ad79fa82e531138434ff35708e2d8c2932889323a21a455342a9b00"
)
DS_CFG = (
    '{"top_k": 2048, "page_size": 64, '
    '"channel_mask_path": "' + MASK + '", '
    '"device_buffer_size": 4096, "scorer_norm": "off", "head_agg": "max", '
    '"anchor_mode": "off", "anchor_budget": 0, "enable_lifted_budget_decode": false, '
    '"lifted_budget_top_k": 0}'
)


def _build_server_args() -> ServerArgs:
    # EXACT serving ServerArgs for this op-point so radix_fixture_config_fingerprint is
    # computed exactly as at boot (GLM TP=8 fp8_e4m3 page64 mem0.8 max-running64) with the
    # REGENERATED mask path.
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


def _passing_verdict(path, label, problems):
    """Return True iff the verdict file's own status == PASS and it records no problems.
    Fail-closed: a missing file, a non-PASS status, or any recorded problems => not a pass.
    """
    try:
        with open(path) as fh:
            v = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{label}: unreadable verdict {path!r}: {exc}")
        return False, {}
    # Accept either "status" (GATE B/C) or "verdict" (GATE A) as the verdict key;
    # "status" takes precedence so a verdict carrying both cannot fail-open.
    verdict = v.get("status") if v.get("status") is not None else v.get("verdict")
    ok = verdict == "PASS" and not v.get("problems")
    if not ok:
        problems.append(
            f"{label}: status={v.get('status')!r} problems={v.get('problems')!r}"
        )
    return ok, v


def _derive(gate_a, gate_b_p2on, gate_b_p4, gate_c_edge):
    problems = []
    flags = {}

    # GATE A recall equivalence: the verdict's own status is authoritative.
    a_ok, _ = _passing_verdict(gate_a, "GATE A recall", problems)
    flags["recall_equivalence_passed"] = a_ok

    # GATE B cross-rank identity (radix-on): PASS + all 8 ranks byte-identical.
    b2_ok, p2 = _passing_verdict(gate_b_p2on, "GATE B cross-rank", problems)
    flags["cross_rank_selection_identity_passed"] = (
        b2_ok
        and p2.get("all_ranks_identical_all_steps") is True
        and p2.get("ranks_present") == list(range(8))
    )
    if b2_ok and not flags["cross_rank_selection_identity_passed"]:
        problems.append(
            f"GATE B cross-rank coverage: identical={p2.get('all_ranks_identical_all_steps')} "
            f"ranks={p2.get('ranks_present')}"
        )

    # GATE B no-dense-fallback: PASS + zero violations.
    b4_ok, p4 = _passing_verdict(gate_b_p4, "GATE B no_dense_fallback", problems)
    flags["no_dense_fallback_passed"] = b4_ok and p4.get("num_violations") == 0
    if b4_ok and not flags["no_dense_fallback_passed"]:
        problems.append(
            f"GATE B no_dense_fallback violations={p4.get('num_violations')}"
        )

    # GATE C edge: FAIL-CLOSED on the edge verdict's OWN status + problems. The robust-n
    # edge driver sets status=PASS only when boundary/partial/eviction are all clean within
    # +/-0.5pp of cold AND eviction->recompute == cold (no stale slot). We do NOT derive a
    # pass from any single subcase while the verdict is FAIL — that was the R25 fail-open.
    c_ok, _ = _passing_verdict(gate_c_edge, "GATE C edge", problems)
    flags["edge_probe_passed"] = c_ok

    # cold/warm flips documented value-neutral: gated on GATE A recall neutrality (PASS).
    flags["cold_warm_flips_value_neutral_documented"] = flags[
        "recall_equivalence_passed"
    ]
    if not flags["cold_warm_flips_value_neutral_documented"]:
        problems.append("cold/warm flips not documentable (GATE A recall not PASS)")

    return flags, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-a", required=True)
    ap.add_argument("--gate-b-p2on", required=True)
    ap.add_argument("--gate-b-p4", required=True)
    ap.add_argument("--gate-c-edge", required=True)
    ap.add_argument(
        "--out",
        default="/sgl-workspace/sglang/development/serve_double_sparsity_radix_fixture.json",
    )
    args = ap.parse_args()

    flags, problems = _derive(
        args.gate_a, args.gate_b_p2on, args.gate_b_p4, args.gate_c_edge
    )
    print("[mint_fixture] derived gate flags:")
    for k, v in flags.items():
        print(f"  {k} = {v}")
    if problems:
        print("[mint_fixture] REFUSING to write a passing artifact — gate(s) failed:")
        for p in problems:
            print(f"  - {p}")
        return 2
    if not all(flags.values()):
        print("[mint_fixture] REFUSING: not every gate flag is True.")
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
    print(f"[mint_fixture] wrote v2 artifact -> {args.out}")
    print(json.dumps(state, indent=2))

    # Print the computed config fingerprint so the operator can confirm the REGENERATED
    # mask's tensor-content sha is what the no-override boot will be authorized against.
    fp = validator.radix_fixture_config_fingerprint(sa)
    print("[mint_fixture] computed config fingerprint:")
    print(json.dumps(fp, indent=2))
    got = fp.get("channel_mask_content_sha256")
    match = got == EXPECTED_MASK_CONTENT_SHA256
    print(
        f"[mint_fixture] channel_mask_content_sha256={got} "
        f"expected={EXPECTED_MASK_CONTENT_SHA256} match={match}"
    )
    if not match:
        print(
            "[mint_fixture] !! WARNING: minted fingerprint does NOT match the expected "
            "REGENERATED-mask content sha — the no-override boot will NOT be authorized."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
