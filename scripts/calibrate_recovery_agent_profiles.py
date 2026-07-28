"""Verify (and help recalibrate) the recovery-agent profile constants.

For each profile and each generation authority, replays the canonical
reference population with the profile's recorded calibration overheads (or
overheads measured from a real tokenizer via ``--tokenizer``) and reports
every retained target dimension: realized value, deviation, and — for the
dimensions the authority gates — whether it is within the profile's declared
tolerance. Exits nonzero if any baked gated dimension fails, so this script
doubles as a calibration gate.

The baked constants live in ``PROFILES`` in
``python/sglang/benchmark/datasets/recovery_agent.py``; rerun this script and
update them whenever any constant or the reference overheads change.
"""

import argparse
import sys

import msgspec
import numpy as np

from sglang.benchmark.datasets.recovery_agent import (
    AUTHORITY_CONTEXT,
    AUTHORITY_INPUT,
    PROFILES,
    SessionProfile,
    _measure_template_overheads,
    _plan_session,
    _realized_stats,
)


def evaluate(
    profile: SessionProfile,
    seed: int,
    initial_overhead: int,
    round_overhead: int,
) -> dict:
    plans = [
        _plan_session(
            profile,
            np.random.RandomState(seed + i),
            initial_overhead,
            round_overhead,
        )
        for i in range(profile.reference_population)
    ]
    return _realized_stats(plans, profile, initial_overhead, round_overhead)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Tokenizer path to measure real chat-template overheads from; "
        "default uses each profile's recorded calibration overheads.",
    )
    args = parser.parse_args()

    measured_overheads = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
        measured_overheads = _measure_template_overheads(tokenizer)
        print(f"measured overheads from {args.tokenizer}: {measured_overheads}")

    all_ok = True
    for name, baked in PROFILES.items():
        overheads = measured_overheads or baked.calibration_overheads
        if measured_overheads and tuple(measured_overheads) != tuple(
            baked.calibration_overheads
        ):
            print(
                f"WARNING: {name} was calibrated at overheads "
                f"{tuple(baked.calibration_overheads)}, measuring at "
                f"{tuple(measured_overheads)}."
            )
        for authority in (AUTHORITY_CONTEXT, AUTHORITY_INPUT):
            profile = msgspec.structs.replace(baked, authority=authority)
            realized = evaluate(profile, args.seed, *overheads)
            status = "OK" if realized["gates_within_tolerance"] else "FAIL"
            print(
                f"\n{name} [{authority}-authoritative] "
                f"population={realized['sessions']} "
                f"repairs={realized['repaired_sessions']} gates={status}"
            )
            for dimension, entry in realized["dimensions"].items():
                if entry["gated"]:
                    marker = "ok " if entry["within_tolerance"] else "FAIL"
                    marker = f"GATE {marker}"
                else:
                    marker = "reported"
                print(
                    f"  {dimension:28s} target={entry['target']:>9.1f} "
                    f"realized={entry['realized']:>9.1f} "
                    f"dev={entry['deviation_frac']:+.3f} [{marker}]"
                )
            all_ok &= realized["gates_within_tolerance"]

    if not all_ok:
        print("\nCalibration gates FAILED for at least one profile/authority.")
        sys.exit(1)
    print("\nAll calibration gates passed.")


if __name__ == "__main__":
    main()
