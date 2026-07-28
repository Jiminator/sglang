"""Calibrate the recovery-agent profiles' turns/context correlation.

For each profile, sweeps the Gaussian-copula correlation between turn count
and final context and reports the value whose request-weighted mean prompt
size (over the profile's reference population) lands closest to the target.
The chosen constants are baked into ``PROFILES`` in
``python/sglang/benchmark/datasets/recovery_agent.py``; rerun this script and
update them whenever any other profile constant changes.

The sweep uses the same planning arithmetic as the dataset builder (no
tokenizer needed — prompt sizes are planned bare-token arithmetic with a
nominal per-turn template overhead).
"""

import argparse

import msgspec
import numpy as np

from sglang.benchmark.datasets.recovery_agent import (
    PROFILES,
    SessionProfile,
    _plan_session,
)

NOMINAL_TEMPLATE_OVERHEAD = 8


def request_weighted_isl_mean(
    profile: SessionProfile, seed: int, num_sessions: int
) -> float:
    prompt_lens = []
    for session_index in range(num_sessions):
        rng = np.random.RandomState(seed + session_index)
        plan = _plan_session(profile, rng, NOMINAL_TEMPLATE_OVERHEAD)
        prompt = profile.head_tokens + NOMINAL_TEMPLATE_OVERHEAD
        for turn_index, turn_input in enumerate(plan.turn_inputs):
            if turn_index == 0:
                prompt += turn_input
            else:
                prompt += (
                    turn_input + profile.output_len_per_turn + NOMINAL_TEMPLATE_OVERHEAD
                )
            prompt_lens.append(prompt)
    return float(np.mean(prompt_lens))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name, profile in PROFILES.items():
        target = profile.request_weighted_isl_target
        # Any correlation whose deviation is within tolerance is admissible;
        # the baked value may trade a little deviation for structural realism
        # (deeper sessions must trend toward bigger contexts, so only
        # non-negative correlations are considered).
        baked_isl = request_weighted_isl_mean(
            profile, seed=args.seed, num_sessions=profile.reference_population
        )
        baked_dev = abs(baked_isl - target) / target
        status = "OK" if baked_dev <= profile.request_weighted_isl_tolerance else "OUT"
        print(
            f"{name}: baked rho={profile.turns_context_correlation:+.2f} "
            f"isl_mean={baked_isl:,.0f} target={target:,} "
            f"deviation={baked_dev:.1%} [{status}]"
        )
        for rho in np.arange(0.0, 0.96, 0.05):
            candidate = msgspec.structs.replace(
                profile, turns_context_correlation=round(float(rho), 2)
            )
            isl = request_weighted_isl_mean(
                candidate, seed=args.seed, num_sessions=profile.reference_population
            )
            deviation = abs(isl - target) / target
            marker = (
                "ok " if deviation <= profile.request_weighted_isl_tolerance else "out"
            )
            print(
                f"  sweep rho={rho:+.2f} isl_mean={isl:,.0f} "
                f"deviation={deviation:.1%} [{marker}]"
            )


if __name__ == "__main__":
    main()
