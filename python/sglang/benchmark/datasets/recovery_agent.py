"""Recovery-agent multi-turn benchmark dataset.

Synthesizes agentic debugging/patching sessions whose *shape* (turns per
session, per-turn new input, final context, per-turn output length) follows
one of two built-in statistical profiles. The data is distribution-matched
synthetic text — it replays no recorded conversations — built so KV-cache
behavior under session-sticky routing is a controlled property of the
benchmark:

- every session shares one fixed head (system prompt + tool-schema-shaped
  pad), the cacheable prefix a live agent fleet presents;
- immediately after the head each session diverges into unique synthetic
  content, so cross-session cache reuse ends at the head;
- each turn appends a sized user message and the server's real reply, and the
  full cumulative history is resent every turn.

Sessions are sampled from lognormal marginals for turn count and final
context (coupled through a Gaussian copula) and a lognormal per-turn input
size, with published caps, and per-session input budgets allocated exactly.
Builds are deterministic per (profile, seed, tokenizer) and cached as JSON
(``{"metadata": {...}, "conversations": [[{messages, prompt_tokens}, ...]]}``);
a prebuilt file loads unchanged via ``--dataset-path``.

Use with a chat backend (``--backend sglang-oai-chat``) and, for
session-sticky routing, an sgl-router policy that consumes
``X-SMG-Routing-Key`` (each session carries a stable unique key).
"""

import hashlib
import json
import math
import os
import string
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec
import numpy as np

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

GENERATOR_VERSION = 3

# z-score of the 95th percentile of the standard normal; lognormal fits below
# derive sigma from the (p50, p95) pair.
_Z95 = 1.6448536269514722

# A sized text never exceeds its bare-token target and may fall short by at
# most this many tokens (decode -> re-encode can merge tokens at slice
# boundaries on byte-fallback tokenizers).
PAD_SIZING_MAX_DEFICIT_TOKENS = 8

# Smallest user message worth sending on a turn, in bare tokens.
MIN_TURN_INPUT_TOKENS = 16

# Bounded deterministic redraws before an infeasible (turns, context) draw is
# repaired by shrinking the turn count or clamping the context.
MAX_FEASIBILITY_REDRAWS = 8

_PAD_ALPHABET = np.array(list(string.ascii_letters + string.digits + " .,;:?!\n"))


# Which statistical dimension the generator treats as exact.
#
# The published per-turn-input and final-context summaries of the measured
# lineages are arithmetically incompatible (mean turns x mean input plus
# replies exceeds the mean final context by ~30%), so a generator can honor
# one of them, not both. Per explicit user decision (recorded in the loop's
# plan-evolution log): context is authoritative by default, and an
# input-authoritative mode is provided so the authority can be flipped with
# one switch (``--recovery-authority input``).
AUTHORITY_CONTEXT = "context"
AUTHORITY_INPUT = "input"

# Realized-vs-target statistics keys, grouped by which generation authority
# gates them. The non-gated dimensions are always reported, never asserted.
GATED_DIMENSIONS = {
    AUTHORITY_CONTEXT: (
        "turns_mean",
        "turns_p50",
        "turns_p95",
        "final_context_mean",
        "final_context_p50",
        "final_context_p95",
        "final_context_max",
        "request_weighted_isl_mean",
    ),
    AUTHORITY_INPUT: (
        "turns_mean",
        "turns_p50",
        "turns_p95",
        "input_per_turn_mean",
        "input_per_turn_p50",
        "input_per_turn_p95",
        "input_per_turn_max",
    ),
}

# Maxima are one-sided upper bounds against the published maxima; the other
# dimensions are two-sided sample statistics with per-dimension tolerances.
UPPER_BOUND_DIMENSIONS = ("input_per_turn_max", "final_context_max")


class SessionProfile(msgspec.Struct, frozen=True):
    """Statistical shape of one workload lineage.

    All token counts are bare tokens unless noted. ``final context`` is the
    rendered prompt of a session's last request: fixed head, all per-turn user
    input, the assistant replies of every turn but the last, and chat-template
    overhead.

    Every published summary the profile retains is stored as an explicit
    target with a per-dimension relative tolerance; ``authority`` selects
    which dimension set is generated exactly (and therefore gated) versus
    derived (and therefore reported only). The calibration constants are
    valid for the recorded reference chat-template overheads.
    """

    name: str
    authority: str  # AUTHORITY_CONTEXT or AUTHORITY_INPUT
    # Turn-count targets (always gated).
    turns_mean: float
    turns_p50: float
    turns_p95: float
    turns_cap: int  # p95-derived cap (2x p95), keeps the sampled tail finite
    turns_tolerance: float
    # Per-turn new-input targets (gated in input-authoritative mode).
    input_per_turn_mean: float
    input_per_turn_p50: float
    input_per_turn_p95: float
    input_per_turn_cap: int  # operational clamp for sampled inputs
    input_per_turn_max_target: float  # published per-turn new-input maximum
    input_per_turn_tolerance: float
    # Final-context targets (gated in context-authoritative mode).
    final_context_mean: float
    final_context_p50: float
    final_context_p95: float
    final_context_cap: int  # operational clamp for sampled contexts
    final_context_max_target: float  # published final-context maximum
    final_context_tolerance: float
    output_len_per_turn: int  # fixed planned per-turn completion budget
    head_tokens: int  # fixed shared system + tool-schema-shaped head
    # Gaussian-copula correlation between turn count and final context
    # (context mode only), calibrated once on the reference population so the
    # request-weighted mean prompt size lands on target.
    turns_context_correlation: float
    request_weighted_isl_target: int
    request_weighted_isl_tolerance: float
    reference_population: int  # population size the calibration ran at
    # Chat-template overheads (initial, per-round) the calibration used;
    # measured on the GLM-5.2 tokenizer. Conformance tests replay these.
    calibration_overheads: Tuple[int, int]

    def gated_dimensions(self) -> Tuple[str, ...]:
        return GATED_DIMENSIONS[self.authority]

    def dimension_tolerance(self, dimension: str) -> float:
        # Published maxima are hard upper bounds, not sample statistics.
        if dimension in UPPER_BOUND_DIMENSIONS:
            return 0.0
        if dimension.startswith("turns_"):
            return self.turns_tolerance
        if dimension.startswith("input_per_turn_"):
            return self.input_per_turn_tolerance
        if dimension.startswith("final_context_"):
            return self.final_context_tolerance
        return self.request_weighted_isl_tolerance

    def dimension_target(self, dimension: str) -> float:
        if dimension == "request_weighted_isl_mean":
            return float(self.request_weighted_isl_target)
        if dimension in UPPER_BOUND_DIMENSIONS:
            return float(getattr(self, f"{dimension}_target"))
        return float(getattr(self, dimension))


# The two lineages, fitted to their published summaries. The correlation
# constants come from scripts/calibrate_recovery_agent_profiles.py runs at
# the reference population sizes with the recorded calibration overheads;
# rerun it if any other constant here changes.
PROFILES: Dict[str, SessionProfile] = {
    "agent-long": SessionProfile(
        name="agent-long",
        authority=AUTHORITY_CONTEXT,
        turns_mean=47.0,
        turns_p50=41.0,
        turns_p95=96.0,
        turns_cap=192,
        turns_tolerance=0.15,
        input_per_turn_mean=514.0,
        input_per_turn_p50=265.0,
        input_per_turn_p95=1700.0,
        input_per_turn_cap=17000,
        input_per_turn_max_target=17000.0,
        input_per_turn_tolerance=0.15,
        final_context_mean=27800.0,
        final_context_p50=24800.0,
        final_context_p95=53800.0,
        final_context_cap=92000,
        final_context_max_target=92000.0,
        final_context_tolerance=0.15,
        output_len_per_turn=190,
        head_tokens=3000,
        turns_context_correlation=0.30,
        request_weighted_isl_target=20900,
        request_weighted_isl_tolerance=0.10,
        reference_population=512,
        calibration_overheads=(13, 4),
    ),
    "agent-short": SessionProfile(
        name="agent-short",
        authority=AUTHORITY_CONTEXT,
        turns_mean=16.0,
        turns_p50=10.0,
        turns_p95=50.0,
        turns_cap=100,
        turns_tolerance=0.15,
        input_per_turn_mean=730.0,
        input_per_turn_p50=340.0,
        input_per_turn_p95=2600.0,
        input_per_turn_cap=31000,
        input_per_turn_max_target=31000.0,
        input_per_turn_tolerance=0.15,
        final_context_mean=14800.0,
        final_context_p50=13000.0,
        final_context_p95=31300.0,
        final_context_cap=101000,
        final_context_max_target=101000.0,
        final_context_tolerance=0.15,
        output_len_per_turn=185,
        head_tokens=3000,
        turns_context_correlation=0.65,
        request_weighted_isl_target=14800,
        request_weighted_isl_tolerance=0.10,
        reference_population=2048,
        calibration_overheads=(13, 4),
    ),
    # Tiny sessions for topology/smoke tests with small models: same
    # generator, sizes that fit a 32k-context model in seconds. Not derived
    # from any measured workload; wide tolerances, gates informational.
    "smoke": SessionProfile(
        name="smoke",
        authority=AUTHORITY_CONTEXT,
        turns_mean=4.6,
        turns_p50=4.0,
        turns_p95=8.0,
        turns_cap=16,
        turns_tolerance=0.30,
        input_per_turn_mean=100.0,
        input_per_turn_p50=60.0,
        input_per_turn_p95=200.0,
        input_per_turn_cap=1000,
        input_per_turn_max_target=1000.0,
        input_per_turn_tolerance=0.50,
        final_context_mean=980.0,
        final_context_p50=800.0,
        final_context_p95=2000.0,
        final_context_cap=4000,
        final_context_max_target=4000.0,
        final_context_tolerance=0.30,
        output_len_per_turn=32,
        head_tokens=200,
        turns_context_correlation=0.5,
        request_weighted_isl_target=808,
        request_weighted_isl_tolerance=0.30,
        reference_population=64,
        calibration_overheads=(13, 4),
    ),
}

DEFAULT_PROFILE = "agent-short"


def _lognormal_params(p50: float, p95: float) -> Tuple[float, float]:
    """(mu, sigma) of the lognormal with the given p50 and p95."""
    mu = math.log(p50)
    sigma = (math.log(p95) - math.log(p50)) / _Z95
    return mu, sigma


class SessionPlan(msgspec.Struct):
    """Arithmetic plan of one session, before text realization."""

    turn_count: int
    final_context: int
    input_budget: int
    turn_inputs: List[int]
    repaired: bool


def _plan_session(
    profile: SessionProfile,
    rng: np.random.RandomState,
    initial_overhead: int,
    round_overhead: int,
) -> SessionPlan:
    """Sample one session's (turns, context, per-turn inputs) plan.

    Context-authoritative profiles sample the final context exactly and
    allocate per-turn inputs to its implied budget; input-authoritative
    profiles sample per-turn inputs exactly and derive the final context.

    ``initial_overhead`` is the chat-template cost of rendering the opening
    system+user pair; ``round_overhead`` is the cost each later round adds
    (one assistant reply plus one user message wrapper).
    """
    if profile.authority == AUTHORITY_INPUT:
        return _plan_session_input_authoritative(
            profile, rng, initial_overhead, round_overhead
        )
    turns_mu, turns_sigma = _lognormal_params(profile.turns_p50, profile.turns_p95)
    ctx_mu, ctx_sigma = _lognormal_params(
        profile.final_context_p50, profile.final_context_p95
    )
    rho = profile.turns_context_correlation
    osl = profile.output_len_per_turn
    per_extra_turn = MIN_TURN_INPUT_TOKENS + osl + round_overhead

    repaired = False
    turn_count = 0
    final_context = 0
    budget = -1
    for _ in range(MAX_FEASIBILITY_REDRAWS):
        z_turns = rng.standard_normal()
        z_ctx = rho * z_turns + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
        turn_count = int(
            np.clip(
                round(math.exp(turns_mu + turns_sigma * z_turns)),
                1,
                profile.turns_cap,
            )
        )
        final_context = int(
            np.clip(
                round(math.exp(ctx_mu + ctx_sigma * z_ctx)),
                1,
                profile.final_context_cap,
            )
        )
        budget = _input_budget(
            profile, turn_count, final_context, initial_overhead, round_overhead
        )
        if budget >= turn_count * MIN_TURN_INPUT_TOKENS:
            break
    else:
        # Persistent low draw: shrink the turn count to what the sampled
        # context affords (the final turn's reply never enters a prompt,
        # hence the +osl in the numerator).
        repaired = True
        turn_count = max(
            1,
            (
                final_context
                - profile.head_tokens
                - initial_overhead
                - MIN_TURN_INPUT_TOKENS
                + per_extra_turn
            )
            // per_extra_turn,
        )
        budget = _input_budget(
            profile, turn_count, final_context, initial_overhead, round_overhead
        )
        if budget < turn_count * MIN_TURN_INPUT_TOKENS:
            # Even one turn does not fit: grow the context to the floor.
            final_context = (
                profile.head_tokens
                + initial_overhead
                + turn_count * MIN_TURN_INPUT_TOKENS
                + (turn_count - 1) * (osl + round_overhead)
            )
            budget = turn_count * MIN_TURN_INPUT_TOKENS

    max_budget = turn_count * profile.input_per_turn_cap
    if budget > max_budget:
        # Excessive context for so few turns: clamp the context down to the
        # largest value the per-turn cap can realize.
        repaired = True
        budget = max_budget
        final_context = (
            profile.head_tokens
            + initial_overhead
            + budget
            + (turn_count - 1) * (osl + round_overhead)
        )

    turn_inputs = _allocate_turn_inputs(profile, rng, turn_count, budget)
    # Front-load the large tool dumps: after the opening issue text, order
    # the remaining inputs descending. An agent reads files and test logs
    # early, then iterates with small deltas — this reproduces the published
    # ratio of mean live context to final context, and only reorders the
    # sampled sizes (their distribution is unchanged).
    turn_inputs = [turn_inputs[0]] + sorted(turn_inputs[1:], reverse=True)
    return SessionPlan(
        turn_count=turn_count,
        final_context=final_context,
        input_budget=budget,
        turn_inputs=turn_inputs,
        repaired=repaired,
    )


def _plan_session_input_authoritative(
    profile: SessionProfile,
    rng: np.random.RandomState,
    initial_overhead: int,
    round_overhead: int,
) -> SessionPlan:
    """Sample per-turn inputs directly from the fitted marginal and derive
    the final context (input-authoritative mode).

    The input marginal is exact up to the published floor/cap clamps; the
    final context is whatever the sampled inputs imply and is reported, not
    gated. A derived context beyond twice the published maximum is redrawn a
    bounded number of times, then trailing turns are dropped (a repair).
    """
    turns_mu, turns_sigma = _lognormal_params(profile.turns_p50, profile.turns_p95)
    d_mu, d_sigma = _lognormal_params(
        profile.input_per_turn_p50, profile.input_per_turn_p95
    )
    osl = profile.output_len_per_turn
    context_safety_cap = 2 * profile.final_context_cap

    def derived_context(inputs: List[int]) -> int:
        return (
            profile.head_tokens
            + initial_overhead
            + sum(inputs)
            + (len(inputs) - 1) * (osl + round_overhead)
        )

    turn_count = int(
        np.clip(
            round(math.exp(turns_mu + turns_sigma * rng.standard_normal())),
            1,
            profile.turns_cap,
        )
    )
    repaired = False
    turn_inputs: List[int] = []
    for _ in range(MAX_FEASIBILITY_REDRAWS):
        raw = np.exp(d_mu + d_sigma * rng.standard_normal(turn_count))
        turn_inputs = [
            int(np.clip(round(d), MIN_TURN_INPUT_TOKENS, profile.input_per_turn_cap))
            for d in raw
        ]
        if derived_context(turn_inputs) <= context_safety_cap:
            break
    else:
        repaired = True
        while (
            len(turn_inputs) > 1 and derived_context(turn_inputs) > context_safety_cap
        ):
            turn_inputs = turn_inputs[:-1]
        turn_count = len(turn_inputs)

    turn_inputs = [turn_inputs[0]] + sorted(turn_inputs[1:], reverse=True)
    return SessionPlan(
        turn_count=turn_count,
        final_context=derived_context(turn_inputs),
        input_budget=sum(turn_inputs),
        turn_inputs=turn_inputs,
        repaired=repaired,
    )


def _input_budget(
    profile: SessionProfile,
    turn_count: int,
    final_context: int,
    initial_overhead: int,
    round_overhead: int,
) -> int:
    return (
        final_context
        - profile.head_tokens
        - initial_overhead
        - (turn_count - 1) * (profile.output_len_per_turn + round_overhead)
    )


def _allocate_turn_inputs(
    profile: SessionProfile,
    rng: np.random.RandomState,
    turn_count: int,
    budget: int,
) -> List[int]:
    """Split ``budget`` into per-turn input sizes summing to it exactly.

    Draws iid lognormal sizes and rescales them to the budget (a pure shift
    in log space, preserving the fitted shape), then floors, caps, and
    redistributes the residual deterministically.

    The caller guarantees ``turn_count * MIN <= budget <= turn_count * cap``.
    """
    d_mu, d_sigma = _lognormal_params(
        profile.input_per_turn_p50, profile.input_per_turn_p95
    )
    raw = np.exp(d_mu + d_sigma * rng.standard_normal(turn_count))
    scaled = raw * (budget / raw.sum())

    sizes = np.floor(scaled).astype(int)
    # Largest-remainder rounding keeps the sum exact before clamping.
    remainder = budget - int(sizes.sum())
    if remainder > 0:
        by_fraction = np.argsort(-(scaled - sizes), kind="stable")
        sizes[by_fraction[:remainder]] += 1

    sizes = np.clip(sizes, MIN_TURN_INPUT_TOKENS, profile.input_per_turn_cap)
    # Clamping broke the sum; hand the residual around round-robin, one
    # token-bounded step per turn with headroom, until it is zero.
    residual = budget - int(sizes.sum())
    while residual != 0:
        if residual > 0:
            room = profile.input_per_turn_cap - sizes
        else:
            room = sizes - MIN_TURN_INPUT_TOKENS
        movable = np.nonzero(room > 0)[0]
        step = int(math.ceil(abs(residual) / len(movable)))
        for turn in movable:
            move = int(min(step, room[turn], abs(residual)))
            sizes[turn] += move if residual > 0 else -move
            residual -= move if residual > 0 else -move
            if residual == 0:
                break
    return [int(s) for s in sizes]


def _measure_template_overheads(tokenizer) -> Tuple[int, int]:
    """Chat-template token overheads, measured on this tokenizer.

    Returns ``(initial_overhead, round_overhead)``: the wrapper cost of
    rendering the opening system+user pair, and the cost each later round
    adds on top (one assistant reply plus one user message). Tokenizers
    without a chat template cost nothing.
    """
    if tokenizer.chat_template is None:
        return 0, 0
    base = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    extended = base + [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "v"},
    ]

    def rendered_len(messages):
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        )

    def bare_len(messages):
        return sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            for m in messages
        )

    initial_overhead = max(0, rendered_len(base) - bare_len(base))
    round_overhead = max(
        0,
        rendered_len(extended) - rendered_len(base) - bare_len(extended[len(base) :]),
    )
    return initial_overhead, round_overhead


def _sized_text(
    target_bare_tokens: int, rng: np.random.RandomState, tokenizer
) -> Tuple[str, int]:
    """Deterministic text encoding to at most ``target_bare_tokens`` bare
    tokens, short by at most ``PAD_SIZING_MAX_DEFICIT_TOKENS``. Returns the
    text and its actual encoded length."""
    if target_bare_tokens <= 0:
        return "", 0
    n_chars = int(target_bare_tokens * 1.7) + 8
    idx = rng.randint(0, len(_PAD_ALPHABET), size=n_chars)
    text = "".join(_PAD_ALPHABET[idx].tolist())
    trimmed, realized = _truncate_to_bare_tokens(text, target_bare_tokens, tokenizer)
    while realized < target_bare_tokens - PAD_SIZING_MAX_DEFICIT_TOKENS:
        extra_target = target_bare_tokens - realized
        idx = rng.randint(0, len(_PAD_ALPHABET), size=int(extra_target * 1.7) + 8)
        extra = "".join(_PAD_ALPHABET[idx].tolist())
        candidate = trimmed + extra
        trimmed, realized = _truncate_to_bare_tokens(
            candidate, target_bare_tokens, tokenizer
        )
    return trimmed, realized


def _truncate_to_bare_tokens(
    text: str, target_bare_tokens: int, tokenizer
) -> Tuple[str, int]:
    """Trim ``text`` to at most ``target_bare_tokens`` bare tokens and return
    ``(text, realized_len)``.

    Byte-fallback tokenizers can re-encode a decoded slice to a different
    count, so verify and re-trim until the realized length is at most the
    target.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= target_bare_tokens:
        return text, len(ids)
    slice_len = target_bare_tokens
    while slice_len > 0:
        candidate = tokenizer.decode(ids[:slice_len], skip_special_tokens=True)
        realized = len(tokenizer.encode(candidate, add_special_tokens=False))
        if realized <= target_bare_tokens:
            return candidate, realized
        slice_len -= realized - target_bare_tokens
    return "", 0


def _session_routing_key(seed: int, session_index: int) -> str:
    digest = hashlib.sha1(f"{seed}:{session_index}".encode("utf-8")).hexdigest()
    return f"session-{digest[:16]}"


# Tokenizer attributes that change encode/decode/render behavior for the
# sizing and template-measurement paths, hashed into the fingerprint.
_TOKENIZER_BEHAVIOR_ATTRS = (
    "clean_up_tokenization_spaces",
    "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output",
    "split_special_tokens",
    "add_prefix_space",
    "do_lower_case",
    "add_bos_token",
    "add_eos_token",
)


def _tokenizer_fingerprint(tokenizer) -> str:
    """Full-SHA-256 digest of one canonical serialization of everything about
    the tokenizer that shapes generated text or measured overheads: the
    serialized fast-backend state (vocabulary, merges, normalizer,
    pre-tokenizer, special tokens) or the slow tokenizer's asset files, the
    chat template, special/added-token maps, the tokenizer class, and every
    encode/decode/render behavior setting used by the sizing paths. A change
    to any of these at an unchanged ``name_or_path`` must invalidate cached
    builds and their overhead measurements.
    """
    from transformers import PreTrainedTokenizerFast

    envelope: Dict[str, Any] = {
        "class": type(tokenizer).__name__,
        "chat_template": tokenizer.chat_template,
        "special_tokens_map": {
            k: str(v) for k, v in sorted(tokenizer.special_tokens_map.items())
        },
        "added_vocab": dict(sorted(tokenizer.get_added_vocab().items())),
        "init_kwargs": {
            k: v
            for k, v in sorted(tokenizer.init_kwargs.items())
            if isinstance(v, (str, int, float, bool, type(None)))
        },
        # These attributes legitimately exist only on some tokenizer classes;
        # absent ones hash as None.
        "behavior": {
            attr: getattr(tokenizer, attr, None) for attr in _TOKENIZER_BEHAVIOR_ATTRS
        },
    }
    if isinstance(tokenizer, PreTrainedTokenizerFast):
        envelope["backend"] = tokenizer.backend_tokenizer.to_str()
    else:
        # Slow tokenizers: hash the actual asset contents (vocabulary,
        # merges / SentencePiece model, ...) referenced by the instance, plus
        # the realized vocabulary as a fallback.
        assets = {}
        for asset_name in sorted(tokenizer.vocab_files_names):
            path = tokenizer.init_kwargs.get(asset_name)
            if isinstance(path, str) and os.path.isfile(path):
                with open(path, "rb") as f:
                    assets[asset_name] = hashlib.sha256(f.read()).hexdigest()
        envelope["assets"] = assets
        envelope["vocab"] = dict(sorted(tokenizer.get_vocab().items()))
    canonical = json.dumps(envelope, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_sessions(
    tokenizer,
    *,
    profile: SessionProfile,
    seed: int,
    num_sessions: int,
) -> Dict[str, Any]:
    """Build ``num_sessions`` synthetic sessions in the prebuilt JSON schema.

    Each session's randomness comes from its own ``RandomState(seed + index)``
    stream, so any prefix of a larger build is byte-identical to a smaller
    build — growing a cache never changes existing sessions.
    """
    initial_overhead, round_overhead = _measure_template_overheads(tokenizer)
    head_text, built_head = _sized_text(
        profile.head_tokens, np.random.RandomState(seed), tokenizer
    )

    conversations: List[List[Dict[str, Any]]] = []
    plans: List[SessionPlan] = []
    built_plans: List[SessionPlan] = []
    for session_index in range(num_sessions):
        rng = np.random.RandomState(seed + session_index)
        plan = _plan_session(profile, rng, initial_overhead, round_overhead)
        plans.append(plan)

        turns: List[Dict[str, Any]] = []
        built_inputs: List[int] = []
        prompt_tokens = built_head + initial_overhead
        for turn_index, turn_input in enumerate(plan.turn_inputs):
            content, built_input = _sized_text(turn_input, rng, tokenizer)
            built_inputs.append(built_input)
            if turn_index == 0:
                messages = [
                    {"role": "system", "content": head_text},
                    {"role": "user", "content": content},
                ]
                prompt_tokens += built_input
            else:
                messages = [{"role": "user", "content": content}]
                prompt_tokens += (
                    built_input + profile.output_len_per_turn + round_overhead
                )
            turns.append({"messages": messages, "prompt_tokens": prompt_tokens})
        conversations.append(turns)
        # The same arithmetic over the *actual encoded* text sizes: what the
        # built dataset physically contains (pad sizing may undershoot each
        # planned size by a bounded deficit).
        built_plans.append(
            SessionPlan(
                turn_count=plan.turn_count,
                final_context=prompt_tokens,
                input_budget=sum(built_inputs),
                turn_inputs=built_inputs,
                repaired=plan.repaired,
            )
        )

    metadata = {
        "dataset": "recovery-agent",
        "generator_version": GENERATOR_VERSION,
        "profile": msgspec.to_builtins(profile),
        "seed": seed,
        "tokenizer_path": tokenizer.name_or_path,
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        "initial_overhead": initial_overhead,
        "round_overhead": round_overhead,
        "built_head_tokens": built_head,
        "num_sessions": num_sessions,
        "synthetic": True,  # distribution-matched synthetic, not a trace replay
        # Per-session arithmetic, retained so any slice of the build can
        # report its own statistics: the sampled plan and the actual encoded
        # ("built") sizes.
        "session_stats": [msgspec.to_builtins(p) for p in plans],
        "built_session_stats": [msgspec.to_builtins(p) for p in built_plans],
        # planned_stats gates the calibrated sampling arithmetic; built_stats
        # reports what the encoded text realizes. Neither claims the
        # server-defined (usage-reported) context — that is the separate live
        # conformance report emitted after a benchmark run.
        "planned_stats": _realized_stats(
            plans, profile, initial_overhead, round_overhead
        ),
        "built_stats": _realized_stats(
            built_plans,
            profile,
            initial_overhead,
            round_overhead,
            head_tokens=built_head,
        ),
    }
    return {"metadata": metadata, "conversations": conversations}


def _realized_stats(
    plans: List[SessionPlan],
    profile: SessionProfile,
    initial_overhead: int,
    round_overhead: int,
    head_tokens: int = None,
) -> Dict[str, Any]:
    """Arithmetic statistics of a population vs its retained targets.

    Works over sampled plans (planned statistics) or actual encoded sizes
    (built statistics; pass the built head via ``head_tokens``). Every
    retained published dimension is reported; only the dimensions the
    profile's generation authority produces exactly are *gated* (checked
    against their per-dimension tolerance). ``conformant_population`` is true
    only when the population reaches the calibration reference size AND every
    gated dimension is within tolerance. These are client-side statistics —
    the server-defined (usage-reported) context is judged separately by
    :func:`live_conformance_report`.
    """
    if head_tokens is None:
        head_tokens = profile.head_tokens
    turns = np.array([p.turn_count for p in plans])
    contexts = np.array([p.final_context for p in plans])
    deltas = np.array([d for p in plans for d in p.turn_inputs])
    request_prompt_lens = []
    for p in plans:
        prompt = head_tokens + initial_overhead
        for turn_index, turn_input in enumerate(p.turn_inputs):
            if turn_index == 0:
                prompt += turn_input
            else:
                prompt += turn_input + profile.output_len_per_turn + round_overhead
            request_prompt_lens.append(prompt)
    isl = np.array(request_prompt_lens)

    realized_values = {
        "turns_mean": float(turns.mean()),
        "turns_p50": float(np.percentile(turns, 50)),
        "turns_p95": float(np.percentile(turns, 95)),
        "input_per_turn_mean": float(deltas.mean()),
        "input_per_turn_p50": float(np.percentile(deltas, 50)),
        "input_per_turn_p95": float(np.percentile(deltas, 95)),
        "input_per_turn_max": float(deltas.max()),
        "final_context_mean": float(contexts.mean()),
        "final_context_p50": float(np.percentile(contexts, 50)),
        "final_context_p95": float(np.percentile(contexts, 95)),
        "final_context_max": float(contexts.max()),
        "request_weighted_isl_mean": float(isl.mean()),
    }
    dimensions = _dimension_records(profile, realized_values)
    population_ok = len(plans) >= profile.reference_population
    gates_ok = all(
        dimensions[d]["within_tolerance"] for d in profile.gated_dimensions()
    )
    return {
        "sessions": len(plans),
        "authority": profile.authority,
        "dimensions": dimensions,
        "repaired_sessions": sum(1 for p in plans if p.repaired),
        "gates_within_tolerance": gates_ok,
        # Conformance is a population property AND a gate property; small
        # builds report sampling deviation without claiming conformance.
        "conformant_population": population_ok and gates_ok,
    }


def live_conformance_report(
    profile: SessionProfile,
    server_prompt_lens: List,
    session_indices: List,
    round_indices: List,
    conversation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Judge the *server-defined* workload shape from ``usage.prompt_tokens``.

    The immutable final-context definition is the server-reported prompt
    length of a session's last request; this report groups the benchmark's
    per-round usage by session and evaluates the observable dimensions
    (turns, final context, request-weighted ISL) against the profile gates.
    Per-turn input is not directly server-observable and is never judged
    here. Sessions that failed or have any round without usage are excluded;
    if any session is excluded, live conformance is unavailable rather than
    silently partial.
    """
    planned = {
        c["session_index"]: c
        for c in conversation_results
        if c["session_index"] is not None
    }
    per_session: Dict[int, Dict[int, int]] = {}
    for prompt_len, session, round_index in zip(
        server_prompt_lens, session_indices, round_indices
    ):
        if session is None:
            continue
        per_session.setdefault(session, {})[round_index] = prompt_len

    turns, finals, isl = [], [], []
    complete = True
    for session, conversation in planned.items():
        rounds = per_session.get(session, {})
        usable = (
            conversation["success"]
            and len(rounds) == conversation["planned_rounds"]
            and all(rounds.get(i) is not None for i in range(len(rounds)))
        )
        if not usable:
            complete = False
            continue
        ordered = [rounds[i] for i in range(len(rounds))]
        turns.append(len(ordered))
        finals.append(ordered[-1])
        isl.extend(ordered)

    if not turns:
        return {
            "available": False,
            "reason": "no session with complete server usage",
        }

    turns_arr = np.array(turns)
    finals_arr = np.array(finals)
    isl_arr = np.array(isl)
    realized_values = {
        "turns_mean": float(turns_arr.mean()),
        "turns_p50": float(np.percentile(turns_arr, 50)),
        "turns_p95": float(np.percentile(turns_arr, 95)),
        "final_context_mean": float(finals_arr.mean()),
        "final_context_p50": float(np.percentile(finals_arr, 50)),
        "final_context_p95": float(np.percentile(finals_arr, 95)),
        "final_context_max": float(finals_arr.max()),
        "request_weighted_isl_mean": float(isl_arr.mean()),
    }
    records = _dimension_records(profile, realized_values)
    live_gated = [d for d in profile.gated_dimensions() if d in records]
    gates_ok = all(records[d]["within_tolerance"] for d in live_gated)
    population_ok = len(turns) >= profile.reference_population
    return {
        "available": True,
        "authority": profile.authority,
        "sessions_observed": len(turns),
        "all_sessions_usable": complete,
        "dimensions": records,
        "live_gated_dimensions": live_gated,
        "gates_within_tolerance": gates_ok and complete,
        # Small live runs report deviations as sampling error, exactly like
        # small builds; conformance additionally needs every session usable.
        "conformant_population": population_ok and gates_ok and complete,
    }


def _dimension_records(
    profile: SessionProfile, realized_values: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """One structured record per retained dimension: target, comparison,
    tolerance, realized value, deviation, gating, and tolerance verdict.
    Non-gated dimensions carry the same fields (informational verdict); the
    published maxima are one-sided upper bounds with zero tolerance."""
    gated = set(profile.gated_dimensions())
    records = {}
    for dimension, realized in realized_values.items():
        target = profile.dimension_target(dimension)
        tolerance = profile.dimension_tolerance(dimension)
        deviation = (realized - target) / target
        if dimension in UPPER_BOUND_DIMENSIONS:
            comparison = "upper_bound"
            within = realized <= target
        else:
            comparison = "two_sided"
            within = abs(deviation) <= tolerance
        records[dimension] = {
            "target": target,
            "comparison": comparison,
            "tolerance": tolerance,
            "realized": round(realized, 1),
            "deviation_frac": round(deviation, 4),
            "gated": dimension in gated,
            "within_tolerance": within,
        }
    return records


# Metadata fields that must match for a cached build to be reused.
_CACHE_COMPAT_FIELDS = (
    "dataset",
    "generator_version",
    "profile",
    "seed",
    "tokenizer_path",
    "tokenizer_fingerprint",
)


def _cache_path(metadata_key: Dict[str, Any]) -> Path:
    digest = hashlib.sha1(
        json.dumps(metadata_key, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    profile_name = metadata_key["profile"]["name"]
    name = f"recovery_agent_{profile_name}_{digest}.json"
    return Path.home() / ".cache" / "sglang" / "benchmark" / name


def _write_cache_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _load_payload_file(path: Path) -> Dict:
    with open(path) as f:
        payload = json.load(f)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("metadata"), dict)
        or not isinstance(payload.get("conversations"), list)
    ):
        raise ValueError(
            f"{path} is not a recovery-agent dataset file: expected a JSON "
            'object with "metadata" and "conversations" keys.'
        )
    return payload


@dataclass
class RecoveryAgentDataset(BaseDataset):
    profile_name: str
    seed: int
    num_sessions: int
    offset: int
    dataset_path: str
    authority: str = AUTHORITY_CONTEXT

    @classmethod
    def from_args(cls, args: Namespace) -> "RecoveryAgentDataset":
        return cls(
            profile_name=args.recovery_profile,
            seed=args.seed,
            num_sessions=args.num_prompts,
            offset=args.dataset_offset,
            dataset_path=args.dataset_path,
            authority=args.recovery_authority,
        )

    def load(self, tokenizer: Any, model_id=None) -> List[DatasetRow]:
        if self.profile_name not in PROFILES:
            raise ValueError(
                f"Unknown recovery-agent profile {self.profile_name!r}; "
                f"choose from {sorted(PROFILES)}"
            )
        if self.authority not in (AUTHORITY_CONTEXT, AUTHORITY_INPUT):
            raise ValueError(
                f"Unknown recovery-agent authority {self.authority!r}; "
                f"choose 'context' or 'input'"
            )
        profile = msgspec.structs.replace(
            PROFILES[self.profile_name], authority=self.authority
        )
        needed = self.offset + self.num_sessions
        routing_seed = self.seed

        if self.dataset_path:
            payload = _load_payload_file(Path(self.dataset_path))
            # A prebuilt file is only accepted when it was built by this
            # generator schema for exactly the requested profile AND
            # authority: silently running a context-built file under a
            # requested input authority (or vice versa) would defeat the
            # authority switch, and old schemas carry incompatible stats.
            file_meta = payload["metadata"]
            if file_meta.get("dataset") != "recovery-agent":
                raise ValueError(
                    f"{self.dataset_path} is not a recovery-agent build "
                    f"(dataset={file_meta.get('dataset')!r})."
                )
            if file_meta.get("generator_version") != GENERATOR_VERSION:
                raise ValueError(
                    f"{self.dataset_path} was built by generator version "
                    f"{file_meta.get('generator_version')!r}; this code is "
                    f"version {GENERATOR_VERSION}. Rebuild the file."
                )
            file_profile = file_meta.get("profile") or {}
            if file_profile.get("name") != profile.name:
                raise ValueError(
                    f"{self.dataset_path} was built with profile "
                    f"{file_profile.get('name')!r}; rerun with "
                    f"--recovery-profile {file_profile.get('name')} "
                    f"(got {profile.name!r})."
                )
            if file_profile.get("authority") != profile.authority:
                raise ValueError(
                    f"{self.dataset_path} was built {file_profile.get('authority')}-"
                    f"authoritative but --recovery-authority requested "
                    f"{profile.authority!r}; rerun with --recovery-authority "
                    f"{file_profile.get('authority')} or rebuild the file."
                )
            try:
                embedded = msgspec.convert(file_profile, SessionProfile)
            except msgspec.ValidationError as error:
                raise ValueError(
                    f"{self.dataset_path} embeds an incompatible profile "
                    f"schema: {error}. Rebuild the file."
                ) from error
            if embedded != profile:
                raise ValueError(
                    f"{self.dataset_path} embeds profile constants that differ "
                    "from this build's profile definition; rebuild the file "
                    "or use the matching code revision."
                )
            routing_seed = file_meta.get("seed", self.seed)
        else:
            payload = self._load_or_build(tokenizer, profile, needed)

        conversations = payload["conversations"]
        if needed > len(conversations):
            raise ValueError(
                f"Requested sessions [{self.offset}, {needed}) but the dataset "
                f"holds only {len(conversations)} conversations. Sessions are "
                "never recycled — lower --num-prompts/--num-sessions or "
                "--dataset-offset, or rebuild without --dataset-path."
            )
        selected = conversations[self.offset : needed]

        rows = []
        for session_index, conversation in enumerate(selected):
            rows.append(
                DatasetRow(
                    prompt=[turn["messages"] for turn in conversation],
                    # Informational: multi-turn replay reports server usage.
                    prompt_len=int(conversation[0].get("prompt_tokens", 0)),
                    output_len=profile.output_len_per_turn,
                    routing_key=_session_routing_key(
                        routing_seed, self.offset + session_index
                    ),
                )
            )

        # Report the statistics of the *selected slice*, not the whole cached
        # build; per-session planned and built arithmetic is retained in the
        # metadata.
        metadata = payload["metadata"]

        def slice_stats(key, head_tokens=None):
            stats = metadata.get(key)
            if not stats:
                return {}
            plans = [
                msgspec.convert(entry, SessionPlan)
                for entry in stats[self.offset : needed]
            ]
            return _realized_stats(
                plans,
                profile,
                metadata.get("initial_overhead", 0),
                metadata.get("round_overhead", 0),
                head_tokens=head_tokens,
            )

        planned = slice_stats("session_stats")
        built = slice_stats(
            "built_session_stats", head_tokens=metadata.get("built_head_tokens")
        )
        print(
            f"recovery-agent profile={profile.name} "
            f"authority={profile.authority} sessions={len(rows)} "
            f"(offset={self.offset}, synthetic distribution-matched data)"
        )
        if planned:
            print(
                json.dumps(
                    {
                        "selected_slice_planned_stats": planned,
                        "selected_slice_built_stats": built,
                    },
                    indent=2,
                )
            )
        if not planned.get("conformant_population", False):
            print(
                "NOTE: this slice does not establish calibration conformance; "
                "deviations above are sampling error at this population size "
                f"(reference population {profile.reference_population}; all "
                "gated dimensions must also be within tolerance)."
            )
        return rows

    def _load_or_build(
        self, tokenizer: Any, profile: SessionProfile, needed: int
    ) -> Dict[str, Any]:
        metadata_key = {
            "dataset": "recovery-agent",
            "generator_version": GENERATOR_VERSION,
            "profile": msgspec.to_builtins(profile),
            "seed": self.seed,
            "tokenizer_path": tokenizer.name_or_path,
            "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
        }
        path = _cache_path(metadata_key)
        if path.is_file():
            try:
                payload = _load_payload_file(path)
            except (ValueError, json.JSONDecodeError):
                print(f"Corrupt cache at {path}; rebuilding.")
                payload = None
            if payload is not None:
                cached_meta = payload["metadata"]
                compatible = all(
                    cached_meta.get(field) == metadata_key[field]
                    for field in _CACHE_COMPAT_FIELDS
                )
                if compatible and len(payload["conversations"]) >= needed:
                    return payload
                if not compatible:
                    print(f"Cache at {path} does not match this build; rebuilding.")

        # Per-session RNG streams make builds prefix-stable, so growing the
        # cache is a plain rebuild at the larger size.
        payload = build_sessions(
            tokenizer,
            profile=profile,
            seed=self.seed,
            num_sessions=needed,
        )
        _write_cache_atomic(path, payload)
        print(f"Built and cached {needed} sessions at {path}")
        return payload
