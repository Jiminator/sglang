"""Offline CPU tests for the recovery-agent dataset and the multi-turn
session harness (content-only replay, session CLI, usage-based metrics).

No network, no GPU: the tokenizer is a locally constructed word-level
tokenizer and all server interactions are faked.
"""

import asyncio
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

import sglang.benchmark.serving as serving
from sglang.benchmark.datasets import recovery_agent
from sglang.benchmark.datasets.recovery_agent import (
    MIN_TURN_INPUT_TOKENS,
    PROFILES,
    RecoveryAgentDataset,
    _allocate_turn_inputs,
    _plan_session,
    build_sessions,
)
from sglang.benchmark.serving import (
    RequestFuncInput,
    RequestFuncOutput,
    aggregate_cache_report,
    calculate_metrics,
    wrap_multi_turn_request_func,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _make_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "[PAD]": 1}
    vocab.update({f"tok_{i}": i + 2 for i in range(512)})
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }} : {{ message['content'] }} \n "
        "{% endfor %}"
    )
    hf_tokenizer.name_or_path = "local-test-tokenizer"
    return hf_tokenizer


def _tiny_profile(**overrides):
    """A shrunken profile so builds stay fast on the word-level tokenizer."""
    base = dict(
        name="tiny",
        authority=recovery_agent.AUTHORITY_CONTEXT,
        turns_mean=3.4,
        turns_p50=3.0,
        turns_p95=6.0,
        turns_cap=12,
        turns_tolerance=0.35,
        input_per_turn_mean=40.0,
        input_per_turn_p50=30.0,
        input_per_turn_p95=90.0,
        input_per_turn_cap=400,
        input_per_turn_max_target=400.0,
        input_per_turn_tolerance=0.50,
        final_context_mean=340.0,
        final_context_p50=300.0,
        final_context_p95=700.0,
        final_context_cap=1500,
        final_context_max_target=1500.0,
        final_context_tolerance=0.35,
        output_len_per_turn=20,
        head_tokens=60,
        turns_context_correlation=0.5,
        request_weighted_isl_target=260,
        request_weighted_isl_tolerance=0.35,
        reference_population=64,
        calibration_overheads=(13, 4),
    )
    base.update(overrides)
    return recovery_agent.SessionProfile(**base)


class TestSessionPlanning(CustomTestCase):
    def test_budget_exact_over_many_seeds(self):
        for profile in PROFILES.values():
            for i in range(300):
                rng = np.random.RandomState(10_000 + i)
                plan = _plan_session(profile, rng, 8, 8)
                self.assertEqual(sum(plan.turn_inputs), plan.input_budget)
                self.assertGreaterEqual(min(plan.turn_inputs), MIN_TURN_INPUT_TOKENS)
                self.assertLessEqual(max(plan.turn_inputs), profile.input_per_turn_cap)
                self.assertLessEqual(plan.turn_count, profile.turns_cap)
                self.assertLessEqual(plan.final_context, profile.final_context_cap)

    def test_allocation_is_exact_at_boundaries(self):
        profile = PROFILES["agent-short"]
        rng = np.random.RandomState(0)
        # Minimum feasible budget: every turn pinned at the floor.
        sizes = _allocate_turn_inputs(profile, rng, 5, 5 * MIN_TURN_INPUT_TOKENS)
        self.assertEqual(sizes, [MIN_TURN_INPUT_TOKENS] * 5)
        # Maximum feasible budget: every turn pinned at the cap.
        cap = profile.input_per_turn_cap
        sizes = _allocate_turn_inputs(profile, rng, 3, 3 * cap)
        self.assertEqual(sizes, [cap] * 3)

    def _canonical_realized(self, profile):
        initial_overhead, round_overhead = profile.calibration_overheads
        plans = [
            _plan_session(
                profile,
                np.random.RandomState(42 + i),
                initial_overhead,
                round_overhead,
            )
            for i in range(profile.reference_population)
        ]
        return recovery_agent._realized_stats(
            plans, profile, initial_overhead, round_overhead
        )

    def test_all_gated_dimensions_conform_in_both_authorities(self):
        """Every dimension the generation authority produces exactly must land
        within its declared tolerance at the calibration reference population,
        for both lineage profiles in both authority modes. Breaks whenever a
        baked constant, the sampler, or the allocation drifts."""
        for name in ("agent-long", "agent-short"):
            for authority in (
                recovery_agent.AUTHORITY_CONTEXT,
                recovery_agent.AUTHORITY_INPUT,
            ):
                profile = msgspec.structs.replace(PROFILES[name], authority=authority)
                realized = self._canonical_realized(profile)
                for dimension in profile.gated_dimensions():
                    entry = realized["dimensions"][dimension]
                    self.assertTrue(
                        entry["within_tolerance"],
                        f"{name}/{authority}: {dimension} realized "
                        f"{entry['realized']} vs target {entry['target']} "
                        f"(dev {entry['deviation_frac']:+.3f})",
                    )
                self.assertTrue(realized["gates_within_tolerance"])
                self.assertTrue(realized["conformant_population"])

    def test_mutated_calibration_constant_breaks_gates(self):
        """The gates are load-bearing: distorting one baked constant must
        flip conformance off (guards against decorative tolerances)."""
        distorted = msgspec.structs.replace(
            PROFILES["agent-short"], final_context_p50=26000.0
        )
        realized = self._canonical_realized(distorted)
        self.assertFalse(realized["gates_within_tolerance"])
        self.assertFalse(realized["conformant_population"])

    def test_input_authority_reports_but_does_not_gate_context(self):
        """In input-authoritative mode the context/ISL dimensions must be
        reported with full records (target/comparison/tolerance/verdict) but
        marked non-gated, so their honest overshoot never fails conformance."""
        profile = msgspec.structs.replace(
            PROFILES["agent-short"], authority=recovery_agent.AUTHORITY_INPUT
        )
        realized = self._canonical_realized(profile)
        for dimension in ("final_context_mean", "request_weighted_isl_mean"):
            entry = realized["dimensions"][dimension]
            self.assertFalse(entry["gated"])
            for field in ("target", "comparison", "tolerance", "within_tolerance"):
                self.assertIn(field, entry)
        for dimension in ("input_per_turn_mean", "input_per_turn_p50"):
            self.assertTrue(realized["dimensions"][dimension]["gated"])
        # The context dims overshoot (documented incompatibility), yet the
        # gates stay green because they are not gated in this mode.
        self.assertTrue(realized["gates_within_tolerance"])

    def test_relaxed_operational_cap_fails_published_max_gate(self):
        """The published maxima are one-sided gates against their own targets,
        independent of the operational clamps: relaxing the input cap in
        input-authoritative mode once let realized maxima reach ~3x the
        published maximum while conformance stayed true (bug regression)."""
        relaxed = msgspec.structs.replace(
            PROFILES["agent-short"],
            authority=recovery_agent.AUTHORITY_INPUT,
            input_per_turn_cap=100000,
        )
        realized = self._canonical_realized(relaxed)
        entry = realized["dimensions"]["input_per_turn_max"]
        self.assertEqual(entry["comparison"], "upper_bound")
        self.assertGreater(entry["realized"], entry["target"])
        self.assertFalse(entry["within_tolerance"])
        self.assertFalse(realized["gates_within_tolerance"])
        self.assertFalse(realized["conformant_population"])

    def test_context_max_gate_is_first_class(self):
        """Context mode gates the published final-context maximum as an
        upper bound; the built-in construction respects it."""
        realized = self._canonical_realized(PROFILES["agent-short"])
        entry = realized["dimensions"]["final_context_max"]
        self.assertTrue(entry["gated"])
        self.assertEqual(entry["comparison"], "upper_bound")
        self.assertEqual(entry["tolerance"], 0.0)
        self.assertTrue(entry["within_tolerance"])

    def test_input_authority_respects_caps(self):
        profile = msgspec.structs.replace(
            PROFILES["agent-short"], authority=recovery_agent.AUTHORITY_INPUT
        )
        for i in range(200):
            plan = _plan_session(profile, np.random.RandomState(500 + i), 13, 4)
            self.assertGreaterEqual(min(plan.turn_inputs), MIN_TURN_INPUT_TOKENS)
            self.assertLessEqual(max(plan.turn_inputs), profile.input_per_turn_cap)
            self.assertEqual(plan.input_budget, sum(plan.turn_inputs))
            self.assertLessEqual(plan.final_context, 2 * profile.final_context_cap)


class TestDatasetBuild(CustomTestCase):
    def setUp(self):
        self.tokenizer = _make_tokenizer()
        self.profile = _tiny_profile()

    def _build(self, seed=42, num_sessions=4):
        return build_sessions(
            self.tokenizer,
            profile=self.profile,
            seed=seed,
            num_sessions=num_sessions,
        )

    def test_deterministic_and_seed_sensitive(self):
        first = self._build()
        second = self._build()
        self.assertEqual(first["conversations"], second["conversations"])
        other_seed = self._build(seed=7)
        self.assertNotEqual(first["conversations"], other_seed["conversations"])

    def test_prefix_stable_growth(self):
        small = self._build(num_sessions=3)
        large = self._build(num_sessions=6)
        self.assertEqual(large["conversations"][:3], small["conversations"])

    def test_shared_head_unique_sessions(self):
        payload = self._build(num_sessions=3)
        heads = {conv[0]["messages"][0]["content"] for conv in payload["conversations"]}
        self.assertEqual(len(heads), 1)  # one fixed head across sessions
        issues = {
            conv[0]["messages"][1]["content"] for conv in payload["conversations"]
        }
        self.assertEqual(len(issues), 3)  # unique issue text per session

    def test_manifest_reports_synthetic_and_sampling_error(self):
        payload = self._build(num_sessions=4)
        metadata = payload["metadata"]
        self.assertTrue(metadata["synthetic"])
        planned = metadata["planned_stats"]
        self.assertEqual(planned["sessions"], 4)
        self.assertFalse(planned["conformant_population"])  # 4 < 64

    def test_stored_prompt_tokens_match_rendered_prompts(self):
        """The planned per-round prompt_tokens must track what the chat
        template actually renders — the overhead model once undercounted by
        charging half the per-round wrapper cost, skewing every final-context
        plan (bug regression)."""
        payload = self._build(num_sessions=2)
        for conversation in payload["conversations"]:
            messages = []
            for turn_index, turn in enumerate(conversation):
                messages.extend(turn["messages"])
                rendered = len(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                )
                # The plan charges the *planned* output length for earlier
                # rounds; rendering with empty assistant slots differs by
                # exactly those planned outputs.
                planned_outputs = turn_index * self.profile.output_len_per_turn
                planned = turn["prompt_tokens"] - planned_outputs
                # Pad sizing may undershoot each sized text by a bounded
                # deficit (head + issue + one delta per later round).
                max_deficit = (
                    2 + turn_index
                ) * recovery_agent.PAD_SIZING_MAX_DEFICIT_TOKENS
                self.assertLessEqual(rendered, planned)
                self.assertGreaterEqual(rendered, planned - max_deficit)
                messages.append({"role": "assistant", "content": ""})


class TestDatasetLoad(CustomTestCase):
    def setUp(self):
        self.tokenizer = _make_tokenizer()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.cache_file = Path(self.tmp_dir.name) / "cache.json"
        patches = [
            patch.object(
                recovery_agent, "_cache_path", lambda metadata_key: self.cache_file
            ),
            patch.dict(recovery_agent.PROFILES, {"agent-short": _tiny_profile()}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _dataset(self, **overrides):
        params = dict(
            profile_name="agent-short",
            seed=42,
            num_sessions=2,
            offset=0,
            dataset_path="",
        )
        params.update(overrides)
        return RecoveryAgentDataset(**params)

    def test_load_builds_rows_with_routing_keys(self):
        rows = self._dataset().load(self.tokenizer)
        self.assertEqual(len(rows), 2)
        keys = {row.routing_key for row in rows}
        self.assertEqual(len(keys), 2)
        for row in rows:
            self.assertTrue(row.routing_key.startswith("session-"))
            self.assertIsInstance(row.prompt, list)
            self.assertEqual(row.prompt[0][0]["role"], "system")

    def test_offset_slices_are_disjoint_and_deterministic(self):
        first = self._dataset(num_sessions=1).load(self.tokenizer)
        second = self._dataset(num_sessions=1, offset=1).load(self.tokenizer)
        self.assertNotEqual(first[0].prompt, second[0].prompt)
        self.assertNotEqual(first[0].routing_key, second[0].routing_key)
        both = self._dataset(num_sessions=2).load(self.tokenizer)
        self.assertEqual(both[0].prompt, first[0].prompt)
        self.assertEqual(both[1].prompt, second[0].prompt)

    def test_prebuilt_over_consumption_is_hard_error(self):
        self._dataset(num_sessions=2).load(self.tokenizer)  # writes the cache
        with self.assertRaisesRegex(ValueError, "never recycled"):
            self._dataset(num_sessions=3, dataset_path=str(self.cache_file)).load(
                self.tokenizer
            )

    def test_tampered_cache_fingerprint_forces_rebuild(self):
        self._dataset().load(self.tokenizer)
        payload = json.loads(self.cache_file.read_text())
        payload["metadata"]["seed"] = 999  # stale fingerprint
        self.cache_file.write_text(json.dumps(payload))
        self._dataset().load(self.tokenizer)
        rebuilt = json.loads(self.cache_file.read_text())
        self.assertEqual(rebuilt["metadata"]["seed"], 42)

    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown recovery-agent profile"):
            self._dataset(profile_name="nope").load(self.tokenizer)

    def test_decode_setting_change_forces_rebuild(self):
        """Decode behavior settings alter the sizing path's decode->re-encode
        results; toggling clean_up_tokenization_spaces at an unchanged path
        once left the fingerprint identical (bug regression)."""
        self._dataset().load(self.tokenizer)
        first = json.loads(self.cache_file.read_text())
        changed = _make_tokenizer()
        changed.clean_up_tokenization_spaces = (
            not self.tokenizer.clean_up_tokenization_spaces
        )
        self._dataset().load(changed)
        rebuilt = json.loads(self.cache_file.read_text())
        self.assertNotEqual(
            first["metadata"]["tokenizer_fingerprint"],
            rebuilt["metadata"]["tokenizer_fingerprint"],
        )

    def test_slow_tokenizer_asset_change_changes_fingerprint(self):
        """Non-fast tokenizers must fingerprint their actual asset contents
        (merges/SentencePiece/vocab files), not only get_vocab(). The
        installed transformers aliases every builtin tokenizer to its fast
        variant, so the slow branch is exercised with a minimal stand-in
        exposing the slow-tokenizer surface."""

        class _SlowStub:
            vocab_files_names = {"merges_file": "merges.txt"}
            chat_template = None
            special_tokens_map = {}
            init_kwargs = {}

            def get_vocab(self):
                return {"a": 0, "b": 1}

            def get_added_vocab(self):
                return {}

        merges = Path(self.tmp_dir.name) / "merges.txt"
        merges.write_text("a b\n")
        stub = _SlowStub()
        stub.init_kwargs = {"merges_file": str(merges)}
        first = recovery_agent._tokenizer_fingerprint(stub)
        merges.write_text("b a\n")  # same vocab, different merge rules
        second = recovery_agent._tokenizer_fingerprint(stub)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)  # full SHA-256, not truncated

    def test_prebuilt_authority_mismatch_rejected(self):
        """--recovery-authority must stay authoritative for prebuilt files:
        loading a context-built file under a requested input authority once
        silently ran as context (bug regression). Both directions rejected."""
        self._dataset().load(self.tokenizer)  # context-authoritative build
        with self.assertRaisesRegex(ValueError, "authority"):
            self._dataset(
                authority=recovery_agent.AUTHORITY_INPUT,
                dataset_path=str(self.cache_file),
            ).load(self.tokenizer)
        input_cache = Path(self.tmp_dir.name) / "input_cache.json"
        with patch.object(
            recovery_agent, "_cache_path", lambda metadata_key: input_cache
        ):
            self._dataset(authority=recovery_agent.AUTHORITY_INPUT).load(self.tokenizer)
        with self.assertRaisesRegex(ValueError, "authority"):
            self._dataset(dataset_path=str(input_cache)).load(self.tokenizer)

    def test_prebuilt_old_generator_version_rejected(self):
        self._dataset().load(self.tokenizer)
        payload = json.loads(self.cache_file.read_text())
        payload["metadata"]["generator_version"] = 1
        stale = Path(self.tmp_dir.name) / "stale.json"
        stale.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "generator version"):
            self._dataset(dataset_path=str(stale)).load(self.tokenizer)

    def test_planned_and_built_stats_are_separate(self):
        """Build metadata must carry planned (sampled arithmetic) and built
        (actual encoded sizes) statistics separately; built sizes may
        undershoot planned by the bounded pad deficit, never exceed them."""
        payload = build_sessions(
            self.tokenizer, profile=_tiny_profile(), seed=42, num_sessions=3
        )
        metadata = payload["metadata"]
        self.assertIn("planned_stats", metadata)
        self.assertIn("built_stats", metadata)
        for planned, built in zip(
            metadata["session_stats"], metadata["built_session_stats"]
        ):
            for planned_size, built_size in zip(
                planned["turn_inputs"], built["turn_inputs"]
            ):
                self.assertLessEqual(built_size, planned_size)
                self.assertGreaterEqual(
                    built_size,
                    planned_size - recovery_agent.PAD_SIZING_MAX_DEFICIT_TOKENS,
                )

    def test_small_slice_labels_deviations_as_sampling_error(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._dataset(num_sessions=2).load(self.tokenizer)
        self.assertIn("sampling error", buffer.getvalue())

    def test_tokenizer_content_change_forces_rebuild(self):
        """The cache key must fingerprint tokenizer *content*: changing the
        chat template at an unchanged name_or_path once silently reused stale
        conversations and stale overhead measurements (bug regression)."""
        self._dataset().load(self.tokenizer)
        first = json.loads(self.cache_file.read_text())
        changed = _make_tokenizer()
        changed.chat_template = (
            "{% for message in messages %}"
            "<<{{ message['role'] }}>> {{ message['content'] }} \n "
            "{% endfor %}"
        )
        self.assertEqual(changed.name_or_path, self.tokenizer.name_or_path)
        self._dataset().load(changed)
        rebuilt = json.loads(self.cache_file.read_text())
        self.assertNotEqual(
            first["metadata"]["tokenizer_fingerprint"],
            rebuilt["metadata"]["tokenizer_fingerprint"],
        )

    def test_offset_load_reports_selected_slice_statistics(self):
        """An offset load must report statistics for its own slice, not the
        whole cached prefix (bug regression: 8-session runs once printed
        16/24/32-session statistics)."""
        self._dataset(num_sessions=3).load(self.tokenizer)  # cache 3 sessions
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self._dataset(num_sessions=1, offset=2).load(self.tokenizer)
        printed = buffer.getvalue()
        payload = json.loads(printed[printed.index("{") : printed.rindex("}") + 1])
        self.assertEqual(payload["selected_slice_planned_stats"]["sessions"], 1)
        self.assertEqual(payload["selected_slice_built_stats"]["sessions"], 1)

    def test_prebuilt_profile_mismatch_rejected(self):
        """A prebuilt file's embedded profile is authoritative; loading it
        under a different CLI profile would silently change the per-turn
        output budget and mislabel results (bug regression)."""
        self._dataset().load(self.tokenizer)  # writes the cache
        payload = json.loads(self.cache_file.read_text())
        payload["metadata"]["profile"]["name"] = "other-profile"
        self.cache_file.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "other-profile"):
            self._dataset(dataset_path=str(self.cache_file)).load(self.tokenizer)


class TestLiveConformance(CustomTestCase):
    """Live conformance judges the server-defined context: only
    usage.prompt_tokens determines the gates, never the planned sizes."""

    def _report(self, profile, sessions):
        server_prompt_lens, session_indices, round_indices, conv_results = (
            [],
            [],
            [],
            [],
        )
        for session_index, (usage_lens, success) in enumerate(sessions):
            conv_results.append(
                {
                    "session_index": session_index,
                    "planned_rounds": len(usage_lens),
                    "completed_rounds": len(usage_lens) if success else 0,
                    "success": success,
                }
            )
            for round_index, prompt_len in enumerate(usage_lens):
                server_prompt_lens.append(prompt_len)
                session_indices.append(session_index)
                round_indices.append(round_index)
        return recovery_agent.live_conformance_report(
            profile,
            server_prompt_lens,
            session_indices,
            round_indices,
            conv_results,
        )

    def test_server_numbers_win_over_planned_sizes(self):
        """A profile whose planned context differs wildly from the observed
        usage must be judged by the usage: gates pass exactly when the server
        numbers (not the plan) are on target."""
        profile = _tiny_profile(reference_population=2)
        on_target = self._report(
            profile,
            [([200, 340], True), ([180, 300, 340], True)],
        )
        self.assertTrue(on_target["available"])
        self.assertTrue(
            on_target["dimensions"]["final_context_mean"]["within_tolerance"]
        )
        off_target = self._report(
            profile,
            [([2000, 3400], True), ([1800, 3000, 3400], True)],
        )
        self.assertFalse(
            off_target["dimensions"]["final_context_mean"]["within_tolerance"]
        )
        self.assertFalse(off_target["conformant_population"])

    def test_missing_usage_or_failed_session_blocks_conformance(self):
        profile = _tiny_profile(reference_population=2)
        report = self._report(
            profile,
            [([200, 340], True), ([180, None, 340], True)],
        )
        self.assertFalse(report["all_sessions_usable"])
        self.assertFalse(report["conformant_population"])
        failed = self._report(profile, [([200, 340], False)])
        self.assertFalse(failed["available"])

    def test_input_dimensions_never_judged_live(self):
        profile = msgspec.structs.replace(
            _tiny_profile(reference_population=1),
            authority=recovery_agent.AUTHORITY_INPUT,
        )
        report = self._report(profile, [([200, 340], True)])
        self.assertNotIn("input_per_turn_mean", report["dimensions"])
        for dimension in report["live_gated_dimensions"]:
            self.assertFalse(dimension.startswith("input_per_turn"))


class TestMultiTurnReplay(CustomTestCase):
    """The wrapper replays content only, aborts failed sessions, and stamps
    session/round identity."""

    def _wrap(self, request_func):
        return wrap_multi_turn_request_func(request_func, backend="sglang-oai-chat")

    def _input(self, rounds, session_index=0):
        return RequestFuncInput(
            model="m",
            prompt=rounds,
            api_url="http://localhost/v1/chat/completions",
            prompt_len=10,
            output_len=8,
            lora_name=None,
            image_data=None,
            extra_request_body={},
            session_index=session_index,
        )

    def test_replay_excludes_reasoning(self):
        seen_prompts = []

        async def fake_request(request_func_input, pbar=None):
            seen_prompts.append(request_func_input.prompt)
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = True
            output.generated_text = "THOUGHTS then the reply"
            output.assistant_replay_text = "the reply"
            return output

        outputs = asyncio.run(
            self._wrap(fake_request)(self._input(["q1", "q2"], session_index=3))
        )
        self.assertEqual(len(outputs), 2)
        replayed_assistant = seen_prompts[1][1]
        self.assertEqual(replayed_assistant["role"], "assistant")
        self.assertEqual(replayed_assistant["content"], "the reply")
        self.assertNotIn("THOUGHTS", replayed_assistant["content"])
        self.assertEqual([o.session_index for o in outputs], [3, 3])
        self.assertEqual([o.round_index for o in outputs], [0, 1])

    def test_missing_replay_text_aborts_session(self):
        """A successful round without assistant_replay_text must abort, not
        fall back to generated_text — the fallback would silently resend
        reasoning into the next round's prefix."""

        async def fake_request(request_func_input, pbar=None):
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = True
            output.generated_text = "THOUGHTS reply"
            output.assistant_replay_text = None
            return output

        outputs = asyncio.run(self._wrap(fake_request)(self._input(["a", "b"])))
        self.assertEqual(len(outputs), 1)
        self.assertFalse(outputs[0].success)
        self.assertIn("assistant_replay_text", outputs[0].error)

    def test_failed_round_aborts_session(self):
        calls = []

        async def fake_request(request_func_input, pbar=None):
            calls.append(1)
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = len(calls) < 2  # second round fails
            output.generated_text = "ok"
            output.assistant_replay_text = "ok"
            return output

        outputs = asyncio.run(self._wrap(fake_request)(self._input(["a", "b", "c"])))
        self.assertEqual(len(calls), 2)  # third round never attempted
        self.assertEqual(len(outputs), 2)
        self.assertFalse(outputs[-1].success)


class TestStreamingReplayExtraction(CustomTestCase):
    """The streaming chat path keeps reasoning in generated_text (metrics)
    while assistant_replay_text carries content deltas only."""

    class _FakeContent:
        def __init__(self, chunks):
            self._chunks = chunks

        def __aiter__(self):
            self._iter = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    def _fake_session(self, chunks=None, json_body=None):
        outer = self

        class _FakeResponse:
            status = 200
            content = outer._FakeContent(chunks or [])

            async def json(self):
                return json_body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        class _FakeSession:
            def post(self, url, json=None, headers=None):
                return _FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        return _FakeSession()

    def test_stream_separates_reasoning_from_replay(self):
        def sse(payload):
            return f"data: {json.dumps(payload)}".encode()

        chunks = [
            sse({"choices": [{"delta": {"reasoning_content": "THINKING "}}]}),
            sse({"choices": [{"delta": {"content": "final "}}]}),
            sse({"choices": [{"delta": {"content": "answer"}}]}),
            sse(
                {"choices": [], "usage": {"prompt_tokens": 77, "completion_tokens": 5}}
            ),
            b"data: [DONE]",
        ]
        request_func_input = RequestFuncInput(
            model="m",
            prompt=[{"role": "user", "content": "q"}],
            api_url="http://localhost/v1/chat/completions",
            prompt_len=10,
            output_len=8,
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )
        with patch.object(
            serving,
            "_create_bench_client_session",
            lambda: self._fake_session(chunks),
        ), patch.object(
            serving,
            "args",
            SimpleNamespace(
                disable_stream=False,
                disable_ignore_eos=True,
                dataset_name="recovery-agent",
            ),
            create=True,
        ):
            output = asyncio.run(
                serving.async_request_openai_chat_completions(request_func_input)
            )
        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, "THINKING final answer")
        self.assertEqual(output.assistant_replay_text, "final answer")
        self.assertEqual(output.server_prompt_len, 77)
        self.assertEqual(output.output_len, 5)

    def test_non_streaming_separates_reasoning_from_replay(self):
        body = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "THINKING ",
                        "content": "final answer",
                    }
                }
            ],
            "usage": {"prompt_tokens": 66, "completion_tokens": 4},
        }
        request_func_input = RequestFuncInput(
            model="m",
            prompt=[{"role": "user", "content": "q"}],
            api_url="http://localhost/v1/chat/completions",
            prompt_len=10,
            output_len=8,
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )
        with patch.object(
            serving,
            "_create_bench_client_session",
            lambda: self._fake_session(json_body=body),
        ), patch.object(
            serving,
            "args",
            SimpleNamespace(
                disable_stream=True,
                disable_ignore_eos=True,
                dataset_name="recovery-agent",
            ),
            create=True,
        ):
            output = asyncio.run(
                serving.async_request_openai_chat_completions(request_func_input)
            )
        self.assertTrue(output.success)
        self.assertEqual(output.generated_text, "THINKING final answer")
        self.assertEqual(output.assistant_replay_text, "final answer")
        self.assertEqual(output.server_prompt_len, 66)


class TestBenchmarkLevelBehavior(CustomTestCase):
    """End-to-end benchmark() runs against a faked request function: honest
    persistence when everything fails, unit-bearing multi-turn output, and
    the inline-thinking warning branches."""

    def _run_benchmark(self, request_func, rows, dataset_name="recovery-agent"):
        import io
        from contextlib import redirect_stdout

        result_dir = tempfile.TemporaryDirectory()
        self.addCleanup(result_dir.cleanup)
        args = SimpleNamespace(
            dataset_name=dataset_name,
            disable_stream=False,
            disable_ignore_eos=True,
            plot_throughput=False,
            recovery_profile="agent-short",
            recovery_authority="context",
            cache_report=False,
            output_file=str(Path(result_dir.name) / "result.jsonl"),
            output_details=False,
            profile_steps=None,
            mooncake_slowdown_factor=1.0,
            mooncake_num_rounds=1,
            warmup_requests=0,
            flush_cache=False,
            disable_tqdm=True,
            seed=42,
            backend="sglang-oai-chat",
            tag=None,
            request_rate=float("inf"),
            max_concurrency=None,
            sharegpt_output_len=None,
            random_input_len=None,
            random_output_len=None,
            random_range_ratio=None,
            num_prompts=2,
        )
        fake_server_info = SimpleNamespace(status_code=500, json=lambda: None)
        buffer = io.StringIO()
        with patch.object(serving, "args", args, create=True), patch.dict(
            serving.ASYNC_REQUEST_FUNCS, {"sglang-oai-chat": request_func}
        ), patch.object(
            serving.requests, "get", lambda *a, **k: fake_server_info
        ), redirect_stdout(
            buffer
        ):
            result = asyncio.run(
                serving.benchmark(
                    backend="sglang-oai-chat",
                    api_url="http://localhost/v1/chat/completions",
                    base_url="http://localhost",
                    model_id="m",
                    tokenizer=_make_tokenizer(),
                    input_requests=rows,
                    request_rate=float("inf"),
                    max_concurrency=None,
                    disable_tqdm=True,
                    lora_names=None,
                    lora_request_distribution=None,
                    lora_zipf_alpha=None,
                    extra_request_body={},
                    profile=False,
                    warmup_requests=0,
                )
            )
        return result, buffer.getvalue()

    def _rows(self, num_sessions=2, rounds=2):
        from sglang.benchmark.datasets.common import DatasetRow

        return [
            DatasetRow(
                prompt=[f"q{r}" for r in range(rounds)],
                prompt_len=10,
                output_len=8,
                routing_key=f"session-{i}",
            )
            for i in range(num_sessions)
        ]

    def test_all_failed_run_persists_honest_json(self):
        """An all-failed run once crashed with IndexError on the empty e2e
        percentile computation before anything was persisted (bug
        regression). It must serialize zero-valued metrics, one failed
        conversation record per planned session, retained errors, and never
        attempt later rounds."""
        calls = []

        async def failing_request(request_func_input, pbar=None):
            calls.append(request_func_input.session_index)
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = False
            output.error = "injected first-round failure"
            return output

        rows = self._rows(num_sessions=2, rounds=3)
        result, printed = self._run_benchmark(failing_request, rows)
        self.assertEqual(sorted(calls), [0, 1])  # one round per session only
        self.assertEqual(result["completed"], 0)
        self.assertEqual(result["completed_conversations"], 0)
        self.assertEqual(result["total_conversations"], 2)
        self.assertEqual(len(result["conversation_results"]), 2)
        for conversation in result["conversation_results"]:
            self.assertFalse(conversation["success"])
            self.assertEqual(conversation["planned_rounds"], 3)
            self.assertEqual(conversation["completed_rounds"], 0)
        self.assertEqual(result["errors"], ["injected first-round failure"] * 2)
        self.assertEqual(result["mean_e2e_latency_ms"], 0.0)
        json.dumps(result)  # must be serializable end to end

    def test_multi_turn_output_uses_session_and_turn_units(self):
        async def ok_request(request_func_input, pbar=None):
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = True
            output.generated_text = "fine"
            output.assistant_replay_text = "fine"
            output.server_prompt_len = 50
            output.output_len = 4
            output.latency = 0.1
            output.ttft = 0.05
            return output

        result, printed = self._run_benchmark(ok_request, self._rows())
        self.assertIn("Session arrival rate (sessions/s):", printed)
        self.assertIn("Turn throughput (turns/s):", printed)
        self.assertNotIn("Traffic request rate:", printed)
        self.assertNotIn("Request throughput (req/s):", printed)
        self.assertIsNotNone(result["turn_throughput_turns_per_s"])
        self.assertIsNotNone(result["session_throughput_sessions_per_s"])
        self.assertEqual(
            result["turn_throughput_turns_per_s"], result["request_throughput"]
        )
        self.assertIn("live_conformance", result)
        self.assertIn("NOTE: recovery-agent replays assistant `content` only", printed)

    def test_inline_thinking_in_replay_warns_once(self):
        async def leaky_request(request_func_input, pbar=None):
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = True
            output.generated_text = "<think>secret</think> answer"
            output.assistant_replay_text = "<think>secret</think> answer"
            output.server_prompt_len = 50
            output.output_len = 4
            output.latency = 0.1
            output.ttft = 0.05
            return output

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_benchmark(leaky_request, self._rows(num_sessions=1))
        inline = [w for w in caught if "Inline thinking markers" in str(w.message)]
        self.assertEqual(len(inline), 1)

    def test_clean_replay_does_not_warn(self):
        async def clean_request(request_func_input, pbar=None):
            output = RequestFuncOutput.init_new(request_func_input)
            output.success = True
            output.generated_text = "reasoned answer"
            output.assistant_replay_text = "reasoned answer"
            output.server_prompt_len = 50
            output.output_len = 4
            output.latency = 0.1
            output.ttft = 0.05
            return output

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._run_benchmark(clean_request, self._rows(num_sessions=1))
        inline = [w for w in caught if "Inline thinking markers" in str(w.message)]
        self.assertEqual(inline, [])


class TestUsageBasedMetrics(CustomTestCase):
    def _output(self, prompt_len, server_prompt_len, cached, success=True):
        output = RequestFuncOutput()
        output.success = success
        output.prompt_len = prompt_len
        output.server_prompt_len = server_prompt_len
        output.cached_tokens = cached
        output.generated_text = "x"
        output.output_len = 4
        output.latency = 1.0
        output.ttft = 0.5
        return output

    def test_cache_denominator_prefers_server_usage(self):
        outputs = [
            self._output(prompt_len=10, server_prompt_len=100, cached=50),
            self._output(prompt_len=10, server_prompt_len=None, cached=5),
            self._output(prompt_len=10, server_prompt_len=999, cached=0, success=False),
        ]
        report = aggregate_cache_report(outputs)
        # 100 (server) + 10 (client fallback); the failure is excluded.
        self.assertEqual(report["total_prompt_tokens"], 110)
        self.assertEqual(report["total_cached"], 55)
        self.assertAlmostEqual(report["hit_rate"], 50.0)

    def test_multi_turn_round_without_usage_is_excluded(self):
        """A multi-turn round with no server usage must not fall back to the
        stale first-round prompt length: doing so undercounts grown prompts
        and inflates the cache-hit rate (bug regression). Such rounds are
        excluded from both cache aggregation and total-input accounting."""
        with_usage = self._output(prompt_len=10, server_prompt_len=5000, cached=4000)
        with_usage.session_index = 0
        without_usage = self._output(prompt_len=10, server_prompt_len=None, cached=4000)
        without_usage.session_index = 0
        report = aggregate_cache_report([with_usage, without_usage])
        self.assertEqual(report["total_prompt_tokens"], 5000)
        self.assertEqual(report["total_cached"], 4000)

        tokenizer = _make_tokenizer()
        with patch.object(
            serving,
            "args",
            SimpleNamespace(dataset_name="recovery-agent", plot_throughput=False),
            create=True,
        ):
            metrics, _ = calculate_metrics(
                input_requests=None,
                outputs=[with_usage, without_usage],
                dur_s=2.0,
                tokenizer=tokenizer,
                backend="sglang-oai-chat",
            )
        self.assertEqual(metrics.total_input, 5000)

    def test_multi_turn_total_input_from_outputs(self):
        tokenizer = _make_tokenizer()
        outputs = [
            self._output(prompt_len=10, server_prompt_len=120, cached=0),
            self._output(prompt_len=10, server_prompt_len=250, cached=0),
        ]
        with patch.object(
            serving,
            "args",
            SimpleNamespace(dataset_name="recovery-agent", plot_throughput=False),
            create=True,
        ):
            metrics, _ = calculate_metrics(
                input_requests=None,
                outputs=outputs,
                dur_s=2.0,
                tokenizer=tokenizer,
                backend="sglang-oai-chat",
            )
        self.assertEqual(metrics.total_input, 370)


class TestSessionCli(CustomTestCase):
    def _run_cli(self, argv):
        with patch.object(serving.sys, "argv", ["bench_serving"] + argv), patch.object(
            serving, "run_benchmark"
        ) as run_mock:
            serving.cli_main()
        return run_mock.call_args.args[0]

    def test_num_sessions_conflicts_with_num_prompts(self):
        with self.assertRaises(SystemExit):
            self._run_cli(
                [
                    "--backend",
                    "sglang-oai-chat",
                    "--dataset-name",
                    "recovery-agent",
                    "--num-sessions",
                    "8",
                    "--num-prompts",
                    "8",
                ]
            )

    def test_num_sessions_rejected_for_single_turn_dataset(self):
        with self.assertRaises(SystemExit):
            self._run_cli(["--dataset-name", "random", "--num-sessions", "8"])

    def test_recovery_agent_requires_chat_backend(self):
        with self.assertRaises(SystemExit):
            self._run_cli(["--backend", "sglang", "--dataset-name", "recovery-agent"])

    def test_num_sessions_normalizes_into_num_prompts(self):
        args = self._run_cli(
            [
                "--backend",
                "sglang-oai-chat",
                "--dataset-name",
                "recovery-agent",
                "--num-sessions",
                "5",
            ]
        )
        self.assertEqual(args.num_prompts, 5)

    def test_default_profile_is_agent_short(self):
        args = self._run_cli(
            ["--backend", "sglang-oai-chat", "--dataset-name", "recovery-agent"]
        )
        self.assertEqual(args.recovery_profile, "agent-short")


if __name__ == "__main__":
    unittest.main()
