"""Offline regressions for the fail-closed ramp evidence validator.

Bug regression: the round-3 ramp printed ``RAMP COMPLETE`` while half its
worker artifacts were missing, its counters were unretained, and its stage-1
"baseline" had silently resolved to DSA — the collection was fail-open. A
round-5 adversarial review then demonstrated five FURTHER false passes
(forged attribution booleans, missing error fields, decreasing cached
counters, relabeled datasets, verdicts trusted from stored summary bits).
These tests prove each door is actually closed: one coherent synthetic GLM
2P2D evidence set passes every check, and each malformed-artifact class
fails on its own (everything else held valid), so no gap can hide behind
another failure.
"""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_TRACKED_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "recovery_agent_pd"
_spec_loader = importlib.util.spec_from_file_location(
    "validate_ramp_evidence", _TRACKED_DIR / "validate_ramp_evidence.py"
)
validator = importlib.util.module_from_spec(_spec_loader)
_spec_loader.loader.exec_module(validator)

COMMIT = "f" * 40
SEED = 42
ROUNDS_PER_SESSION = 3
NUM_SESSIONS = 32
MODEL = "/models/GLM-test-NVFP4"

RUN_SPEC = {
    "model_path": MODEL,
    "dataset_sha256": None,  # filled in per fixture after the dataset is built
    "dataset": {
        "id": "recovery-agent",
        "profile": "agent-short",
        "authority": "context",
        "seed": SEED,
        "num_sessions": NUM_SESSIONS,
    },
    "topology": {"tp_size": 4, "prefill_workers": 2, "decode_workers": 2},
    "sglang_version": "0.5.16",
}


def _server_args_line(stage, role, rank):
    """One coherent GLM 2P2D configuration: the common runtime identity on
    every rank plus the stage's feature delta."""
    fields = [
        f"model_path='{MODEL}'",
        "tp_size=4",
        "quantization='modelopt_fp4'",
        "kv_cache_dtype='fp8_e4m3'",
        f"disaggregation_mode='{role}'",
        "disaggregation_transfer_backend='nixl'",
        "reasoning_parser='glm45'",
        "enable_metrics=True",
        "enable_cache_report=True",
    ]
    fields.append(
        "attention_backend='triton'" if stage == 1 else "attention_backend='dsa'"
    )
    if stage >= 2 and role == "prefill":
        fields.append("dsa_prefill_backend='trtllm'")
    if stage >= 3:
        fields += [
            "speculative_algorithm='EAGLE'",
            "speculative_num_steps=3",
            "speculative_eagle_topk=1",
            "speculative_num_draft_tokens=4",
        ]
        if role == "decode":
            fields.append("speculative_attention_mode='decode'")
    else:
        fields.append("speculative_algorithm=None")
    if stage >= 4 and role == "prefill":
        fields += [
            "enable_hierarchical_cache=True",
            "hicache_size=32",
            "hicache_io_backend='direct'",
            "hicache_mem_layout='page_first_direct'",
            "hicache_write_policy='write_back'",
        ]
    else:
        fields.append("enable_hierarchical_cache=False")
    return f"[ts] rank{rank} server_args=ServerArgs(" + ", ".join(fields) + ")"


def _counters_pair(per_worker_rounds):
    """Aggregate snapshots coherent with the per-session affinity records:
    each worker's request delta equals the sum of its per-session deltas."""
    pre = {
        key: {"chat_requests": 10, "prompt_tokens": 1000, "cached_tokens": 100}
        for key in validator.WORKER_KEYS
    }
    post = {
        key: {
            "chat_requests": 10 + per_worker_rounds[key],
            "prompt_tokens": 1000 + 100 * per_worker_rounds[key],
            "cached_tokens": 100 + 10 * per_worker_rounds[key],
        }
        for key in validator.WORKER_KEYS
    }
    return pre, post


def _bench(sessions, rounds_per_session, dataset_sha, offset):
    total = sessions * rounds_per_session
    session_indices, round_indices = [], []
    for s in range(sessions):
        for r in range(rounds_per_session):
            session_indices.append(s)
            round_indices.append(r)
    server_prompt_lens = [100 + i for i in range(total)]
    return {
        "completed_conversations": sessions,
        "total_conversations": sessions,
        "input_metrics_complete": True,
        "completed": total,
        "total_input_tokens": sum(server_prompt_lens),
        # Metric fields consumed by the report generator (not the validator).
        "turn_throughput_turns_per_s": 1.0,
        "session_throughput_sessions_per_s": 0.1,
        "mean_ttft_ms": 10.0,
        "mean_e2e_latency_ms": 100.0,
        "live_conformance": {
            "available": False,
            "sessions_observed": sessions,
            "all_sessions_usable": True,
            "gates_within_tolerance": False,
            "conformant_population": False,
            "dimensions": {},
        },
        "errors": ["" for _ in range(total)],
        "server_prompt_lens": server_prompt_lens,
        "effective_prompt_lens": list(server_prompt_lens),
        "session_indices": session_indices,
        "round_indices": round_indices,
        "conversation_results": [
            {
                "session_index": s,
                "planned_rounds": rounds_per_session,
                "completed_rounds": rounds_per_session,
                "success": True,
            }
            for s in range(sessions)
        ],
        "dataset_binding": {
            "dataset_name": "recovery-agent",
            "dataset_path_sha256": dataset_sha,
            "dataset_offset": offset,
            "num_sessions": sessions,
            "profile": "agent-short",
            "authority": "context",
            "routing_keys": [
                validator.routing_key(SEED, offset + i) for i in range(sessions)
            ],
        },
    }


def _clip(text):
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "head": text[:120],
    }


def _wire():
    return {
        "round1_reasoning_present": True,
        "round1_reasoning": _clip("let me think about this carefully"),
        "round1_content_replay": _clip("the answer is 391"),
        "round2_assistant_content": _clip("the answer is 391"),
        "reasoning_absent_from_round2": True,
        "round2_assistant_equals_content_only": True,
        "rounds_succeeded": [True, True],
    }


def _affinity():
    sessions = []
    for s in range(8):
        prefill = f"P{s % 2}"
        decode = f"D{2 + (s % 2)}"
        delta = {key: 0 for key in validator.WORKER_KEYS}
        delta[prefill] = ROUNDS_PER_SESSION
        delta[decode] = ROUNDS_PER_SESSION
        sessions.append(
            {
                "session": s,
                "bench_rc": 0,
                "delta": delta,
                "prefill_workers": [prefill],
                "decode_workers": [decode],
                "sticky": True,
            }
        )
    return {"all_sticky": True, "spread_ok": True, "sessions": sessions}


def _attribution():
    keys = []
    for i in range(NUM_SESSIONS):
        prefill = [f"P{i % 2}"]
        decode = [f"D{2 + (i % 2)}"]
        probe = {"ok": True, "prefill_workers": prefill, "decode_workers": decode}
        keys.append(
            {
                "routing_key": validator.routing_key(SEED, i),
                "probes": [probe, json.loads(json.dumps(probe))],
                "sticky": True,
                "stable": True,
            }
        )
    return {"all_sticky": True, "keys": keys}


def build_valid_evidence(ramp_dir):
    """A coherent synthetic GLM 2P2D evidence set every check accepts.
    Returns the (spec, tracked_script) pair the validator must be run with."""
    dataset = {
        "metadata": {
            "dataset": "recovery-agent",
            "profile": {"name": "agent-short", "authority": "context"},
            "seed": SEED,
            "num_sessions": NUM_SESSIONS,
            "planned_stats": {"dimensions": {}},
        },
        "conversations": [
            [{"messages": []} for _ in range(ROUNDS_PER_SESSION)]
            for _ in range(NUM_SESSIONS)
        ],
    }
    dataset_path = ramp_dir / validator.DATASET_NAME
    dataset_path.write_text(json.dumps(dataset))
    spec = json.loads(json.dumps(RUN_SPEC))
    spec["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    tracked_script = ramp_dir / "tracked_launch_script.sh"
    tracked_script.write_text("#!/bin/bash\necho launch\n")
    script_digest = hashlib.sha256(tracked_script.read_bytes()).hexdigest()
    (ramp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "commit": COMMIT,
                "dataset_sha256": spec["dataset_sha256"],
                "worker_script_sha256": script_digest,
                "ranks": {
                    str(rank): {
                        "sglang_version": "0.5.16",
                        "record_sha256": "a" * 64,
                    }
                    for rank in validator.RANK_ROLES
                },
            }
        )
    )
    (ramp_dir / "router_config.txt").write_text(
        "python3 -m sglang_router.launch_router --pd-disaggregation "
        "--policy consistent_hashing --prefill-policy consistent_hashing "
        "--decode-policy consistent_hashing "
        "--prefill http://a:30001 8998 --prefill http://b:30001 8998 "
        "--decode http://c:30001 --decode http://d:30001 "
        "--host 0.0.0.0 --port 30500"
    )
    for stage in (1, 2, 3, 4, 5):
        stage_dir = ramp_dir / f"stage{stage}"
        stage_dir.mkdir()
        for rank, role in validator.RANK_ROLES.items():
            args_line = _server_args_line(stage, role, rank)
            (stage_dir / f"server_args_rank{rank}_{role}.txt").write_text(args_line)
            (stage_dir / f"startup_rank{rank}_{role}.log").write_text(
                f"stage {stage} rank {rank} startup\n{args_line}\nready\n"
            )
        if stage < 5:
            # 8 sessions alternate P0/P1 and D2/D3, so each worker serves
            # 4 sessions x ROUNDS_PER_SESSION requests.
            per_worker = {key: 4 * ROUNDS_PER_SESSION for key in validator.WORKER_KEYS}
        else:
            per_worker = {
                key: NUM_SESSIONS * ROUNDS_PER_SESSION // 2
                for key in validator.WORKER_KEYS
            }
        pre, post = _counters_pair(per_worker)
        (stage_dir / "counters_pre.json").write_text(json.dumps(pre))
        (stage_dir / "counters_post.json").write_text(json.dumps(post))
        (stage_dir / "wire_probe.json").write_text(json.dumps(_wire()))
        if stage < 5:
            (stage_dir / "affinity.json").write_text(json.dumps(_affinity()))
            for session in range(8):
                (stage_dir / f"bench_s{session}.jsonl").write_text(
                    json.dumps(
                        _bench(1, ROUNDS_PER_SESSION, spec["dataset_sha256"], session)
                    )
                    + "\n"
                )
        else:
            (stage_dir / "bench_32s.jsonl").write_text(
                json.dumps(
                    _bench(NUM_SESSIONS, ROUNDS_PER_SESSION, spec["dataset_sha256"], 0)
                )
                + "\n"
            )
            (stage_dir / "attribution.json").write_text(json.dumps(_attribution()))
    return spec, tracked_script


class TestRampEvidenceValidator(CustomTestCase):
    def setUp(self):
        self.ramp_dir = Path(tempfile.mkdtemp(prefix="ramp_evidence_"))
        self.addCleanup(shutil.rmtree, self.ramp_dir, ignore_errors=True)
        self.spec, self.tracked_script = build_valid_evidence(self.ramp_dir)

    def _failed(self, expected_commit=COMMIT, only_stage=None):
        results = validator.validate(
            self.ramp_dir,
            only_stage=only_stage,
            expected_commit=expected_commit,
            spec=self.spec,
            tracked_script=self.tracked_script,
        )
        return [r["check"] for r in results if not r["ok"]]

    def _edit_json(self, relative, mutate):
        path = self.ramp_dir / relative
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

    # -- positive ----------------------------------------------------------

    def test_well_formed_evidence_passes_every_check(self):
        self.assertEqual(self._failed(), [])

    # -- worker artifacts / configuration identity -------------------------

    def test_unresolved_server_args_fail(self):
        (self.ramp_dir / "stage3/server_args_rank2_decode.txt").write_text("MISSING")
        self.assertTrue(
            any("server_args rank2" in c for c in self._failed()), self._failed()
        )

    def test_empty_startup_log_fails(self):
        (self.ramp_dir / "stage2/startup_rank1_prefill.log").write_text("")
        failed = self._failed()
        self.assertTrue(any("startup log rank1" in c for c in failed), failed)
        self.assertTrue(any("distinct full captures" in c for c in failed), failed)

    def test_copied_log_content_fails_distinctness(self):
        """Full-content digests, not prefixes: a log copied from another rank
        (identical content) must fail even though a 200-byte-prefix check
        would consider unique-prefix copies fine."""
        source = (self.ramp_dir / "stage1/startup_rank0_prefill.log").read_text()
        (self.ramp_dir / "stage1/startup_rank1_prefill.log").write_text(source)
        failed = self._failed()
        self.assertTrue(any("distinct full captures" in c for c in failed), failed)

    def test_log_missing_its_args_line_fails(self):
        (self.ramp_dir / "stage1/startup_rank0_prefill.log").write_text(
            "startup happened but no args recorded\n"
        )
        self.assertTrue(
            any(
                "startup log rank0" in c and "resolved args line" in c
                for c in self._failed()
            )
        )

    def test_wrong_model_path_fails_identity(self):
        args = _server_args_line(2, "prefill", 0).replace(MODEL, "/models/other")
        (self.ramp_dir / "stage2/server_args_rank0_prefill.txt").write_text(args)
        (self.ramp_dir / "stage2/startup_rank0_prefill.log").write_text(
            f"head\n{args}\n"
        )
        self.assertTrue(any("model_path" in c for c in self._failed()))

    def test_wrong_tp_size_fails_identity(self):
        args = _server_args_line(1, "decode", 3).replace("tp_size=4", "tp_size=2")
        (self.ramp_dir / "stage1/server_args_rank3_decode.txt").write_text(args)
        (self.ramp_dir / "stage1/startup_rank3_decode.log").write_text(f"h\n{args}\n")
        self.assertTrue(any("tp_size=4" in c for c in self._failed()))

    def test_wrong_role_fails_identity(self):
        args = _server_args_line(3, "prefill", 2)  # a prefill config on a decode rank
        (self.ramp_dir / "stage3/server_args_rank2_decode.txt").write_text(args)
        (self.ramp_dir / "stage3/startup_rank2_decode.log").write_text(f"h\n{args}\n")
        self.assertTrue(
            any("disaggregation_mode='decode'" in c for c in self._failed())
        )

    def test_missing_nixl_or_parser_fails_identity(self):
        args = _server_args_line(1, "prefill", 0).replace(
            "disaggregation_transfer_backend='nixl'",
            "disaggregation_transfer_backend='mooncake'",
        )
        (self.ramp_dir / "stage1/server_args_rank0_prefill.txt").write_text(args)
        (self.ramp_dir / "stage1/startup_rank0_prefill.log").write_text(f"h\n{args}\n")
        self.assertTrue(
            any("disaggregation_transfer_backend='nixl'" in c for c in self._failed())
        )

    def test_stage1_resolving_to_dsa_fails_positive_and_negative_markers(self):
        """The archived bug: a 'baseline' stage that silently resolved to
        DSA must fail both the triton requirement and the DSA prohibition."""
        args = _server_args_line(2, "prefill", 0)
        (self.ramp_dir / "stage1/server_args_rank0_prefill.txt").write_text(args)
        (self.ramp_dir / "stage1/startup_rank0_prefill.log").write_text(f"h\n{args}\n")
        failed = self._failed()
        self.assertTrue(any("requires" in c and "triton" in c for c in failed), failed)
        self.assertTrue(any("must NOT resolve" in c for c in failed), failed)

    def test_early_eagle_leak_fails_negative_marker(self):
        args = _server_args_line(3, "decode", 3)
        (self.ramp_dir / "stage2/server_args_rank3_decode.txt").write_text(args)
        (self.ramp_dir / "stage2/startup_rank3_decode.log").write_text(f"h\n{args}\n")
        self.assertTrue(
            any("stage2" in c and "must NOT resolve" in c for c in self._failed())
        )

    def test_decode_hicache_fails_final_stage(self):
        args = (
            _server_args_line(4, "prefill", 2)
            .replace("dsa_prefill_backend='trtllm', ", "")
            .replace("disaggregation_mode='prefill'", "disaggregation_mode='decode'")
        )
        (self.ramp_dir / "stage5/server_args_rank2_decode.txt").write_text(args)
        (self.ramp_dir / "stage5/startup_rank2_decode.log").write_text(f"h\n{args}\n")
        self.assertTrue(
            any(
                "stage5" in c and "enable_hierarchical_cache=False" in c
                for c in self._failed()
            )
        )

    def test_wrong_eagle_values_fail(self):
        args = _server_args_line(4, "prefill", 0).replace(
            "speculative_num_draft_tokens=4", "speculative_num_draft_tokens=6"
        )
        (self.ramp_dir / "stage4/server_args_rank0_prefill.txt").write_text(args)
        (self.ramp_dir / "stage4/startup_rank0_prefill.log").write_text(f"h\n{args}\n")
        self.assertTrue(
            any("speculative_num_draft_tokens=4" in c for c in self._failed())
        )

    # -- counters ----------------------------------------------------------

    def test_empty_counter_snapshot_fails(self):
        (self.ramp_dir / "stage1/counters_pre.json").write_text("{}")
        self.assertTrue(any("counter snapshots complete" in c for c in self._failed()))

    def test_boolean_counter_value_fails(self):
        self._edit_json(
            "stage1/counters_pre.json", lambda c: c["P0"].update(chat_requests=True)
        )
        self.assertTrue(any("counter snapshots complete" in c for c in self._failed()))

    def test_negative_counter_value_fails(self):
        self._edit_json(
            "stage1/counters_post.json",
            lambda c: c["D2"].update(cached_tokens=-1000),
        )
        self.assertTrue(any("counter snapshots complete" in c for c in self._failed()))

    def test_decreasing_cached_tokens_fail_monotonicity(self):
        """Round-5 demonstrated false pass: cached_tokens decreasing between
        snapshots (still non-negative) passed the old two-field check."""
        self._edit_json(
            "stage2/counters_post.json", lambda c: c["P0"].update(cached_tokens=1)
        )
        self.assertTrue(any("monotonic" in c for c in self._failed()))

    def test_cached_delta_exceeding_prompt_delta_fails(self):
        self._edit_json(
            "stage3/counters_post.json",
            lambda c: c["P1"].update(cached_tokens=10 ** 9),
        )
        self.assertTrue(
            any("cached-token deltas bounded" in c for c in self._failed())
        )

    def test_insufficient_request_delta_fails(self):
        self._edit_json(
            "stage3/counters_post.json",
            lambda c: [v.update(chat_requests=10) for v in c.values()],
        )
        self.assertTrue(any("pool request deltas" in c for c in self._failed()))

    # -- benches -----------------------------------------------------------

    def test_incomplete_bench_usage_fails(self):
        self._edit_json(
            "stage1/bench_s4.jsonl",
            lambda b: b.update(input_metrics_complete=False),
        )
        self.assertTrue(any("bench_s4" in c for c in self._failed()))

    def test_missing_errors_field_fails(self):
        """Round-5 demonstrated false pass: deleting `errors` entirely was
        treated as error-free."""
        self._edit_json("stage4/bench_s0.jsonl", lambda b: b.pop("errors"))
        self.assertTrue(
            any(
                "stage4: bench_s0" in c and "error slots" in c
                for c in self._failed()
            )
        )

    def test_bench_round_error_fails(self):
        self._edit_json(
            "stage4/bench_s0.jsonl", lambda b: b["errors"].__setitem__(1, "boom")
        )
        self.assertTrue(
            any("stage4: bench_s0" in c and "error slots" in c for c in self._failed())
        )

    def test_short_prompt_len_array_fails(self):
        self._edit_json(
            "stage2/bench_s1.jsonl", lambda b: b["server_prompt_lens"].pop()
        )
        self.assertTrue(any("prompt lengths" in c for c in self._failed()))

    def test_incoherent_round_index_sequence_fails(self):
        self._edit_json(
            "stage2/bench_s2.jsonl", lambda b: b["round_indices"].__setitem__(1, 5)
        )
        self.assertTrue(any("index sequences" in c for c in self._failed()))

    def test_failed_conversation_result_fails(self):
        self._edit_json(
            "stage5/bench_32s.jsonl",
            lambda b: b["conversation_results"][7].update(success=False),
        )
        self.assertTrue(any("conversation_results" in c for c in self._failed()))

    def test_bench_round_count_mismatch_fails(self):
        self._edit_json(
            "stage5/bench_32s.jsonl", lambda b: b.update(completed=b["completed"] - 1)
        )
        self.assertTrue(any("stage5: bench_32s" in c for c in self._failed()))

    def test_failed_wire_rounds_fail(self):
        """Round-6 demonstrated false pass: rounds_succeeded=[False, False]
        is a truthy list and passed the old summary check."""
        self._edit_json(
            "stage2/wire_probe.json",
            lambda w: w.update(rounds_succeeded=[False, False]),
        )
        self.assertTrue(
            any("stage2" in c and "wire probe" in c for c in self._failed())
        )

    def test_forged_wire_equality_bit_fails(self):
        """The replay-equality bit must agree with the retained hashes."""

        def forge(payload):
            payload["round2_assistant_content"] = _clip("something else entirely")
            # stored bits left forged True

        self._edit_json("stage3/wire_probe.json", forge)
        self.assertTrue(
            any("stage3" in c and "wire probe" in c for c in self._failed())
        )

    def test_decode_cached_delta_exceeding_prompt_delta_fails(self):
        """Round-6 demonstrated false pass: the bound only covered prefill
        workers, so a decode worker's cached delta could exceed its prompt
        delta."""
        self._edit_json(
            "stage2/counters_post.json",
            lambda c: c["D2"].update(cached_tokens=10 ** 9),
        )
        self.assertTrue(
            any("cached-token deltas bounded" in c for c in self._failed())
        )

    def test_aggregate_vs_per_session_delta_mismatch_fails(self):
        """Round-6 demonstrated false pass: aggregate snapshots showing twice
        the per-session traffic passed because they were never reconciled."""

        def inflate(counters):
            for value in counters.values():
                value["chat_requests"] += 12

        self._edit_json("stage1/counters_post.json", inflate)
        self.assertTrue(any("per-session sums" in c for c in self._failed()))

    def test_duplicate_conversation_session_indices_fail(self):
        """Round-6 demonstrated false pass: relabeling every conversation
        result as session 0 passed the sorted-list check."""
        self._edit_json(
            "stage5/bench_32s.jsonl",
            lambda b: [c.update(session_index=0) for c in b["conversation_results"]],
        )
        self.assertTrue(any("uniquely cover" in c for c in self._failed()))

    def test_negative_total_input_tokens_fails(self):
        """Round-6 demonstrated false pass: the report headline total was
        never checked against the per-round prompt lengths."""
        self._edit_json(
            "stage5/bench_32s.jsonl", lambda b: b.update(total_input_tokens=-999999)
        )
        self.assertTrue(
            any("aggregate input tokens" in c for c in self._failed())
        )

    def test_missing_dataset_binding_fails(self):
        self._edit_json(
            "stage1/bench_s0.jsonl", lambda b: b.pop("dataset_binding")
        )
        self.assertTrue(
            any(
                "stage1: bench_s0" in c and "bound to the retained population" in c
                for c in self._failed()
            )
        )

    def test_wrong_binding_routing_key_fails(self):
        self._edit_json(
            "stage2/bench_s3.jsonl",
            lambda b: b["dataset_binding"]["routing_keys"].__setitem__(
                0, "session-0000000000000000"
            ),
        )
        self.assertTrue(
            any(
                "bench_s3" in c and "bound to the retained population" in c
                for c in self._failed()
            )
        )

    def test_wrong_binding_offset_fails(self):
        self._edit_json(
            "stage3/bench_s2.jsonl",
            lambda b: b["dataset_binding"].update(dataset_offset=7),
        )
        self.assertTrue(
            any(
                "bench_s2" in c and "bound to the retained population" in c
                for c in self._failed()
            )
        )

    def test_malformed_nested_types_record_failures_not_crashes(self):
        (self.ramp_dir / "stage4/wire_probe.json").write_text(
            json.dumps({"round1_reasoning": 5, "rounds_succeeded": {"a": 1}})
        )
        self._edit_json(
            "stage4/bench_s0.jsonl",
            lambda b: b.update(conversation_results=[17, None, "x"]),
        )
        self._edit_json(
            "stage4/affinity.json",
            lambda a: a["sessions"].__setitem__(2, {"session": 2, "delta": "junk"}),
        )
        failed = self._failed()  # must not raise
        self.assertTrue(any("stage4" in c for c in failed), failed)

    # -- affinity (derived, not trusted) -----------------------------------

    def test_affinity_without_raw_deltas_fails(self):
        """Round-5 demonstrated false pass: session records carrying only
        {session, bench_rc} were accepted. The verdict must be derivable."""

        def strip(payload):
            payload["sessions"] = [
                {"session": s["session"], "bench_rc": 0} for s in payload["sessions"]
            ]

        self._edit_json("stage1/affinity.json", strip)
        self.assertTrue(
            any("derived from raw per-worker deltas" in c for c in self._failed())
        )

    def test_affinity_worker_list_contradicting_deltas_fails(self):
        def contradict(payload):
            record = payload["sessions"][3]
            current = record["prefill_workers"]
            record["prefill_workers"] = ["P1"] if current == ["P0"] else ["P0"]

        self._edit_json("stage2/affinity.json", contradict)
        self.assertTrue(
            any(
                "stage2" in c and "derived from raw per-worker deltas" in c
                for c in self._failed()
            )
        )

    def test_forged_affinity_summary_bits_fail(self):
        def forge(payload):
            record = payload["sessions"][0]
            record["delta"]["P1"] = record["delta"]["P0"]  # two prefill hits
            record["prefill_workers"] = ["P0", "P1"]
            # sticky/all_sticky left forged True

        self._edit_json("stage3/affinity.json", forge)
        self.assertTrue(
            any(
                "stage3" in c and "derived from raw per-worker deltas" in c
                for c in self._failed()
            )
        )

    # -- attribution (derived, not trusted) --------------------------------

    def test_forged_attribution_bits_fail(self):
        """Round-5 demonstrated false pass: a failed second probe with a
        different pair, forged sticky/stable/all_sticky=True, passed."""

        def forge(payload):
            record = payload["keys"][5]
            record["probes"][1] = {
                "ok": False,
                "prefill_workers": ["P1"],
                "decode_workers": ["D3"],
            }
            # sticky/stable/all_sticky left forged True

        self._edit_json("stage5/attribution.json", forge)
        self.assertTrue(
            any("two successful identical single-pair probes" in c for c in self._failed())
        )

    def test_duplicate_routing_key_fails(self):
        def duplicate(payload):
            payload["keys"][1]["routing_key"] = payload["keys"][0]["routing_key"]

        self._edit_json("stage5/attribution.json", duplicate)
        self.assertTrue(
            any("dataset-derived routing keys" in c for c in self._failed())
        )

    def test_unexpected_routing_key_fails(self):
        self._edit_json(
            "stage5/attribution.json",
            lambda p: p["keys"][2].update(routing_key="session-not-derived"),
        )
        self.assertTrue(
            any("dataset-derived routing keys" in c for c in self._failed())
        )

    def test_single_pool_attribution_fails_spread(self):
        def collapse(payload):
            for record in payload["keys"]:
                for probe in record["probes"]:
                    probe["prefill_workers"] = ["P0"]

        self._edit_json("stage5/attribution.json", collapse)
        self.assertTrue(any("spread" in c for c in self._failed()))

    def test_missing_attribution_fails(self):
        (self.ramp_dir / "stage5/attribution.json").unlink()
        failed = self._failed()
        self.assertTrue(any("routing keys" in c for c in failed), failed)

    # -- run identity ------------------------------------------------------

    def test_dataset_sha_mismatch_fails(self):
        dataset_path = self.ramp_dir / validator.DATASET_NAME
        dataset_path.write_text(dataset_path.read_text() + " ")
        self.assertTrue(
            any("sha matches the tracked run spec" in c for c in self._failed())
        )

    def test_relabeled_dataset_semantics_fail(self):
        """Round-5 demonstrated false pass: a dataset with a different seed
        and profile, re-hashed into the manifest, was accepted. The semantic
        identity now comes from the TRACKED spec."""

        def relabel(payload):
            payload["metadata"]["seed"] = 999
            payload["metadata"]["profile"]["name"] = "agent-long"

        self._edit_json(validator.DATASET_NAME, relabel)
        digest = hashlib.sha256(
            (self.ramp_dir / validator.DATASET_NAME).read_bytes()
        ).hexdigest()
        self._edit_json("manifest.json", lambda m: m.update(dataset_sha256=digest))
        failed = self._failed()
        self.assertTrue(any("semantic identity" in c for c in failed), failed)
        self.assertTrue(
            any("sha matches the tracked run spec" in c for c in failed), failed
        )

    def test_stale_manifest_commit_fails(self):
        failed = self._failed(expected_commit="a" * 40)
        self.assertEqual(failed, ["manifest: stamped with the exact running commit"])

    def test_worker_script_digest_mismatch_fails(self):
        self.tracked_script.write_text("#!/bin/bash\necho DIFFERENT\n")
        self.assertTrue(
            any("TRACKED launch script" in c for c in self._failed())
        )

    def test_wrong_sglang_version_fails(self):
        self._edit_json(
            "manifest.json",
            lambda m: m["ranks"]["2"].update(sglang_version="0.5.15"),
        )
        self.assertTrue(any("sglang 0.5.16" in c for c in self._failed()))

    def test_divergent_package_records_fail(self):
        self._edit_json(
            "manifest.json",
            lambda m: m["ranks"]["1"].update(record_sha256="b" * 64),
        )
        self.assertTrue(any("RECORD digest" in c for c in self._failed()))

    def test_missing_router_topology_fails(self):
        (self.ramp_dir / "router_config.txt").write_text(
            "--policy round_robin --prefill http://a:30001 8998 --decode http://c:30001"
        )
        self.assertTrue(any("router" in c for c in self._failed()))

    def test_identity_checks_run_in_per_stage_mode(self):
        """The driver's per-stage gate must not skip identity (a per-stage
        gate that skips dataset/script/commit identity is fail-open)."""
        results = validator.validate(
            self.ramp_dir,
            only_stage=2,
            expected_commit="a" * 40,
            spec=self.spec,
            tracked_script=self.tracked_script,
        )
        checks = [r["check"] for r in results]
        self.assertIn(
            "dataset: retained population sha matches the tracked run spec", checks
        )
        failed = [r["check"] for r in results if not r["ok"]]
        self.assertEqual(failed, ["manifest: stamped with the exact running commit"])


def _load_report_module():
    spec_loader = importlib.util.spec_from_file_location(
        "make_ramp_report", _TRACKED_DIR / "make_ramp_report.py"
    )
    report_module = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(report_module)
    return report_module


class TestReportGenerator(CustomTestCase):
    """The report generator must re-validate the artifacts it renders (a
    stale passing verdict JSON must not survive — round-5 demonstrated
    staleness hole) and must be TOTAL over failed/missing evidence (a
    missing stage once crashed generation with FileNotFoundError — round-6
    demonstrated failure path)."""

    def setUp(self):
        self.ramp_dir = Path(tempfile.mkdtemp(prefix="ramp_report_"))
        self.addCleanup(shutil.rmtree, self.ramp_dir, ignore_errors=True)
        build_valid_evidence(self.ramp_dir)
        self.report = _load_report_module()
        self.output = self.ramp_dir / "report.md"

    def test_stale_passing_json_does_not_survive_generation(self):
        # Forge a stale "all_ok" verdict, then break the evidence.
        (self.ramp_dir / "evidence_validation.json").write_text(
            json.dumps({"checks": [], "all_ok": True})
        )
        (self.ramp_dir / "stage1/server_args_rank0_prefill.txt").write_text("MISSING")
        self.report.generate(self.ramp_dir, self.output)
        text = self.output.read_text()
        self.assertIn("NOT a validated controlled ramp", text)
        self.assertNotIn("evidence-validated", text)
        # The stored JSON was refreshed by generation, not trusted.
        refreshed = json.loads(
            (self.ramp_dir / "evidence_validation.json").read_text()
        )
        self.assertFalse(refreshed["all_ok"])

    def test_missing_stage_renders_failed_row_with_last_passing_stage(self):
        """Round-6 demonstrated failure: removing stage4 crashed generation
        instead of reporting the failed stage and last passing stage."""
        shutil.rmtree(self.ramp_dir / "stage4")
        (self.ramp_dir / "ledger.json").write_text(
            json.dumps(
                {
                    "run_id": "test-run",
                    "commit": COMMIT,
                    "spec_sha256": "0" * 64,
                    "stages": {
                        "1": {"status": "passed", "reason": None},
                        "2": {"status": "passed", "reason": None},
                        "3": {"status": "passed", "reason": None},
                        "4": {"status": "failed", "reason": "workers not healthy"},
                        "5": {"status": "failed", "reason": "prior stage failed"},
                    },
                    "last_passing_stage": 3,
                }
            )
        )
        self.report.generate(self.ramp_dir, self.output)  # must not raise
        text = self.output.read_text()
        self.assertIn("last passing stage: 3", text)
        self.assertIn("FAILED/MISSING", text)
        self.assertIn("workers not healthy", text)
        self.assertIn("NOT a validated controlled ramp", text)

    def test_missing_final_stage_renders_without_metrics(self):
        shutil.rmtree(self.ramp_dir / "stage5")
        self.report.generate(self.ramp_dir, self.output)  # must not raise
        text = self.output.read_text()
        self.assertNotIn("Prefill-pool radix hit rate", text)
        self.assertIn("FAILED/MISSING", text)

    def test_notes_are_appended_verbatim(self):
        notes = self.ramp_dir / "notes.md"
        notes.write_text("archive-specific narrative sentinel")
        self.report.generate(self.ramp_dir, self.output, notes=notes)
        self.assertIn(
            "archive-specific narrative sentinel", self.output.read_text()
        )


if __name__ == "__main__":
    unittest.main()
