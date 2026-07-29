"""Offline regressions for the fail-closed ramp evidence validator.

Bug regression: the round-3 ramp printed ``RAMP COMPLETE`` while half its
worker artifacts were missing, its counters were unretained, and its stage-1
"baseline" had silently resolved to DSA — the collection was fail-open. These
tests prove the validator actually closes each of those doors: a synthetic,
fully well-formed evidence directory passes, and each malformed-artifact
class fails on its own (everything else held valid), so no gap can hide
behind another failure.
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

_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "recovery_agent_pd"
    / "validate_ramp_evidence.py"
)
_spec = importlib.util.spec_from_file_location("validate_ramp_evidence", _VALIDATOR_PATH)
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)

COMMIT = "f" * 40
ROUNDS_PER_SESSION = 3
NUM_SESSIONS = 32


def _server_args_line(stage, role, rank):
    fields = [f"model_path='/models/m{rank}'"]
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
    return "server_args=ServerArgs(" + ", ".join(fields) + ")"


def _counters(base):
    return {
        key: {
            "chat_requests": base,
            "prompt_tokens": base * 100,
            "cached_tokens": base * 10,
        }
        for key in validator.WORKER_KEYS
    }


def _bench(sessions, rounds):
    return {
        "completed_conversations": sessions,
        "total_conversations": sessions,
        "input_metrics_complete": True,
        "errors": ["" for _ in range(rounds)],
        "completed": rounds,
    }


def _wire():
    return {
        "reasoning_absent_from_round2": True,
        "round2_assistant_equals_content_only": True,
        "rounds_succeeded": True,
    }


def _attribution():
    keys = []
    for i in range(NUM_SESSIONS):
        prefill = [f"P{i % 2}"]
        decode = [f"D{2 + (i % 2)}"]
        probe = {"ok": True, "prefill_workers": prefill, "decode_workers": decode}
        keys.append(
            {
                "routing_key": f"session-{i:016x}",
                "probes": [probe, dict(probe)],
                "sticky": True,
                "stable": True,
            }
        )
    return {"all_sticky": True, "keys": keys}


def build_valid_evidence(ramp_dir):
    """A minimal synthetic evidence directory every check accepts."""
    dataset = {
        "metadata": {"seed": 42},
        "conversations": [
            [{"messages": []} for _ in range(ROUNDS_PER_SESSION)]
            for _ in range(NUM_SESSIONS)
        ],
    }
    dataset_path = ramp_dir / validator.DATASET_NAME
    dataset_path.write_text(json.dumps(dataset))
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    (ramp_dir / "manifest.txt").write_text(
        f"branch: recovery-bench @ {COMMIT}\n"
        f"dataset: fixed sha256={digest}\n"
    )
    for stage in (1, 2, 3, 4, 5):
        stage_dir = ramp_dir / f"stage{stage}"
        stage_dir.mkdir()
        for rank, role in validator.RANK_ROLES.items():
            (stage_dir / f"server_args_rank{rank}_{role}.txt").write_text(
                _server_args_line(stage, role, rank)
            )
            (stage_dir / f"startup_rank{rank}_{role}.log").write_text(
                f"stage {stage} rank {rank} startup\n"
            )
        rounds = ROUNDS_PER_SESSION * (8 if stage < 5 else NUM_SESSIONS)
        (stage_dir / "counters_pre.json").write_text(json.dumps(_counters(10)))
        (stage_dir / "counters_post.json").write_text(
            json.dumps(_counters(10 + rounds))
        )
        (stage_dir / "wire_probe.json").write_text(json.dumps(_wire()))
        if stage < 5:
            (stage_dir / "affinity.json").write_text(
                json.dumps(
                    {
                        "all_sticky": True,
                        "spread_ok": True,
                        "sessions": [
                            {"session": s, "bench_rc": 0} for s in range(8)
                        ],
                    }
                )
            )
            for session in range(8):
                (stage_dir / f"bench_s{session}.jsonl").write_text(
                    json.dumps(_bench(1, ROUNDS_PER_SESSION)) + "\n"
                )
        else:
            (stage_dir / "bench_32s.jsonl").write_text(
                json.dumps(_bench(NUM_SESSIONS, ROUNDS_PER_SESSION * NUM_SESSIONS))
                + "\n"
            )
            (stage_dir / "attribution.json").write_text(json.dumps(_attribution()))


class TestRampEvidenceValidator(CustomTestCase):
    def setUp(self):
        self.ramp_dir = Path(tempfile.mkdtemp(prefix="ramp_evidence_"))
        self.addCleanup(shutil.rmtree, self.ramp_dir, ignore_errors=True)
        build_valid_evidence(self.ramp_dir)

    def _failed(self):
        results = validator.validate(self.ramp_dir, expected_commit=COMMIT)
        return [r["check"] for r in results if not r["ok"]]

    def _edit_json(self, relative, mutate):
        path = self.ramp_dir / relative
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

    def test_well_formed_evidence_passes_every_check(self):
        self.assertEqual(self._failed(), [])

    def test_unresolved_server_args_fail(self):
        (self.ramp_dir / "stage3/server_args_rank2_decode.txt").write_text("MISSING")
        self.assertTrue(
            any("server_args rank2" in c for c in self._failed()), self._failed()
        )

    def test_empty_startup_log_fails(self):
        (self.ramp_dir / "stage2/startup_rank1_prefill.log").write_text("")
        failed = self._failed()
        self.assertTrue(any("startup log rank1" in c for c in failed), failed)
        self.assertTrue(any("distinct captures" in c for c in failed), failed)

    def test_stage1_resolving_to_dsa_fails_positive_and_negative_markers(self):
        """The archived bug: a 'baseline' stage that silently resolved to
        DSA must fail both the triton requirement and the DSA prohibition."""
        (self.ramp_dir / "stage1/server_args_rank0_prefill.txt").write_text(
            _server_args_line(2, "prefill", 0)
        )
        failed = self._failed()
        self.assertTrue(
            any("requires" in c and "triton" in c for c in failed), failed
        )
        self.assertTrue(any("must NOT resolve" in c for c in failed), failed)

    def test_early_eagle_leak_fails_negative_marker(self):
        (self.ramp_dir / "stage2/server_args_rank3_decode.txt").write_text(
            _server_args_line(3, "decode", 3)
        )
        self.assertTrue(
            any("stage2" in c and "must NOT resolve" in c for c in self._failed())
        )

    def test_decode_hicache_fails_final_stage(self):
        (self.ramp_dir / "stage5/server_args_rank2_decode.txt").write_text(
            _server_args_line(4, "prefill", 2).replace(
                "dsa_prefill_backend='trtllm', ", ""
            )
        )
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
        self.assertTrue(
            any("speculative_num_draft_tokens=4" in c for c in self._failed())
        )

    def test_empty_counter_snapshot_fails(self):
        (self.ramp_dir / "stage1/counters_pre.json").write_text("{}")
        self.assertTrue(
            any("counter snapshots complete" in c for c in self._failed())
        )

    def test_non_monotonic_counters_fail(self):
        self._edit_json(
            "stage2/counters_post.json",
            lambda c: c["P0"].update(chat_requests=0),
        )
        self.assertTrue(any("monotonic" in c for c in self._failed()))

    def test_insufficient_request_delta_fails(self):
        self._edit_json(
            "stage3/counters_post.json",
            lambda c: [v.update(chat_requests=10) for v in c.values()],
        )
        self.assertTrue(any("pool request deltas" in c for c in self._failed()))

    def test_incomplete_bench_usage_fails(self):
        self._edit_json(
            "stage1/bench_s4.jsonl",
            lambda b: b.update(input_metrics_complete=False),
        )
        self.assertTrue(any("bench_s4" in c for c in self._failed()))

    def test_bench_round_error_fails(self):
        self._edit_json(
            "stage4/bench_s0.jsonl", lambda b: b["errors"].__setitem__(1, "boom")
        )
        self.assertTrue(
            any("stage4" in c and "bench_s0" in c for c in self._failed())
        )

    def test_bench_round_count_mismatch_fails(self):
        self._edit_json(
            "stage5/bench_32s.jsonl",
            lambda b: b.update(completed=b["completed"] - 1),
        )
        self.assertTrue(any("32/32" in c for c in self._failed()))

    def test_unstable_attribution_fails(self):
        def flip(payload):
            payload["keys"][5]["probes"][1]["prefill_workers"] = ["P1"]
            payload["keys"][5]["probes"][1]["decode_workers"] = ["D3"]
            payload["keys"][5]["stable"] = False

        self._edit_json("stage5/attribution.json", flip)
        self.assertTrue(any("two-probe stable" in c for c in self._failed()))

    def test_single_pool_attribution_fails_spread(self):
        def collapse(payload):
            for record in payload["keys"]:
                for probe in record["probes"]:
                    probe["prefill_workers"] = ["P0"]

        self._edit_json("stage5/attribution.json", collapse)
        self.assertTrue(any("spread" in c for c in self._failed()))

    def test_missing_attribution_fails(self):
        (self.ramp_dir / "stage5/attribution.json").unlink()
        self.assertTrue(any("per-key attribution" in c for c in self._failed()))

    def test_dataset_sha_mismatch_fails(self):
        dataset_path = self.ramp_dir / validator.DATASET_NAME
        dataset_path.write_text(dataset_path.read_text() + " ")
        failed = self._failed()
        self.assertTrue(any("sha matches manifest" in c for c in failed), failed)

    def test_stale_manifest_commit_fails(self):
        results = validator.validate(self.ramp_dir, expected_commit="a" * 40)
        failed = [r["check"] for r in results if not r["ok"]]
        self.assertEqual(
            failed, ["manifest: stamped with the exact running commit"]
        )

    def test_identity_checks_run_in_per_stage_mode(self):
        """The driver's per-stage gate must not skip dataset/commit identity
        (a per-stage gate that skips identity is fail-open)."""
        results = validator.validate(
            self.ramp_dir, only_stage=2, expected_commit="a" * 40
        )
        checks = [r["check"] for r in results]
        self.assertIn("dataset: retained fixed population sha matches manifest", checks)
        failed = [r["check"] for r in results if not r["ok"]]
        self.assertEqual(failed, ["manifest: stamped with the exact running commit"])


if __name__ == "__main__":
    unittest.main()
