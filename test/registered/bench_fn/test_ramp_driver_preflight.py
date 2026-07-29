"""Offline regressions for the ramp driver's fail-closed preflight.

Bug regression (round-6 review): with the remote identity helpers stubbed to
valid values, a fleet with one rank, the wrong role, and a wrong model path
passed preflight and wrote a manifest — the mismatch would only surface after
fleet mutation. Preflight must reject every identity error BEFORE any
kill/launch/benchmark, refuse a non-fresh directory for a full run, and
refuse a resume whose manifest or prior-stage evidence does not validate.
No test here touches a fleet: every remote helper is stubbed.
"""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TRACKED_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "recovery_agent_pd"
_loader = importlib.util.spec_from_file_location(
    "ramp_driver", _TRACKED_DIR / "ramp_driver.py"
)
driver = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(driver)

COMMIT = "f" * 40


def _fleet_config(model="/models/GLM-test", ranks=None):
    if ranks is None:
        ranks = {
            "0": {"role": "prefill", "ip": "10.0.0.1"},
            "1": {"role": "prefill", "ip": "10.0.0.2"},
            "2": {"role": "decode", "ip": "10.0.0.3"},
            "3": {"role": "decode", "ip": "10.0.0.4"},
        }
    return {
        "ranks": ranks,
        "router_url": "http://127.0.0.1:30500",
        "remote_cmd": ["true"],
        "wrapper": "",
        "source_dataset": "/nonexistent",
        "worker_script": "/scratch/launch.sh",
        "router_python": "python3",
        "model_path": model,
    }


class TestPreflight(CustomTestCase):
    def setUp(self):
        self.ramp_dir = Path(tempfile.mkdtemp(prefix="ramp_preflight_"))
        self.addCleanup(shutil.rmtree, self.ramp_dir, ignore_errors=True)
        dataset = json.dumps({"metadata": {}, "conversations": []})
        (self.ramp_dir / driver.DATASET_NAME).write_text(dataset)
        self.dataset_sha = hashlib.sha256(dataset.encode()).hexdigest()
        self.spec = {
            "model_path": "/models/GLM-test",
            "dataset_sha256": self.dataset_sha,
            "dataset": {},
            "topology": {"tp_size": 4, "prefill_workers": 2, "decode_workers": 2},
            "sglang_version": "0.5.16",
        }
        self.tracked_digest = hashlib.sha256(
            driver.TRACKED_WORKER_SCRIPT.read_bytes()
        ).hexdigest()

    def _preflight(self, start_stage=1, config=None, script_digest=None,
                   identity=None, stage_valid=True):
        fleet = driver.Fleet(config or _fleet_config(), self.ramp_dir)
        with patch.object(
            driver, "rank_script_digest",
            lambda *_: script_digest or self.tracked_digest,
        ), patch.object(
            driver, "rank_package_identity",
            lambda *_: dict(
                identity or {"sglang_version": "0.5.16", "record_sha256": "a" * 64}
            ),
        ), patch.object(driver, "head_commit", lambda: COMMIT), patch.object(
            driver, "validate_stage", lambda *_a, **_k: stage_valid
        ):
            driver.preflight(fleet, self.spec, start_stage)

    def test_valid_full_run_passes_and_writes_manifest(self):
        self._preflight()
        manifest = json.loads((self.ramp_dir / "manifest.json").read_text())
        self.assertEqual(manifest["commit"], COMMIT)
        self.assertEqual(manifest["dataset_sha256"], self.dataset_sha)
        self.assertEqual(manifest["worker_script_sha256"], self.tracked_digest)
        self.assertEqual(len(manifest["ranks"]), 4)

    def test_wrong_model_path_refused(self):
        with self.assertRaisesRegex(RuntimeError, "fleet model"):
            self._preflight(config=_fleet_config(model="/wrong/model"))

    def test_wrong_ranks_or_roles_refused(self):
        one_rank = {"0": {"role": "prefill", "ip": "10.0.0.1"}}
        with self.assertRaisesRegex(RuntimeError, "ranks/roles"):
            self._preflight(config=_fleet_config(ranks=one_rank))
        swapped = {
            "0": {"role": "decode", "ip": "10.0.0.1"},
            "1": {"role": "prefill", "ip": "10.0.0.2"},
            "2": {"role": "prefill", "ip": "10.0.0.3"},
            "3": {"role": "decode", "ip": "10.0.0.4"},
        }
        with self.assertRaisesRegex(RuntimeError, "ranks/roles"):
            self._preflight(config=_fleet_config(ranks=swapped))

    def test_duplicate_worker_ips_refused(self):
        ranks = {
            "0": {"role": "prefill", "ip": "10.0.0.1"},
            "1": {"role": "prefill", "ip": "10.0.0.1"},
            "2": {"role": "decode", "ip": "10.0.0.3"},
            "3": {"role": "decode", "ip": "10.0.0.4"},
        }
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            self._preflight(config=_fleet_config(ranks=ranks))

    def test_full_run_refuses_non_fresh_dir(self):
        (self.ramp_dir / "stage2").mkdir()
        with self.assertRaisesRegex(RuntimeError, "not fresh"):
            self._preflight()

    def test_wrong_remote_script_digest_refused(self):
        with self.assertRaisesRegex(RuntimeError, "tracked script"):
            self._preflight(script_digest="0" * 64)

    def test_wrong_sglang_version_refused(self):
        with self.assertRaisesRegex(RuntimeError, "sglang identity"):
            self._preflight(
                identity={"sglang_version": "0.5.15", "record_sha256": "a" * 64}
            )

    def test_resume_without_ledger_refused(self):
        with self.assertRaisesRegex(RuntimeError, "no ledger"):
            self._preflight(start_stage=3)

    def test_resume_with_mismatched_manifest_refused(self):
        self._preflight()  # writes a fresh, matching manifest + no ledger yet
        (self.ramp_dir / "ledger.json").write_text("{}")
        manifest = json.loads((self.ramp_dir / "manifest.json").read_text())
        manifest["worker_script_sha256"] = "0" * 64
        (self.ramp_dir / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "does not match the live"):
            self._preflight(start_stage=2)

    def test_resume_with_invalid_prior_stage_refused(self):
        self._preflight()
        (self.ramp_dir / "ledger.json").write_text("{}")
        (self.ramp_dir / "stage1").mkdir()
        with self.assertRaisesRegex(RuntimeError, "fails validation"):
            self._preflight(start_stage=2, stage_valid=False)

    def test_resume_with_missing_prior_stage_refused(self):
        self._preflight()
        (self.ramp_dir / "ledger.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "evidence missing"):
            self._preflight(start_stage=2)


if __name__ == "__main__":
    unittest.main()
