#!/usr/bin/env python3
"""Run only the selected transformers fallback unittest methods.

Examples:
  python3 scripts/playground/run_transformers_fallback_selected_tests.py \
    --class torchao --tests gsm8k mmlu

  python3 scripts/playground/run_transformers_fallback_selected_tests.py \
    --class torchao --tests mmlu
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import unittest
from pathlib import Path


def _detect_repo_root() -> Path:
    env_repo_root = os.getenv("SGLANG_REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root).resolve()

    cwd_repo_root = Path.cwd().resolve()
    if (cwd_repo_root / "test" / "registered" / "models" / "test_transformers_models.py").exists():
        return cwd_repo_root

    return Path(__file__).resolve().parents[2]


REPO_ROOT = _detect_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TEST_FILE = REPO_ROOT / "test" / "registered" / "models" / "test_transformers_models.py"


def _load_test_module():
    spec = importlib.util.spec_from_file_location(
        "transformers_fallback_tests",
        TEST_FILE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load test module from {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_test_module = _load_test_module()
TestTransformersFallbackEndpoint = _test_module.TestTransformersFallbackEndpoint
TestTransformersFallbackTorchAO = _test_module.TestTransformersFallbackTorchAO


CLASS_CHOICES = {
    "endpoint": TestTransformersFallbackEndpoint,
    "torchao": TestTransformersFallbackTorchAO,
}

TEST_METHODS = {
    "gsm8k": "test_gsm8k",
    "mmlu": "test_mmlu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only selected transformers fallback tests."
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        choices=sorted(CLASS_CHOICES),
        default="torchao",
        help="Which unittest class to run.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=("gsm8k", "mmlu"),
        default=("gsm8k", "mmlu"),
        help="Ordered list of test methods to run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_class = CLASS_CHOICES[args.class_name]

    suite = unittest.TestSuite()
    for test_name in args.tests:
        suite.addTest(test_class(TEST_METHODS[test_name]))

    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
