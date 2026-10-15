"""Startup-only regressions; execute the real method without torch/native imports."""

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[4] / (
    "python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py"
)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Store:
    def __init__(self, clock, actions=None):
        self.clock = clock
        self.actions = {key: list(value) for key, value in (actions or {}).items()}
        self.calls = []
        self.value = None

    def call(self, stage, key, default):
        self.calls.append((stage, key))
        outcomes = self.actions.get(stage, [])
        result = outcomes.pop(0) if outcomes else default
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result()
        return result

    def put(self, key, value):
        result = self.call("put", key, 0)
        if result == 0 and self.value is None:
            # Native Put returns 0 for an existing key without replacing it.
            self.value = value
        return result

    def is_exist(self, key):
        return self.call("is_exist", key, 1)

    def get(self, key):
        return self.call("get", key, self.value)

    def remove(self, key):
        return self.call("remove", key, 0)


def fixture(actions=None, env=None):
    clock = Clock()
    store = Store(clock, actions)
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "warmup"
    )
    namespace = {
        "time": clock,
        "os": SimpleNamespace(environ=env or {}),
        "uuid": SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="test-only-key")),
        "logger": logging.getLogger("warmup-regression"),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    owner = SimpleNamespace(store=store, local_rank=0)
    return lambda: namespace["warmup"](owner), store, clock


class WarmupTests(unittest.TestCase):
    def test_normal_once_and_cleanup(self):
        run, store, clock = fixture()
        run()
        self.assertEqual(
            [s for s, _ in store.calls], ["put", "is_exist", "get", "remove"]
        )
        self.assertEqual(clock.sleeps, [])

    def test_first_empty_retries_complete_roundtrip_same_key(self):
        run, store, clock = fixture({"get": [b""]})
        run()
        self.assertEqual(
            [s for s, _ in store.calls], ["put", "is_exist", "get"] * 2 + ["remove"]
        )
        self.assertEqual(len({k for _, k in store.calls}), 1)
        self.assertEqual(clock.sleeps, [1.0])

    def test_put_and_existence_failures_recover(self):
        for stage, code in [("put", -901), ("is_exist", 0), ("is_exist", -703)]:
            with self.subTest(stage=stage, code=code):
                run, store, _ = fixture({stage: [code]})
                run()
                self.assertEqual(sum(s == "put" for s, _ in store.calls), 2)

    def test_permanent_failures_are_not_ready(self):
        for stage, result in [("put", -901), ("is_exist", 0), ("get", b"")]:
            with self.subTest(stage=stage):
                run, store, clock = fixture({stage: [result] * 10})
                with self.assertRaisesRegex(RuntimeError, "after 10 attempts"):
                    run()
                self.assertEqual(sum(s == "put" for s, _ in store.calls), 10)
                self.assertEqual(sum(s == "remove" for s, _ in store.calls), 1)
                self.assertEqual(len(clock.sleeps), 9)

    def test_budget_no_next_call_and_late_success_rejected(self):
        for stage, result in [("put", 0), ("is_exist", 1), ("get", bytes(4096))]:
            with self.subTest(stage=stage):
                run, store, clock = fixture()

                def late():
                    clock.now = 11.0
                    return result

                store.actions[stage] = [late]
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    run()
                stages = [s for s, _ in store.calls]
                self.assertEqual(stages[-2:], [stage, "remove"])
                self.assertEqual(clock.sleeps, [])

    def test_wrong_nonempty_value_and_types_fail_immediately(self):
        for value in [b"wrong", b"x" * 4096, None, bytearray(4096), ""]:
            with self.subTest(type=type(value).__name__):
                run, store, clock = fixture({"get": [value]})
                with self.assertRaises(RuntimeError):
                    run()
                self.assertEqual(sum(s == "get" for s, _ in store.calls), 1)
                self.assertEqual(clock.sleeps, [])

    def test_existing_different_value_not_overwritten_or_accepted(self):
        run, store, _ = fixture()
        store.value = b"unexpected-existing-value"
        with self.assertRaisesRegex(RuntimeError, "nonempty value mismatch"):
            run()

    def test_native_exceptions_and_cancellation_preserved(self):
        for stage in ["put", "is_exist", "get"]:
            for exception in [
                ValueError("private-native-message"),
                KeyboardInterrupt(),
                asyncio.CancelledError(),
            ]:
                with self.subTest(stage=stage, exception=type(exception).__name__):
                    run, store, _ = fixture(
                        {stage: [exception], "remove": [RuntimeError("cleanup-secret")]}
                    )
                    with self.assertRaises(type(exception)) as caught:
                        run()
                    self.assertIs(caught.exception, exception)
                    self.assertEqual(store.calls[-1][0], "remove")

    def test_cleanup_failures_do_not_erase_verified_success(self):
        for outcome in [-704, -901, RuntimeError("private-cleanup-message")]:
            with self.subTest(outcome=type(outcome).__name__):
                run, store, _ = fixture({"remove": [outcome]})
                run()
                self.assertEqual(sum(s == "remove" for s, _ in store.calls), 1)

    def test_cleanup_cancel_on_success_propagates(self):
        cancel = asyncio.CancelledError()
        run, _, _ = fixture({"remove": [cancel]})
        with self.assertRaises(asyncio.CancelledError) as caught:
            run()
        self.assertIs(caught.exception, cancel)

    def test_cleanup_cancel_does_not_replace_original_failure(self):
        original = ValueError("private-primary")
        run, _, _ = fixture({"get": [original], "remove": [asyncio.CancelledError()]})
        with self.assertRaises(ValueError) as caught:
            run()
        self.assertIs(caught.exception, original)

    def test_retry_sleep_cancellation_preserved(self):
        run, store, clock = fixture({"get": [b""]})
        cancel = asyncio.CancelledError()

        def cancel_sleep(seconds):
            raise cancel

        clock.sleep = cancel_sleep
        with self.assertRaises(asyncio.CancelledError) as caught:
            run()
        self.assertIs(caught.exception, cancel)
        self.assertEqual(store.calls[-1][0], "remove")

    def test_invalid_status_types_fail_without_retry(self):
        for stage in ["put", "is_exist"]:
            run, _, clock = fixture({stage: ["private-status"]})
            with self.assertRaisesRegex(RuntimeError, "result type"):
                run()
            self.assertEqual(clock.sleeps, [])

    def test_disabled_or_invalid_rpc_deadline_skips_cleanup(self):
        for timeout in ["-1", "invalid"]:
            run, store, _ = fixture(env={"MC_RPC_TIMEOUT_MS": timeout})
            run()
            self.assertNotIn("remove", [s for s, _ in store.calls])

    def test_diagnostics_exclude_key_and_payload(self):
        run, _, _ = fixture({"get": [b""], "remove": [ValueError("secret-cleanup")]})
        with self.assertLogs("warmup-regression", level="WARNING") as captured:
            run()
        output = "\n".join(captured.output)
        self.assertIn("get_ambiguous_empty", output)
        self.assertNotIn("test-only-key", output)
        self.assertNotIn("secret-cleanup", output)


if __name__ == "__main__":
    unittest.main()
