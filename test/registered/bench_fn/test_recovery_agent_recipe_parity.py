"""Static parity between the recovery-agent PD recipe table and its
executable launch scripts.

The parity/deviation table is the recipe's contract: every row marked
``carried`` must correspond to a flag the launch scripts actually emit, and
every ``--flag`` a launch script emits must be covered somewhere in the
README (a table row or the documented harness-flag note). Guards against the
table and the executable recipe drifting apart (bug regression: DP/EP/MoE
rows were once marked carried while no script emitted them).
"""

import re
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RECIPE_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "recovery_agent_pd"

_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]+")


def _table_rows(readme_text):
    for line in readme_text.splitlines():
        if (
            not line.startswith("| ")
            or line.startswith("| Reference")
            or set(line.replace("|", "").strip()) <= {"-"}
        ):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 3:
            yield cells


class TestRecipeParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (RECIPE_DIR / "README.md").read_text()
        cls.scripts = "\n".join(
            path.read_text() for path in sorted(RECIPE_DIR.glob("*.sh"))
        )

    def test_carried_rows_appear_in_launch_scripts(self):
        for reference, placement, status in _table_rows(self.readme):
            if not status.startswith("carried"):
                continue
            flags = _FLAG_RE.findall(placement)
            if not flags:
                # Placement says "same": derive the flag from the reference
                # column's `name: value` spelling.
                match = re.search(r"`([a-z][a-z0-9-]+):", reference)
                if match is None:
                    continue
                flags = [f"--{match.group(1)}"]
            for flag in flags:
                self.assertIn(
                    flag,
                    self.scripts,
                    f"table marks {flag!r} as carried "
                    f"(row: {reference[:60]}...) but no launch script emits it",
                )

    def test_deviation_rows_are_not_emitted_by_pd_worker(self):
        # The prefill/decode tables describe launch_glm_pd_worker.sh: rows
        # explicitly marked as omitted deviations must not silently reappear
        # there. (The unified shakeout script may step to reference values by
        # design; the tables do not describe it.)
        pd_worker = (RECIPE_DIR / "launch_glm_pd_worker.sh").read_text()
        for reference, placement, status in _table_rows(self.readme):
            if not status.startswith("deviation"):
                continue
            if placement in ("—", "-", ""):
                match = re.search(r"`([a-z][a-z0-9-]+):", reference)
                if match is None:
                    continue
                flag = f"--{match.group(1)}"
                self.assertNotIn(
                    flag,
                    pd_worker,
                    f"{flag!r} is documented as omitted (deviation) but the "
                    "PD worker script emits it",
                )

    def test_every_emitted_flag_is_documented(self):
        for flag in sorted(set(_FLAG_RE.findall(self.scripts))):
            self.assertIn(
                flag.lstrip("-"),
                self.readme,
                f"launch scripts emit {flag!r} but the README neither lists "
                "it in a table row nor documents it as a harness flag",
            )


if __name__ == "__main__":
    unittest.main()
