"""Role/stage-exact parity between the recovery-agent PD recipe table and the
argv its launch scripts actually emit.

Every launch script supports ``DRY_RUN=1``, which prints each command it would
run as a ``#COMMAND <label>`` header followed by one argv token per line. The
tests run the full command matrix (PD worker prefill/decode x stages 1-4, the
unified shakeout stages, the small 2P2D topology script, and the router) and
check the README parity/deviation table against the exact applicable argv:

- every ``carried``/``addition`` row's flag AND value appear in the argv of
  each command the row applies to (role table x stage annotation);
- every omitted ``deviation`` row's flag is absent from every applicable
  command, and replacement deviations carry their replacement value;
- every flag a command emits maps to exactly one row of its role table (or,
  failing that, to the documented harness-flag note) — no undocumented flags,
  no ambiguously documented flags;
- consecutive PD stages differ by exactly the documented feature delta, so a
  stage regression cannot masquerade as its neighbor (bug regression: stage 1
  once ran the v0.5.16 default attention, which auto-selects DSA for this
  model, making the "baseline" indistinguishable from stage 2).

Guards against the table and the executable recipe drifting apart (bug
regression: DP/EP/MoE rows were once marked carried while no script emitted
them, and an aggregate text check could not see which role or stage a flag
belonged to).
"""

import os
import re
import subprocess
import unittest
from pathlib import Path

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RECIPE_DIR = Path(__file__).resolve().parents[3] / "benchmark" / "recovery_agent_pd"

PD_STAGES = (1, 2, 3, 4)
UNIFIED_STAGES = ("verified", "doc-spec", "doc-full")

# `name: value` spans in a reference cell and `--flag value` spans anywhere.
_REF_PAIR_RE = re.compile(r"`([a-z][a-z0-9-]+): ([^`]+)`")
_FLAG_PAIR_RE = re.compile(r"--([a-z][a-z0-9-]+)(?: ([^\s`,*]+))?")
_STAGE_RE = re.compile(r"stage (\d)")


class TableRow(msgspec.Struct, frozen=True):
    section: str
    reference: str
    placement: str
    status: str

    def expected_pairs(self):
        """(--flag, value|None) pairs the row promises, placement spelling
        first (it overrides the reference spelling, e.g. `--tp 4`)."""
        pairs = _flag_pairs(self.placement)
        if not pairs:
            pairs = _ref_pairs(self.reference)
        if not pairs:
            pairs = _flag_pairs(self.reference)
        return pairs

    def mentioned_flags(self):
        """Every flag this row documents, under any spelling it uses."""
        return {
            flag
            for source in (self.placement, self.reference)
            for flag, _ in _flag_pairs(source) + _ref_pairs(source)
        }

    def min_stage(self):
        match = _STAGE_RE.search(self.status)
        return int(match.group(1)) if match else 1


def _ref_pairs(cell):
    pairs = []
    for name, value in _REF_PAIR_RE.findall(cell):
        value = value.strip().strip("'")
        pairs.append((f"--{name}", None if value == "true" else value))
    return pairs


def _flag_pairs(cell):
    return [
        (f"--{name}", value or None) for name, value in _FLAG_PAIR_RE.findall(cell)
    ]


def _parse_tables(readme_text):
    rows, section = [], ""
    for line in readme_text.splitlines():
        if line.startswith("### "):
            section = line[4:].strip()
        if (
            not line.startswith("| ")
            or line.startswith("| Reference")
            or set(line.replace("|", "").strip()) <= {"-"}
        ):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 3:
            rows.append(TableRow(section, *cells))
    return rows


def _note_flags(readme_text):
    notes = readme_text.split("\nNotes:\n", 1)[1]
    return {f"--{name}" for name, _ in _FLAG_PAIR_RE.findall(notes)}


def _argmap(argv):
    """argv tokens -> {--flag: value}. A flag's value is None (boolean), the
    single following token, or a tuple of following tokens (multi-value flags
    like the router's ``--prefill URL PORT``). A repeated flag maps to a list
    with one entry per occurrence."""
    occurrences, i = [], 0
    while i < len(argv):
        flag = argv[i]
        assert flag.startswith("--"), f"expected a flag token, got {flag!r}"
        values = []
        i += 1
        while i < len(argv) and not argv[i].startswith("--"):
            values.append(argv[i])
            i += 1
        if not values:
            occurrences.append((flag, None))
        elif len(values) == 1:
            occurrences.append((flag, values[0]))
        else:
            occurrences.append((flag, tuple(values)))
    mapping = {}
    for flag, value in occurrences:
        if flag in mapping:
            existing = mapping[flag]
            mapping[flag] = (
                existing + [value] if isinstance(existing, list) else [existing, value]
            )
        else:
            mapping[flag] = value
    return mapping


def _dry_run(script, env, commands):
    result = subprocess.run(
        ["bash", script],
        env={**os.environ, **env, "DRY_RUN": "1"},
        capture_output=True,
        text=True,
        cwd=RECIPE_DIR,
        check=True,
    )
    label = None
    for line in result.stdout.splitlines():
        if line.startswith("#COMMAND "):
            label = line.split(" ", 1)[1]
            commands[label] = []
        elif label is not None:
            commands[label].append(line)


def _command_matrix():
    commands = {}
    for role in ("prefill", "decode"):
        for stage in PD_STAGES:
            _dry_run(
                "launch_glm_pd_worker.sh",
                {"ROLE": role, "STAGE": str(stage)},
                commands,
            )
    for stage in UNIFIED_STAGES:
        _dry_run("launch_glm_tp4_unified.sh", {"STAGE": stage}, commands)
    _dry_run("launch_small_2p2d.sh", {}, commands)
    _dry_run("launch_router.sh", {}, commands)
    return {label: _argmap(argv) for label, argv in commands.items()}


class TestRecipeParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        readme = (RECIPE_DIR / "README.md").read_text()
        cls.rows = _parse_tables(readme)
        cls.note_flags = _note_flags(readme)
        cls.commands = _command_matrix()

    # -- applicability -----------------------------------------------------

    def _pd_labels(self, role, min_stage=1):
        return [f"pd-worker:{role}:stage{s}" for s in PD_STAGES if s >= min_stage]

    def _row_targets(self, row):
        """Command labels a row's carried promise applies to."""
        if row.section == "Prefill workers":
            return self._pd_labels("prefill", row.min_stage())
        if row.section == "Decode workers":
            return self._pd_labels("decode", row.min_stage())
        if row.section == "Frontend / router":
            if "worker-side" in row.placement:
                return self._pd_labels("prefill") + self._pd_labels("decode")
            return ["router"]
        return []  # Worker environment: env vars, not argv

    def _row_absence_targets(self, row):
        """Command labels an omitted-deviation row must stay absent from
        (all stages: an omission has no stage threshold)."""
        if row.section == "Prefill workers":
            return self._pd_labels("prefill")
        if row.section == "Decode workers":
            return self._pd_labels("decode")
        if row.section == "Frontend / router":
            return ["router"]
        return []

    # -- checks ------------------------------------------------------------

    def test_carried_rows_match_exact_applicable_argv(self):
        for row in self.rows:
            if not (
                row.status.startswith("carried") or row.status.startswith("addition")
            ):
                continue
            pairs = row.expected_pairs()
            targets = self._row_targets(row)
            self.assertTrue(
                not pairs or targets,
                f"carried row has no applicable command: {row.reference[:60]}",
            )
            for label in targets:
                argmap = self.commands[label]
                for flag, value in pairs:
                    self.assertIn(
                        flag,
                        argmap,
                        f"{label} does not emit {flag!r} "
                        f"(carried row: {row.reference[:60]}...)",
                    )
                    if value is not None:
                        self.assertEqual(
                            argmap[flag],
                            value,
                            f"{label} emits {flag} {argmap[flag]!r}, "
                            f"table promises {value!r}",
                        )

    def test_deviation_rows_are_omitted_or_replaced_exactly(self):
        for row in self.rows:
            if row.status.startswith("deviation"):
                replacement = _flag_pairs(row.placement)
                if replacement:
                    for label in self._row_targets(row):
                        argmap = self.commands[label]
                        for flag, value in replacement:
                            self.assertIn(flag, argmap, f"{label} misses {flag!r}")
                            if value is not None:
                                self.assertEqual(argmap[flag], value, label)
                else:
                    absent = {flag for flag, _ in _ref_pairs(row.reference)} | {
                        flag for flag, _ in _flag_pairs(row.reference)
                    }
                    for label in self._row_absence_targets(row):
                        for flag in absent:
                            self.assertNotIn(
                                flag,
                                self.commands[label],
                                f"{flag!r} is documented as omitted (deviation) "
                                f"but {label} emits it",
                            )
            elif row.status.startswith("intentionally absent"):
                for flag, _ in _flag_pairs(row.status):
                    for label in self._row_absence_targets(row):
                        self.assertNotIn(flag, self.commands[label])

    def test_every_emitted_flag_maps_to_exactly_one_record(self):
        worker_side = [
            r
            for r in self.rows
            if r.section == "Frontend / router" and "worker-side" in r.placement
        ]
        role_tables = {
            "prefill": [r for r in self.rows if r.section == "Prefill workers"]
            + worker_side,
            "decode": [r for r in self.rows if r.section == "Decode workers"]
            + worker_side,
            "router": [
                r
                for r in self.rows
                if r.section == "Frontend / router" and r not in worker_side
            ],
        }
        strict = {
            **{
                f"pd-worker:{role}:stage{s}": role_tables[role]
                for role in ("prefill", "decode")
                for s in PD_STAGES
            },
            "router": role_tables["router"],
        }
        all_row_flags = {
            flag for row in self.rows for flag in row.mentioned_flags()
        }
        for label, argmap in self.commands.items():
            if label in strict:
                for flag in argmap:
                    covering = [
                        row
                        for row in strict[label]
                        if flag in row.mentioned_flags()
                    ]
                    if covering:
                        self.assertEqual(
                            len(covering),
                            1,
                            f"{label}: {flag!r} is covered by "
                            f"{len(covering)} table rows (must be exactly one)",
                        )
                    else:
                        self.assertIn(
                            flag,
                            self.note_flags,
                            f"{label} emits {flag!r} but it appears in no "
                            "table row and not in the harness-flag note",
                        )
            else:
                # Unified/small shakeout commands: every flag must still be
                # documented, in any table or the note.
                for flag in argmap:
                    self.assertIn(
                        flag,
                        all_row_flags | self.note_flags,
                        f"{label} emits undocumented flag {flag!r}",
                    )

    def test_stage_deltas_are_exactly_the_documented_features(self):
        """AC-10.3 attribution property: each PD stage differs from its
        predecessor by exactly the feature the README stage list documents."""

        def delta(role, prev, curr):
            before = self.commands[f"pd-worker:{role}:stage{prev}"]
            after = self.commands[f"pd-worker:{role}:stage{curr}"]
            changed = {}
            for flag in set(before) | set(after):
                if before.get(flag, "<absent>") != after.get(flag, "<absent>"):
                    changed[flag] = (
                        before.get(flag, "<absent>"),
                        after.get(flag, "<absent>"),
                    )
            return changed

        self.assertEqual(
            delta("prefill", 1, 2),
            {
                "--attention-backend": ("triton", "dsa"),
                "--dsa-prefill-backend": ("<absent>", "trtllm"),
            },
        )
        self.assertEqual(
            delta("decode", 1, 2), {"--attention-backend": ("triton", "dsa")}
        )
        eagle = {
            "--speculative-algorithm": ("<absent>", "EAGLE"),
            "--speculative-num-steps": ("<absent>", "3"),
            "--speculative-eagle-topk": ("<absent>", "1"),
            "--speculative-num-draft-tokens": ("<absent>", "4"),
        }
        self.assertEqual(delta("prefill", 2, 3), eagle)
        self.assertEqual(
            delta("decode", 2, 3),
            {**eagle, "--speculative-attention-mode": ("<absent>", "decode")},
        )
        self.assertEqual(
            delta("prefill", 3, 4),
            {
                "--enable-hierarchical-cache": ("<absent>", None),
                "--hicache-size": ("<absent>", "32"),
                "--hicache-io-backend": ("<absent>", "direct"),
                "--hicache-mem-layout": ("<absent>", "page_first_direct"),
                "--hicache-write-policy": ("<absent>", "write_back"),
            },
        )
        self.assertEqual(delta("decode", 3, 4), {})


if __name__ == "__main__":
    unittest.main()
