"""Fail-closed evidence validator for the controlled 2P2D recovery-agent ramp.

A ramp stage counts only if EVERY required artifact exists and is internally
consistent; anything missing, vacuous, or contradictory is a hard failure,
never a silent gap. The ramp driver runs this after each stage (aborting the
ramp on failure) and once globally as the final gate; it can also be run
standalone against an archived evidence directory:

    python3 validate_ramp_evidence.py --ramp-dir DIR [--stage N] [--json OUT]
        [--expect-commit SHA]

Per stage it requires, for ALL four workers: a resolved ``server_args`` line,
a nonempty distinct startup log, and the stage's exact POSITIVE and NEGATIVE
configuration markers (attention backend, DSA prefill backend, the full EAGLE
3-1-4 values, every HiCache value on stage-4+ prefill, and HiCache absence on
decode and pre-stage-4 prefill). It further requires raw pre/post counter
snapshots with all four workers, numeric fields, per-worker monotonicity, and
a pool request delta covering the stage's planned rounds; benchmark records
that are usage-complete, error-free, and match the retained population's
planned round counts exactly; passing wire probes; and for the final stage a
per-key router-attribution artifact with per-record stickiness, two-probe
stability, and both-pool spread. Dataset-sha-vs-manifest and
manifest-commit-vs-HEAD identity checks run in every mode, including
``--stage``. Exit status is nonzero if any check fails.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

RANK_ROLES = {0: "prefill", 1: "prefill", 2: "decode", 3: "decode"}
DATASET_NAME = "recovery_agent_fixed_32.json"
PREFILL_KEYS = ("P0", "P1")
DECODE_KEYS = ("D2", "D3")
WORKER_KEYS = PREFILL_KEYS + DECODE_KEYS
COUNTER_FIELDS = ("chat_requests", "prompt_tokens", "cached_tokens")
FINAL_STAGE = 5
SMALL_STAGE_SESSIONS = 8
FINAL_STAGE_SESSIONS = 32


def _field(name, value):
    """Anchored marker regex: matches the exact ServerArgs field assignment,
    not a longer field name that merely ends with it (e.g. the plain
    ``attention_backend`` must not match ``prefill_attention_backend`` or
    ``dsa_prefill_backend``)."""
    return rf"(?<![a-z_]){re.escape(name)}={re.escape(value)}(?=[,)])"


_EAGLE = [
    _field("speculative_algorithm", "'EAGLE'"),
    _field("speculative_num_steps", "3"),
    _field("speculative_eagle_topk", "1"),
    _field("speculative_num_draft_tokens", "4"),
]
_NO_SPEC = [_field("speculative_algorithm", "None")]
_HICACHE = [
    _field("enable_hierarchical_cache", "True"),
    _field("hicache_size", "32"),
    _field("hicache_io_backend", "'direct'"),
    _field("hicache_mem_layout", "'page_first_direct'"),
    _field("hicache_write_policy", "'write_back'"),
]
_NO_HICACHE = [_field("enable_hierarchical_cache", "False")]
_TRITON = [_field("attention_backend", "'triton'")]
_DSA = [_field("attention_backend", "'dsa'")]
_DSA_PREFILL = [_field("dsa_prefill_backend", "'trtllm'")]
_SPEC_DECODE = [_field("speculative_attention_mode", "'decode'")]


def stage_markers(stage, role):
    """(positive, negative) marker regexes the resolved args must satisfy:
    the stage delta must be attributable from retained evidence, not launch
    intent, and a feature from a LATER stage must not leak in early."""
    if stage == 1:
        return _TRITON + _NO_SPEC + _NO_HICACHE, _DSA + _EAGLE[:1]
    if stage == 2:
        positive = _DSA + _NO_SPEC + _NO_HICACHE
        if role == "prefill":
            positive = positive + _DSA_PREFILL
        return positive, _TRITON + _EAGLE[:1]
    if stage == 3:
        positive = _DSA + _EAGLE + _NO_HICACHE
        if role == "prefill":
            positive = positive + _DSA_PREFILL
        else:
            positive = positive + _SPEC_DECODE
        return positive, _TRITON
    # Stages 4/5: full final configuration; HiCache on prefill ONLY.
    if role == "prefill":
        return _DSA + _DSA_PREFILL + _EAGLE + _HICACHE, _TRITON
    return _DSA + _EAGLE + _SPEC_DECODE + _NO_HICACHE, _TRITON


def _plain(marker):
    """Human-readable form of a marker regex for check names."""
    return marker.replace(r"(?<![a-z_])", "").replace(r"(?=[,)])", "").replace("\\", "")


def _check(results, name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    return ok


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _bench_last(path):
    try:
        return json.loads(path.read_text().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None


def _planned_rounds(ramp_dir):
    """Per-session planned round counts from the retained fixed population;
    None if the retained copy is missing/unreadable (dependent checks fail)."""
    dataset = _read_json(ramp_dir / DATASET_NAME)
    if not dataset or "conversations" not in dataset:
        return None
    return [len(conversation) for conversation in dataset["conversations"]]


def validate_worker_artifacts(results, stage_dir, stage):
    heads = set()
    for rank, role in RANK_ROLES.items():
        args_path = stage_dir / f"server_args_rank{rank}_{role}.txt"
        log_path = stage_dir / f"startup_rank{rank}_{role}.log"
        args_text = args_path.read_text() if args_path.exists() else ""
        resolved = "server_args=ServerArgs(" in args_text
        log_nonempty = log_path.exists() and log_path.stat().st_size > 0
        if log_nonempty:
            heads.add(log_path.read_text()[:200])
        _check(
            results,
            f"stage{stage}: server_args rank{rank} resolved",
            resolved,
            str(args_path),
        )
        _check(
            results,
            f"stage{stage}: startup log rank{rank} nonempty",
            log_nonempty,
            str(log_path),
        )
        positive, negative = stage_markers(stage, role)
        for marker in positive:
            _check(
                results,
                f"stage{stage}: rank{rank} requires {_plain(marker)}",
                re.search(marker, args_text) is not None,
                str(args_path),
            )
        for marker in negative:
            _check(
                results,
                f"stage{stage}: rank{rank} must NOT resolve {_plain(marker)}",
                resolved and re.search(marker, args_text) is None,
                str(args_path),
            )
    _check(
        results,
        f"stage{stage}: per-rank startup logs are distinct captures",
        len(heads) == 4,
        f"{len(heads)}/4 distinct heads",
    )


def _snapshot_ok(snapshot):
    return (
        isinstance(snapshot, dict)
        and set(snapshot) >= set(WORKER_KEYS)
        and all(
            isinstance(snapshot[key], dict)
            and isinstance(snapshot[key].get(field), (int, float))
            for key in WORKER_KEYS
            for field in COUNTER_FIELDS
        )
    )


def validate_counters(results, stage_dir, stage, expected_rounds):
    pre = _read_json(stage_dir / "counters_pre.json")
    post = _read_json(stage_dir / "counters_post.json")
    complete = _check(
        results,
        f"stage{stage}: counter snapshots complete (all four workers, numeric)",
        _snapshot_ok(pre) and _snapshot_ok(post),
        f"{stage_dir}/counters_pre.json,counters_post.json",
    )
    if not complete:
        return
    monotonic = all(
        post[key][field] >= pre[key][field]
        for key in WORKER_KEYS
        for field in ("chat_requests", "prompt_tokens")
    )
    _check(results, f"stage{stage}: per-worker counters monotonic", monotonic)
    prefill_delta = sum(
        post[key]["chat_requests"] - pre[key]["chat_requests"] for key in PREFILL_KEYS
    )
    decode_delta = sum(
        post[key]["chat_requests"] - pre[key]["chat_requests"] for key in DECODE_KEYS
    )
    prompt_delta = sum(
        post[key]["prompt_tokens"] - pre[key]["prompt_tokens"] for key in PREFILL_KEYS
    )
    _check(
        results,
        f"stage{stage}: pool request deltas cover the planned rounds",
        expected_rounds is not None
        and prefill_delta >= expected_rounds
        and decode_delta >= expected_rounds
        and prompt_delta > 0,
        f"prefill {prefill_delta}, decode {decode_delta} vs planned "
        f"{expected_rounds}; prompt tokens {prompt_delta}",
    )


def _bench_complete(bench, sessions, planned_rounds):
    return (
        isinstance(bench, dict)
        and bench.get("completed_conversations")
        == bench.get("total_conversations")
        == sessions
        and bench.get("input_metrics_complete")
        # Per-round error slots: empty string == no error for that round.
        and not any(bench.get("errors") or [])
        and planned_rounds is not None
        and bench.get("completed") == planned_rounds
    )


def validate_wire(results, stage_dir, stage):
    wire = _read_json(stage_dir / "wire_probe.json")
    _check(
        results,
        f"stage{stage}: wire probe passed",
        isinstance(wire, dict)
        and wire.get("reasoning_absent_from_round2")
        and wire.get("round2_assistant_equals_content_only")
        and wire.get("rounds_succeeded"),
        f"{stage_dir}/wire_probe.json",
    )


def validate_small_stage(results, stage_dir, stage, planned_rounds):
    validate_worker_artifacts(results, stage_dir, stage)
    expected = sum(planned_rounds[:SMALL_STAGE_SESSIONS]) if planned_rounds else None
    validate_counters(results, stage_dir, stage, expected)
    affinity = _read_json(stage_dir / "affinity.json")
    _check(
        results,
        f"stage{stage}: 8/8 sessions sticky (rc=0) with pool spread",
        isinstance(affinity, dict)
        and affinity.get("all_sticky")
        and affinity.get("spread_ok")
        and len(affinity.get("sessions", [])) == SMALL_STAGE_SESSIONS
        and all(v.get("bench_rc") == 0 for v in affinity["sessions"]),
        f"{stage_dir}/affinity.json",
    )
    for session in range(SMALL_STAGE_SESSIONS):
        bench = _bench_last(stage_dir / f"bench_s{session}.jsonl")
        rounds = planned_rounds[session] if planned_rounds else None
        _check(
            results,
            f"stage{stage}: bench_s{session} usage-complete, error-free, "
            "exact planned rounds",
            _bench_complete(bench, sessions=1, planned_rounds=rounds),
            f"{stage_dir}/bench_s{session}.jsonl",
        )
    validate_wire(results, stage_dir, stage)


def _probed_workers(records, field):
    """Distinct workers named by `field` across every probe of every record."""
    return {
        worker
        for record in records
        for probe in record.get("probes", [])
        for worker in probe.get(field, [])
    }


def validate_final_stage(results, stage_dir, planned_rounds):
    validate_worker_artifacts(results, stage_dir, FINAL_STAGE)
    total = sum(planned_rounds) if planned_rounds else None
    validate_counters(results, stage_dir, FINAL_STAGE, total)
    bench = _bench_last(stage_dir / "bench_32s.jsonl")
    _check(
        results,
        "stage5: 32/32 conversations usage-complete, error-free, exact rounds",
        _bench_complete(bench, sessions=FINAL_STAGE_SESSIONS, planned_rounds=total),
        f"{stage_dir}/bench_32s.jsonl",
    )
    attribution = _read_json(stage_dir / "attribution.json")
    records = attribution.get("keys", []) if isinstance(attribution, dict) else []
    per_record = bool(records) and all(
        record.get("sticky") and record.get("stable") for record in records
    )
    prefill_used = _probed_workers(records, "prefill_workers")
    decode_used = _probed_workers(records, "decode_workers")
    _check(
        results,
        "stage5: per-key attribution — every key sticky and two-probe stable",
        isinstance(attribution, dict)
        and attribution.get("all_sticky")
        and len(records) == FINAL_STAGE_SESSIONS
        and per_record,
        f"{stage_dir}/attribution.json",
    )
    _check(
        results,
        "stage5: attribution spread — both pools serve >= 2 workers",
        len(prefill_used) >= 2 and len(decode_used) >= 2,
        f"prefill {sorted(prefill_used)}, decode {sorted(decode_used)}",
    )
    validate_wire(results, stage_dir, FINAL_STAGE)


def validate_dataset_and_manifest(results, ramp_dir, expected_commit=None):
    manifest_path = ramp_dir / "manifest.txt"
    manifest = manifest_path.read_text() if manifest_path.exists() else ""
    sha_match = re.search(r"sha256=(\w{64})", manifest)
    dataset_path = ramp_dir / DATASET_NAME
    ok = False
    detail = str(dataset_path)
    if dataset_path.exists() and sha_match:
        digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        ok = digest == sha_match.group(1)
        detail = f"sha256={digest[:16]}… vs manifest {sha_match.group(1)[:16]}…"
    _check(
        results,
        "dataset: retained fixed population sha matches manifest",
        ok,
        detail,
    )
    commit_match = re.search(r"branch: \S+ @ (\w{40})", manifest)
    if expected_commit is None:
        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ramp_dir,
        ).stdout.strip()
    _check(
        results,
        "manifest: stamped with the exact running commit",
        commit_match is not None and commit_match.group(1) == expected_commit,
        f"manifest {commit_match.group(1)[:12] if commit_match else '<none>'} "
        f"vs expected {expected_commit[:12] if expected_commit else '<none>'}",
    )


def validate(ramp_dir, only_stage=None, expected_commit=None):
    results = []
    planned_rounds = _planned_rounds(ramp_dir)
    stages = [only_stage] if only_stage else [1, 2, 3, 4, FINAL_STAGE]
    for stage in stages:
        stage_dir = ramp_dir / f"stage{stage}"
        if not _check(results, f"stage{stage}: directory present", stage_dir.is_dir()):
            continue
        if stage == FINAL_STAGE:
            validate_final_stage(results, stage_dir, planned_rounds)
        else:
            validate_small_stage(results, stage_dir, stage, planned_rounds)
    # Identity checks run in EVERY mode: a per-stage gate that skips dataset
    # and commit identity is a fail-open gate.
    validate_dataset_and_manifest(results, ramp_dir, expected_commit)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramp-dir", required=True)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--expect-commit",
        default=None,
        help="Commit the manifest must be stamped with (default: git HEAD of "
        "the ramp dir's repository)",
    )
    args = parser.parse_args()
    results = validate(Path(args.ramp_dir), args.stage, args.expect_commit)
    failed = [r for r in results if not r["ok"]]
    for r in results:
        print(
            f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']}"
            + (f"  [{r['detail']}]" if r["detail"] and not r["ok"] else "")
        )
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"checks": results, "all_ok": not failed}, indent=2)
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
