"""Fail-closed evidence validator for the controlled 2P2D recovery-agent ramp.

A ramp stage counts only if EVERY required artifact exists and is internally
consistent; anything missing, vacuous, forged, or contradictory is a hard
failure, never a silent gap. The ramp driver runs this after each stage
(aborting the ramp on failure) and once globally as the final gate; it can
also be run standalone against an archived evidence directory:

    python3 validate_ramp_evidence.py --ramp-dir DIR [--stage N] [--json OUT]
        [--expect-commit SHA]

Contract highlights (all fail-closed):

- **Run identity** against the tracked ``ramp_run_spec.json``: retained
  dataset sha256 AND semantic identity (dataset id, profile, authority, seed,
  session count, nonempty conversations); a structured ``manifest.json``
  whose commit equals HEAD, whose worker-script sha equals the TRACKED
  ``launch_glm_pd_worker.sh``, and whose per-rank packages match the spec's
  sglang version with one RECORD digest across ranks; a router artifact
  proving 2 prefill + 2 decode consistent-hashing PD topology.
- **Per-rank resolved configuration**: common identity on every rank (spec
  model path, tp 4, modelopt_fp4, fp8_e4m3, the rank's disaggregation role,
  nixl transfer, glm45 reasoning parser, metrics + cache report) PLUS the
  stage's exact positive and negative feature markers (dense triton stage 1,
  DSA + trtllm prefill backend, full EAGLE 3-1-4 values, every HiCache value
  on stage-4+ prefill, HiCache absence on decode/earlier, no early leaks).
  Startup logs must be nonempty, pairwise distinct by full-content digest,
  and each must contain its rank's retained resolved-args line.
- **Verdicts derived from raw records, never trusted**: small-stage
  stickiness/spread is recomputed from each session's per-worker counter
  deltas (sessions 0-7 exactly, one worker per pool, hit sets consistent
  with deltas, stored summary bits must equal the derived result); final
  attribution requires the 32 routing keys derived from the retained
  dataset, exactly two successful probes per key, one known worker per pool
  per probe, identical pairs, both-pool spread, and stored bits equal to
  derived.
- **Structural, typed artifact checks**: benches must retain per-round
  arrays (errors REQUIRED and all-empty, positive finite prompt lengths,
  coherent session/round index sequences, all-success conversation_results
  matching the retained population's planned rounds); counters reject
  bool/NaN/Inf/negative values, require all three fields monotonic, pool
  request deltas covering the planned rounds, and cached-token deltas
  bounded by prompt-token deltas per worker.

Identity checks run in every mode, including ``--stage``. Exit status is
nonzero if any check fails.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

TRACKED_DIR = Path(__file__).resolve().parent
RANK_ROLES = {0: "prefill", 1: "prefill", 2: "decode", 3: "decode"}
DATASET_NAME = "recovery_agent_fixed_32.json"
PREFILL_KEYS = ("P0", "P1")
DECODE_KEYS = ("D2", "D3")
WORKER_KEYS = PREFILL_KEYS + DECODE_KEYS
COUNTER_FIELDS = ("chat_requests", "prompt_tokens", "cached_tokens")
FINAL_STAGE = 5
SMALL_STAGE_SESSIONS = 8


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


def common_markers(role, spec):
    """Runtime identity every rank must resolve at every stage: the ramp is
    only evidence for GLM/v0.5.16/TP4/2P2D if the args prove that deployment,
    not merely the staged feature delta."""
    return [
        _field("model_path", f"'{spec['model_path']}'"),
        _field("tp_size", "4"),
        _field("quantization", "'modelopt_fp4'"),
        _field("kv_cache_dtype", "'fp8_e4m3'"),
        _field("disaggregation_mode", f"'{role}'"),
        _field("disaggregation_transfer_backend", "'nixl'"),
        _field("reasoning_parser", "'glm45'"),
        _field("enable_metrics", "True"),
        _field("enable_cache_report", "True"),
    ]


def stage_markers(stage, role):
    """(positive, negative) feature markers for the staged delta: the stage
    must be attributable from retained evidence, not launch intent, and a
    feature from a LATER stage must not leak in early."""
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


def routing_key(seed, session_index):
    """Same derivation as recovery_agent._session_routing_key: the expected
    attribution keys are derived from the retained dataset, never trusted."""
    digest = hashlib.sha1(f"{seed}:{session_index}".encode("utf-8")).hexdigest()
    return f"session-{digest[:16]}"


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


def _counter_value_ok(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _positive_int_list(values, expected_len):
    return (
        isinstance(values, list)
        and len(values) == expected_len
        and all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values
        )
    )


# ---------------------------------------------------------------------------
# Run identity: spec, dataset semantics, manifest, router topology
# ---------------------------------------------------------------------------


def _dataset(ramp_dir):
    return _read_json(ramp_dir / DATASET_NAME)


def _planned_rounds(dataset):
    if not isinstance(dataset, dict) or "conversations" not in dataset:
        return None
    return [len(conversation) for conversation in dataset["conversations"]]


def validate_run_identity(results, ramp_dir, spec, expected_commit, tracked_script):
    dataset_path = ramp_dir / DATASET_NAME
    sha_ok = False
    detail = str(dataset_path)
    if dataset_path.exists():
        digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        sha_ok = digest == spec["dataset_sha256"]
        detail = f"sha256={digest[:16]}… vs spec {spec['dataset_sha256'][:16]}…"
    _check(
        results,
        "dataset: retained population sha matches the tracked run spec",
        sha_ok,
        detail,
    )
    dataset = _dataset(ramp_dir)
    meta = dataset.get("metadata", {}) if isinstance(dataset, dict) else {}
    profile = meta.get("profile") or {}
    expected = spec["dataset"]
    semantic_ok = (
        meta.get("dataset") == expected["id"]
        and profile.get("name") == expected["profile"]
        and profile.get("authority") == expected["authority"]
        and meta.get("seed") == expected["seed"]
        and meta.get("num_sessions") == expected["num_sessions"]
        and isinstance(dataset.get("conversations"), list)
        and len(dataset["conversations"]) == expected["num_sessions"]
        and all(isinstance(c, list) and len(c) > 0 for c in dataset["conversations"])
    )
    _check(
        results,
        "dataset: semantic identity (id/profile/authority/seed/sessions) "
        "matches the tracked run spec",
        semantic_ok,
        str(dataset_path),
    )

    manifest = _read_json(ramp_dir / "manifest.json")
    manifest_ok = isinstance(manifest, dict)
    _check(
        results,
        "manifest: structured manifest.json present",
        manifest_ok,
        str(ramp_dir / "manifest.json"),
    )
    if manifest_ok:
        _check(
            results,
            "manifest: stamped with the exact running commit",
            manifest.get("commit") == expected_commit,
            f"manifest {str(manifest.get('commit'))[:12]} vs expected "
            f"{expected_commit[:12] if expected_commit else '<none>'}",
        )
        _check(
            results,
            "manifest: dataset sha equals the tracked run spec",
            manifest.get("dataset_sha256") == spec["dataset_sha256"],
        )
        script_digest = (
            hashlib.sha256(tracked_script.read_bytes()).hexdigest()
            if tracked_script.exists()
            else None
        )
        _check(
            results,
            "manifest: worker script sha equals the TRACKED launch script",
            script_digest is not None
            and manifest.get("worker_script_sha256") == script_digest,
            f"manifest {str(manifest.get('worker_script_sha256'))[:16]}… vs "
            f"tracked {script_digest[:16] if script_digest else '<none>'}…",
        )
        ranks = manifest.get("ranks", {})
        versions_ok = all(
            isinstance(ranks.get(str(rank)), dict)
            and ranks[str(rank)].get("sglang_version") == spec["sglang_version"]
            for rank in RANK_ROLES
        )
        records = {
            ranks.get(str(rank), {}).get("record_sha256") for rank in RANK_ROLES
        }
        _check(
            results,
            f"manifest: all four ranks report sglang {spec['sglang_version']} "
            "with one package RECORD digest",
            versions_ok
            and len(records) == 1
            and None not in records
            and "" not in records,
        )

    router = ramp_dir / "router_config.txt"
    router_text = router.read_text() if router.exists() else ""
    _check(
        results,
        "router: 2 prefill + 2 decode consistent-hashing PD topology artifact",
        "--pd-disaggregation" in router_text
        and router_text.count("consistent_hashing") >= 3
        and router_text.count("--prefill ") == spec["topology"]["prefill_workers"]
        and router_text.count("--decode ") == spec["topology"]["decode_workers"],
        str(router),
    )


# ---------------------------------------------------------------------------
# Per-stage worker artifacts
# ---------------------------------------------------------------------------


def validate_worker_artifacts(results, stage_dir, stage, spec):
    log_digests = set()
    for rank, role in RANK_ROLES.items():
        args_path = stage_dir / f"server_args_rank{rank}_{role}.txt"
        log_path = stage_dir / f"startup_rank{rank}_{role}.log"
        args_text = args_path.read_text() if args_path.exists() else ""
        log_text = log_path.read_text() if log_path.exists() else ""
        resolved = "server_args=ServerArgs(" in args_text
        _check(
            results,
            f"stage{stage}: server_args rank{rank} resolved",
            resolved,
            str(args_path),
        )
        _check(
            results,
            f"stage{stage}: startup log rank{rank} nonempty and contains its "
            "resolved args line",
            len(log_text) > 0 and resolved and args_text.strip() in log_text,
            str(log_path),
        )
        if log_text:
            log_digests.add(hashlib.sha256(log_text.encode()).hexdigest())
        positive, negative = stage_markers(stage, role)
        for marker in common_markers(role, spec) + positive:
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
        f"stage{stage}: per-rank startup logs are distinct full captures",
        len(log_digests) == 4,
        f"{len(log_digests)}/4 distinct content digests",
    )


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def _snapshot_ok(snapshot):
    return (
        isinstance(snapshot, dict)
        and set(snapshot) >= set(WORKER_KEYS)
        and all(
            isinstance(snapshot[key], dict)
            and all(_counter_value_ok(snapshot[key].get(f)) for f in COUNTER_FIELDS)
            for key in WORKER_KEYS
        )
    )


def validate_counters(results, stage_dir, stage, expected_rounds):
    pre = _read_json(stage_dir / "counters_pre.json")
    post = _read_json(stage_dir / "counters_post.json")
    complete = _check(
        results,
        f"stage{stage}: counter snapshots complete (all four workers, "
        "finite non-negative numbers)",
        _snapshot_ok(pre) and _snapshot_ok(post),
        f"{stage_dir}/counters_pre.json,counters_post.json",
    )
    if not complete:
        return
    _check(
        results,
        f"stage{stage}: per-worker counters monotonic (all three fields)",
        all(
            post[key][field] >= pre[key][field]
            for key in WORKER_KEYS
            for field in COUNTER_FIELDS
        ),
    )
    _check(
        results,
        f"stage{stage}: per-worker cached-token deltas bounded by prompt-token deltas",
        all(
            post[key]["cached_tokens"] - pre[key]["cached_tokens"]
            <= post[key]["prompt_tokens"] - pre[key]["prompt_tokens"]
            for key in PREFILL_KEYS
        ),
    )
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


# ---------------------------------------------------------------------------
# Bench artifacts (structural, against the retained population)
# ---------------------------------------------------------------------------


def validate_bench(results, name, bench, session_rounds):
    """session_rounds: planned rounds per in-run session index, from the
    retained population; every per-round array must structurally agree."""
    if session_rounds is None or not isinstance(bench, dict):
        _check(results, f"{name}: bench artifact parseable with population", False)
        return
    sessions = len(session_rounds)
    total_rounds = sum(session_rounds)
    _check(
        results,
        f"{name}: all conversations completed with usage-complete metrics",
        bench.get("completed_conversations")
        == bench.get("total_conversations")
        == sessions
        and bench.get("input_metrics_complete") is True
        and bench.get("completed") == total_rounds,
        f"{bench.get('completed_conversations')}/{bench.get('total_conversations')} "
        f"conversations, {bench.get('completed')} rounds vs planned {total_rounds}",
    )
    errors = bench.get("errors")
    _check(
        results,
        f"{name}: per-round error slots retained and all empty",
        isinstance(errors, list)
        and len(errors) == total_rounds
        and all(e == "" for e in errors),
    )
    _check(
        results,
        f"{name}: per-round server/effective prompt lengths retained, "
        "positive finite",
        _positive_int_list(bench.get("server_prompt_lens"), total_rounds)
        and _positive_int_list(bench.get("effective_prompt_lens"), total_rounds),
    )
    session_indices = bench.get("session_indices")
    round_indices = bench.get("round_indices")
    sequences_ok = (
        isinstance(session_indices, list)
        and isinstance(round_indices, list)
        and len(session_indices) == len(round_indices) == total_rounds
    )
    if sequences_ok:
        per_session = {s: [] for s in range(sessions)}
        for session, round_index in zip(session_indices, round_indices):
            if session not in per_session:
                sequences_ok = False
                break
            per_session[session].append(round_index)
        sequences_ok = sequences_ok and all(
            per_session[s] == list(range(session_rounds[s])) for s in range(sessions)
        )
    _check(
        results,
        f"{name}: session/round index sequences exactly cover the planned rounds",
        sequences_ok,
    )
    conversations = bench.get("conversation_results")
    _check(
        results,
        f"{name}: conversation_results all succeed with exact planned rounds",
        isinstance(conversations, list)
        and len(conversations) == sessions
        and all(
            isinstance(c, dict)
            and c.get("success") is True
            and c.get("planned_rounds")
            == c.get("completed_rounds")
            == session_rounds[i]
            for i, c in enumerate(
                sorted(conversations, key=lambda c: c.get("session_index", -1))
            )
        ),
    )


# ---------------------------------------------------------------------------
# Affinity (derived from raw per-session records)
# ---------------------------------------------------------------------------


def validate_affinity(results, stage_dir, stage):
    affinity = _read_json(stage_dir / "affinity.json")
    records = affinity.get("sessions") if isinstance(affinity, dict) else None
    structural = (
        isinstance(records, list)
        and [r.get("session") for r in records] == list(range(SMALL_STAGE_SESSIONS))
        and all(r.get("bench_rc") == 0 for r in records)
    )
    derived_sticky = False
    derived_spread = False
    if structural:
        derived_sticky = True
        prefill_used, decode_used = set(), set()
        for record in records:
            delta = record.get("delta")
            if not isinstance(delta, dict) or not set(delta) >= set(WORKER_KEYS):
                derived_sticky = False
                break
            hit_prefill = sorted(k for k in PREFILL_KEYS if delta[k] > 0)
            hit_decode = sorted(k for k in DECODE_KEYS if delta[k] > 0)
            record_ok = (
                sorted(record.get("prefill_workers", [])) == hit_prefill
                and sorted(record.get("decode_workers", [])) == hit_decode
                and len(hit_prefill) == 1
                and len(hit_decode) == 1
                and record.get("sticky") is True
            )
            if not record_ok:
                derived_sticky = False
                break
            prefill_used.update(hit_prefill)
            decode_used.update(hit_decode)
        derived_spread = len(prefill_used) >= 2 and len(decode_used) >= 2
    _check(
        results,
        f"stage{stage}: sessions 0-7 sticky, derived from raw per-worker deltas",
        structural and derived_sticky,
        f"{stage_dir}/affinity.json",
    )
    _check(
        results,
        f"stage{stage}: pool spread derived from raw deltas; stored summary "
        "bits equal the derived verdict",
        structural
        and derived_spread
        and isinstance(affinity, dict)
        and affinity.get("all_sticky") is derived_sticky
        and affinity.get("spread_ok") is derived_spread,
    )


# ---------------------------------------------------------------------------
# Final-stage attribution (derived from raw probes + retained dataset keys)
# ---------------------------------------------------------------------------


def validate_attribution(results, stage_dir, dataset):
    attribution = _read_json(stage_dir / "attribution.json")
    records = attribution.get("keys") if isinstance(attribution, dict) else None
    meta = dataset.get("metadata", {}) if isinstance(dataset, dict) else {}
    num_sessions = meta.get("num_sessions") or 0
    expected_keys = {routing_key(meta.get("seed"), i) for i in range(num_sessions)}
    keys_ok = (
        isinstance(records, list)
        and num_sessions > 0
        and len(records) == num_sessions
        and {r.get("routing_key") for r in records} == expected_keys
    )
    _check(
        results,
        "stage5: attribution covers exactly the dataset-derived routing keys",
        keys_ok,
        f"{stage_dir}/attribution.json",
    )
    derived_all = False
    prefill_used, decode_used = set(), set()
    if keys_ok:
        derived_all = True
        for record in records:
            probes = record.get("probes")
            probes_ok = isinstance(probes, list) and len(probes) == 2
            pairs = []
            if probes_ok:
                for probe in probes:
                    hit_prefill = probe.get("prefill_workers")
                    hit_decode = probe.get("decode_workers")
                    if not (
                        probe.get("ok") is True
                        and isinstance(hit_prefill, list)
                        and isinstance(hit_decode, list)
                        and len(hit_prefill) == 1
                        and len(hit_decode) == 1
                        and hit_prefill[0] in PREFILL_KEYS
                        and hit_decode[0] in DECODE_KEYS
                    ):
                        probes_ok = False
                        break
                    pairs.append((hit_prefill[0], hit_decode[0]))
            derived = probes_ok and len(set(pairs)) == 1
            if derived:
                prefill_used.add(pairs[0][0])
                decode_used.add(pairs[0][1])
            # Stored bits must EQUAL the derived verdict — a forged True is a
            # failure even when other records are genuinely sticky.
            if not (
                derived
                and record.get("sticky") is True
                and record.get("stable") is True
            ):
                derived_all = False
    _check(
        results,
        "stage5: every key has two successful identical single-pair probes; "
        "stored bits equal derived",
        keys_ok and derived_all and attribution.get("all_sticky") is derived_all,
    )
    _check(
        results,
        "stage5: attribution spread — both pools serve >= 2 workers",
        len(prefill_used) >= 2 and len(decode_used) >= 2,
        f"prefill {sorted(prefill_used)}, decode {sorted(decode_used)}",
    )


# ---------------------------------------------------------------------------
# Wire probes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage assemblies
# ---------------------------------------------------------------------------


def validate_small_stage(results, stage_dir, stage, spec, planned_rounds):
    validate_worker_artifacts(results, stage_dir, stage, spec)
    expected = sum(planned_rounds[:SMALL_STAGE_SESSIONS]) if planned_rounds else None
    validate_counters(results, stage_dir, stage, expected)
    validate_affinity(results, stage_dir, stage)
    for session in range(SMALL_STAGE_SESSIONS):
        rounds = (
            [planned_rounds[session]]
            if planned_rounds and session < len(planned_rounds)
            else None
        )
        validate_bench(
            results,
            f"stage{stage}: bench_s{session}",
            _bench_last(stage_dir / f"bench_s{session}.jsonl"),
            rounds,
        )
    validate_wire(results, stage_dir, stage)


def validate_final_stage(results, stage_dir, spec, dataset, planned_rounds):
    validate_worker_artifacts(results, stage_dir, FINAL_STAGE, spec)
    validate_counters(
        results,
        stage_dir,
        FINAL_STAGE,
        sum(planned_rounds) if planned_rounds else None,
    )
    validate_bench(
        results,
        "stage5: bench_32s",
        _bench_last(stage_dir / "bench_32s.jsonl"),
        planned_rounds,
    )
    validate_attribution(results, stage_dir, dataset)
    validate_wire(results, stage_dir, FINAL_STAGE)


def load_spec(path=None):
    return json.loads((path or TRACKED_DIR / "ramp_run_spec.json").read_text())


def validate(
    ramp_dir,
    only_stage=None,
    expected_commit=None,
    spec=None,
    tracked_script=None,
):
    results = []
    spec = spec or load_spec()
    tracked_script = tracked_script or TRACKED_DIR / "launch_glm_pd_worker.sh"
    if expected_commit is None:
        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ramp_dir,
        ).stdout.strip()
    dataset = _dataset(ramp_dir)
    planned_rounds = _planned_rounds(dataset)
    stages = [only_stage] if only_stage else [1, 2, 3, 4, FINAL_STAGE]
    for stage in stages:
        stage_dir = ramp_dir / f"stage{stage}"
        if not _check(results, f"stage{stage}: directory present", stage_dir.is_dir()):
            continue
        if stage == FINAL_STAGE:
            validate_final_stage(results, stage_dir, spec, dataset, planned_rounds)
        else:
            validate_small_stage(results, stage_dir, stage, spec, planned_rounds)
    # Identity checks run in EVERY mode: a per-stage gate that skips dataset,
    # manifest, script, or topology identity is a fail-open gate.
    validate_run_identity(results, ramp_dir, spec, expected_commit, tracked_script)
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
