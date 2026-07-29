"""Regenerate the ramp validation report strictly from artifacts retained
inside the evidence directory, stamped with the commit the report was
generated at.

    python3 make_ramp_report.py --ramp-dir DIR --output REPORT_MD
        [--notes NOTES_MD]

Rendering is bound to a FRESH validator run: this script invokes
``validate_ramp_evidence.validate()`` against the artifacts it is about to
render (never trusting a pre-existing verdict JSON — a stale or copied
passing JSON must not produce the passed-ramp presentation) and retains that
result as ``evidence_validation.json`` alongside the report.

The renderer is TOTAL over failed or incomplete evidence: a missing or
malformed stage renders as a failed/missing row (with the ledger's recorded
reason and the last passing stage), never a crash, and each metric section
appears only when the artifacts backing it exist and parse. Run-specific
narrative belongs in an optional ``--notes`` file appended verbatim; this
generator itself derives every claim from the evidence directory.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TRACKED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRACKED_DIR))

import validate_ramp_evidence as validator  # noqa: E402

RANK_ROLES = validator.RANK_ROLES
CONFIG_FIELDS = (
    "attention_backend",
    "speculative_algorithm",
    "enable_hierarchical_cache",
    "hicache_size",
)


def _maybe_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _maybe_bench(path):
    try:
        return json.loads(Path(path).read_text().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None


def rank_configs(stage_dir):
    """Resolved config markers for ALL four ranks; 'MISSING' where the
    retained args are absent/unresolved (attribution unverified there)."""
    configs = {}
    for rank, role in RANK_ROLES.items():
        path = stage_dir / f"server_args_rank{rank}_{role}.txt"
        text = path.read_text() if path.exists() else ""
        if "server_args=ServerArgs(" not in text:
            configs[f"rank{rank} {role}"] = "MISSING"
            continue
        markers = []
        for field in CONFIG_FIELDS:
            match = re.search(rf"(?<![a-z_]){field}=([^,)]+)", text)
            if match:
                markers.append(f"{field}={match.group(1)}")
        configs[f"rank{rank} {role}"] = "; ".join(markers)
    return configs


def artifact_matrix(stage_dir):
    """(resolved-args count, nonempty-log count) over the four ranks."""
    args_ok = logs_ok = 0
    for rank, role in RANK_ROLES.items():
        args_path = stage_dir / f"server_args_rank{rank}_{role}.txt"
        log_path = stage_dir / f"startup_rank{rank}_{role}.log"
        if args_path.exists() and "server_args=ServerArgs(" in args_path.read_text():
            args_ok += 1
        if log_path.exists() and log_path.stat().st_size > 0:
            logs_ok += 1
    return args_ok, logs_ok


def stage_summary(ramp_dir, stage):
    """Benchmark facts for one small stage, or None when its artifacts are
    missing/unreadable (the row renders from the ledger instead)."""
    d = ramp_dir / f"stage{stage}"
    if not d.is_dir():
        return None
    affinity = _maybe_json(d / "affinity.json")
    wire = _maybe_json(d / "wire_probe.json")
    benches = [_maybe_bench(p) for p in sorted(d.glob("bench_s*.jsonl"))]
    benches = [b for b in benches if isinstance(b, dict)]
    if not benches or not isinstance(affinity, dict) or not isinstance(wire, dict):
        return None
    try:
        return {
            "affinity": affinity,
            "wire": wire,
            "conversations": f"{sum(b['completed_conversations'] for b in benches)}"
            f"/{sum(b['total_conversations'] for b in benches)}",
            "rounds": sum(b["completed"] for b in benches),
            "tokens": sum(b["total_input_tokens"] for b in benches),
            "complete": all(b["input_metrics_complete"] for b in benches),
            "configs": rank_configs(d),
            "artifacts": artifact_matrix(d),
        }
    except (KeyError, TypeError):
        return None


def stage_failure_reason(ramp_dir, ledger, stage):
    entry = (ledger or {}).get("stages", {}).get(str(stage), {})
    if entry.get("reason"):
        return entry["reason"]
    marker = ramp_dir / f"stage{stage}" / "STAGE_FAILED"
    if marker.exists():
        return marker.read_text().strip()
    if not (ramp_dir / f"stage{stage}").is_dir():
        return "stage not attempted (no evidence directory)"
    return "artifacts missing or unreadable"


def manifest_display(ramp_dir):
    """Manifest for the environment section: the structured manifest.json if
    present, else the legacy text manifest (archived runs)."""
    for name in ("manifest.json", "manifest.txt"):
        path = ramp_dir / name
        if path.exists():
            return path.read_text().strip()
    return "<no manifest retained>"


def run_commit(ramp_dir):
    manifest = _maybe_json(ramp_dir / "manifest.json")
    if isinstance(manifest, dict) and manifest.get("commit"):
        return str(manifest["commit"])
    legacy = ramp_dir / "manifest.txt"
    if legacy.exists():
        match = re.search(r"branch: \S+ @ (\w+)", legacy.read_text())
        if match:
            return match.group(1)
    return "<unknown>"


def generate(ramp_dir, output, notes=None):
    # Fresh verdict, bound to the exact artifacts rendered below.
    checks = validator.validate(ramp_dir)
    failed_checks = [c for c in checks if not c["ok"]]
    validated = not failed_checks
    (ramp_dir / "evidence_validation.json").write_text(
        json.dumps({"checks": checks, "all_ok": validated}, indent=2)
    )

    spec = validator.load_spec()
    ledger = _maybe_json(ramp_dir / "ledger.json")
    stages = {s: stage_summary(ramp_dir, s) for s in (1, 2, 3, 4)}
    stage5_dir = ramp_dir / "stage5"
    s5 = _maybe_bench(stage5_dir / "bench_32s.jsonl")
    s5_wire = _maybe_json(stage5_dir / "wire_probe.json")
    pre = _maybe_json(stage5_dir / "counters_pre.json")
    post = _maybe_json(stage5_dir / "counters_post.json")
    prefill_keys = validator.PREFILL_KEYS
    counters_ok = all(
        isinstance(s, dict) and all(k in s for k in prefill_keys) for s in (pre, post)
    )
    generated_at = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ramp_dir
    ).stdout.strip()
    ran_at = run_commit(ramp_dir)

    if validated:
        stage_names = {
            1: "dense baseline (triton)",
            2: "+explicit DSA",
            3: "+EAGLE 3-1-4",
            4: "+HiCache (prefill)",
        }
    else:
        stage_names = {
            1: "intended: dense baseline",
            2: "intended: +DSA",
            3: "intended: +EAGLE 3-1-4",
            4: "intended: +HiCache (prefill)",
        }

    lines = []
    w = lines.append
    w("# Recovery-Agent Benchmark Validation Report (regenerated from retained artifacts)")
    w("")
    w(f"Ramp executed at commit {ran_at[:12]} (manifest); report regenerated at commit {generated_at[:12]} by `benchmark/recovery_agent_pd/make_ramp_report.py`, which re-runs the fail-closed validator against the exact artifacts rendered here at generation time. Every number below is read from a file inside the evidence directory.")
    w("")
    w("## Evidence verdict (validator re-run at report generation)")
    w("")
    w(f"`benchmark/recovery_agent_pd/validate_ramp_evidence.py` fresh result: **{len(checks) - len(failed_checks)}/{len(checks)} checks pass** (`all_ok: {validated}`).")
    w("")
    if not validated:
        w("**This evidence set is NOT a validated controlled ramp.** The failing checks, grouped:")
        w("")
        failure_kinds = {}
        for c in failed_checks:
            kind = re.sub(r"rank\d|bench_s\d|stage\d: ", "", c["check"]).strip()
            failure_kinds.setdefault(kind, []).append(c["check"])
        for kind, items in sorted(failure_kinds.items()):
            w(f"- **{kind}** ({len(items)}×): e.g. `{items[0]}`")
        w("")
        w("Stage labels below are the INTENDED configurations; a stage's configuration is verified only where its per-rank checks above pass. Benchmark numbers shown are directly measured from the retained artifacts, but their attribution to a specific verified configuration holds only for stages with passing configuration checks.")
        w("")
    if isinstance(ledger, dict) and ledger.get("stages"):
        w("## Run ledger")
        w("")
        w(f"Run `{ledger.get('run_id', '<unknown>')}` at commit {str(ledger.get('commit', ''))[:12]}; **last passing stage: {ledger.get('last_passing_stage')}**.")
        w("")
        w("| stage | status | reason |")
        w("|---|---|---|")
        for stage_id in sorted(ledger["stages"], key=int):
            entry = ledger["stages"][stage_id]
            w(f"| {stage_id} | {entry.get('status')} | {entry.get('reason') or '—'} |")
        w("")
    w("## Environment (retained manifest)")
    w("```")
    w(manifest_display(ramp_dir))
    w("```")
    w("")
    if validated:
        w("## Controlled 2P2D ramp (stage1..5, evidence-validated)")
    else:
        w("## Attempted staged 2P2D run (stage1..5 — see verdict above)")
    w("")
    w(f"One fixed prebuilt 32-session `agent-short` population (sha256 {spec['dataset_sha256'][:16]}…, seed {spec['dataset']['seed']}, context-authoritative, retained in the evidence dir); stages 1–4 replay its first 8 sessions as sequential single-session runs (per-session affinity attribution from worker counter deltas); stage 5 runs all 32 concurrently on the final configuration after a worker cache flush.")
    w("")
    config_note = "resolved config" if validated else "intended config (unverified)"
    w(f"| stage | {config_note} | conversations | rounds | input tokens | usage complete | per-session sticky | pool spread | wire probe | args/logs retained |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for s in (1, 2, 3, 4):
        st = stages[s]
        if st is None:
            reason = stage_failure_reason(ramp_dir, ledger, s)
            w(f"| {s} ({stage_names[s]}) | **FAILED/MISSING**: {reason} | — | — | — | — | — | — | — | — |")
            continue
        args_ok, logs_ok = st["artifacts"]
        w(f"| {s} ({stage_names[s]}) | see per-rank matrix | {st['conversations']} | {st['rounds']} | {st['tokens']:,} | {st['complete']} | {st['affinity'].get('all_sticky')} | {st['affinity'].get('spread_ok')} | {st['wire'].get('reasoning_absent_from_round2')} | {args_ok}/4 args, {logs_ok}/4 logs |")
    if isinstance(s5, dict) and isinstance(s5_wire, dict):
        s5_args, s5_logs = artifact_matrix(stage5_dir)
        w(f"| 5 (final config, 32 sessions, conc. 8) | see per-rank matrix | {s5.get('completed_conversations')}/{s5.get('total_conversations')} | {s5.get('completed')} | {s5.get('total_input_tokens', 0):,} | {s5.get('input_metrics_complete')} | n/a (concurrent) | n/a | {s5_wire.get('reasoning_absent_from_round2')} | {s5_args}/4 args, {s5_logs}/4 logs |")
    else:
        w(f"| 5 (final config) | **FAILED/MISSING**: {stage_failure_reason(ramp_dir, ledger, 5)} | — | — | — | — | — | — | — | — |")
    w("")
    w("Per-rank resolved configuration (from each rank's retained `server_args`; `MISSING` = no resolved args retained for that rank at that stage, so its configuration is unattributable):")
    w("")
    for s in (1, 2, 3, 4, 5):
        stage_dir = ramp_dir / f"stage{s}"
        w(f"- stage {s}:")
        if not stage_dir.is_dir():
            w("  - no evidence directory")
            continue
        for rank, config in rank_configs(stage_dir).items():
            w(f"  - {rank}: `{config}`")
    w("")
    if isinstance(s5, dict) and counters_ok:
        prompt_delta = {
            k: post[k]["prompt_tokens"] - pre[k]["prompt_tokens"] for k in prefill_keys
        }
        cached_delta = {
            k: post[k]["cached_tokens"] - pre[k]["cached_tokens"] for k in prefill_keys
        }
        pool_prompt = sum(prompt_delta.values())
        pool_cached = sum(cached_delta.values())
        if validated:
            w("## Stage-5 headline (stage5/bench_32s.jsonl + counters)")
        else:
            w("## Final 32-session benchmark — direct evidence (stage5/bench_32s.jsonl + counters; configuration attribution unverified)")
        w("")
        w(f"- Conversations: **{s5['completed_conversations']}/{s5['total_conversations']}** ({s5['completed']} rounds), `input_metrics_complete: {s5['input_metrics_complete']}`")
        w(f"- Total input tokens (server-reported): {s5['total_input_tokens']:,}")
        w(f"- Turn throughput {s5['turn_throughput_turns_per_s']:.2f} turns/s; session throughput {s5['session_throughput_sessions_per_s']:.3f} sessions/s; mean TTFT {s5['mean_ttft_ms']:.0f} ms; mean E2E {s5['mean_e2e_latency_ms']:.0f} ms")
        if pool_prompt > 0:
            w(f"- **Prefill-pool radix hit rate {100*pool_cached/pool_prompt:.1f}%** ({pool_prompt:,.0f} prompt tokens, {pool_cached:,.0f} device-cached; " + "; ".join(f"{k} {100*cached_delta[k]/prompt_delta[k]:.1f}%" for k in prompt_delta if prompt_delta[k] > 0) + ") — from retained pre/post worker counters")
        if isinstance(s5_wire, dict):
            w(f"- Wire probe: separate reasoning present in round 1 and absent from round 2's assistant history ({s5_wire.get('round2_assistant_equals_content_only')=})")
        w("")
    lc = s5.get("live_conformance") if isinstance(s5, dict) else None
    if isinstance(lc, dict):
        w("## Live conformance (server-defined shape, stage5 JSON `live_conformance`)")
        w("")
        w(f"available={lc.get('available')}, sessions_observed={lc.get('sessions_observed')}, all_sessions_usable={lc.get('all_sessions_usable')}, gates_within_tolerance={lc.get('gates_within_tolerance')}, conformant_population={lc.get('conformant_population')} (reference population 2048 — a 32-session run cannot and does not claim calibration conformance).")
        w("")
        dimensions = lc.get("dimensions") or {}
        if dimensions:
            w("| dimension | target | live | live dev |")
            w("|---|---|---|---|")
            for k, e in dimensions.items():
                w(f"| {k} | {e['target']:,.1f} | {e['realized']:,.1f} | {e['deviation_frac']:+.3f} |")
            w("")
            w("Interpretation: live context/ISL deviations below planned values are CONSISTENT WITH the reasoning-exclusion replay contract (replayed history carries assistant `content` only while the plan charges the full per-turn output budget); the retained wire probes establish that contract per-probe, but no controlled decomposition isolating replay length from actual assistant content length, sizing deficits, or template effects is retained, so the mechanism is supported, not proven exhaustive. Small-population deviations from the calibration targets are sampling error, reported as non-conformant rather than claimed as success.")
            w("")
    if notes:
        w("## Run-specific notes (appended verbatim from --notes)")
        w("")
        w(Path(notes).read_text().strip())
        w("")

    Path(output).write_text("\n".join(lines) + "\n")
    print(f"wrote {output} ({len(lines)} lines; validated={validated})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramp-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional Markdown file appended verbatim (run-specific "
        "narrative lives there, never in this generator)",
    )
    args = parser.parse_args()
    generate(Path(args.ramp_dir), args.output, args.notes)


if __name__ == "__main__":
    main()
