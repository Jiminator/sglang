#!/usr/bin/env python3
"""Report helpers for the OpenHands agentic evalscope sweeps.

Subcommands:
  latest              -> print latest sweep timestamp dir name
  metrics [--ts T]    -> print TTFT / Accept-Len / KV-hit tables
  logs [ts]           -> print server log directory + files

A timestamp argument may be either a bare timestamp (e.g. ``20260515_031121``)
or a full path to the sweep directory. If omitted, ``latest`` is used.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
from pathlib import Path

# Sweeps land in directories whose name is a bare YYYYMMDD_HHMMSS timestamp.
TS_RE = re.compile(r"^\d{8}_\d{6}$")

SWEEP_ROOT = Path(__file__).resolve().parent / "outputs"

CONFIGS = [
    "attn_tp4_moe_tp4",
    "attn_tp4_moe_ep4",
    "attn_tp8_moe_tp8",
    "attn_tp8_moe_ep8",
]
CONC_ORDER = [16, 8, 4, 2, 1]


def latest_sweep() -> Path:
    if not SWEEP_ROOT.is_dir():
        sys.exit(f"no agentic sweeps under {SWEEP_ROOT}")
    candidates = sorted(
        p for p in SWEEP_ROOT.iterdir() if p.is_dir() and TS_RE.match(p.name)
    )
    if not candidates:
        sys.exit(f"no agentic sweeps under {SWEEP_ROOT}")
    return candidates[-1]


def resolve_sweep(ts: str | None) -> Path:
    if not ts:
        return latest_sweep()
    p = Path(ts)
    if p.is_absolute() and p.is_dir():
        return p
    return SWEEP_ROOT / ts


_LOG_TS_RE = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def server_accept_lengths(
    sweep: Path, cfg: str, step_mtimes: list[tuple[float, int]]
) -> dict[int, float]:
    """True per-step acceptance length from SGLang's own spec-decode counters.

    evalscope's ``Decoded Tok/Iter`` is computed client-side as
    ``(completion_tokens - 1) / (n_chunks - 1)`` where ``n_chunks`` is the number
    of streamed SSE chunks. Under concurrency those chunks coalesce (several
    decode iterations flushed into one chunk), so ``n_chunks`` is undercounted and
    the ratio drifts from the true value (and can overshoot the physical ceiling
    of ``num_spec + 1``, e.g. 6.006 at conc 8 for a 5-token MTP head). The server's
    own per-interval counters are exact, so we use those when a server log exists:
    SGLang logs ``Decode batch, #running-req: R, ..., accept len: L`` every
    ``decode_log_interval`` steps (``L = 1 + num_spec * accept_rate``, already
    bounded by ``num_spec + 1``); the per-step value is the ``#running-req``-
    weighted mean of ``L`` (running-req approximates the per-interval drafted
    count, so this is an accepted/drafted-weighted estimator).

    Each interval is attributed to the sweep step whose ``(prev_end, end]`` window
    it falls in, where a step's end is the mtime of its ``benchmark_summary.json``
    (written when evalscope finishes the step). The server log prints wall-clock
    and filesystem mtimes are epochs in the same tz on the bench box, so the two
    clocks line up. Returns ``{}`` (callers fall back to the evalscope value) when
    no server log is present — e.g. a sweep pulled as a mirror without logs.
    """
    log = sweep / "server_logs" / f"{cfg}.log"
    # The sweep dir is normally timestamp-named (YYYYMMDD_HHMMSS). A mirror of a
    # remote sweep can instead be pointed at its inner ``results/`` dir, whose
    # parent carries the timestamp — fall back to the parent for the year so the
    # server-log intervals can still be attributed to steps.
    stamp = sweep.name if TS_RE.match(sweep.name) else sweep.parent.name
    if not log.is_file() or not TS_RE.match(stamp):
        return {}
    year = int(stamp[:4])
    num_spec = 5
    wins: list[tuple[int, float, int]] = []  # (epoch, accept_len, weight)
    for line in log.read_text(errors="ignore").splitlines():
        if "speculative_num_steps=" in line:  # server_args line
            m = re.search(r"speculative_num_steps=(\d+)", line)
            if m:
                num_spec = int(m.group(1))
        if "Decode batch" in line and "accept len:" in line:  # per-interval
            ts = _LOG_TS_RE.search(line)
            al = re.search(r"accept len:\s*([\d.]+)", line)
            rr = re.search(r"#running-req:\s*(\d+)", line)
            if ts and al:
                mo, da, h, mi, s = (int(x) for x in ts.groups())
                ep = calendar.timegm((year, mo, da, h, mi, s, 0, 0, 0))
                w = int(rr.group(1)) if rr else 1
                wins.append((ep, float(al.group(1)), max(w, 1)))
    if not wins:
        return {}
    # A step's final interval can be flushed a moment after evalscope writes its
    # summary, so each step's upper bound is its mtime + a small slack; the *next*
    # step starts from that same shifted bound (not the bare mtime), so the windows
    # tile the timeline without overlap — else a boundary interval double-counts.
    slack = 2
    ceiling = num_spec + 1
    out: dict[int, float] = {}
    prev = 0.0
    for mtime, conc in sorted(step_mtimes):
        lo, hi = prev, mtime + slack
        prev = hi
        seg = [(al, w) for ep, al, w in wins if lo < ep <= hi]
        wsum = sum(w for _, w in seg)
        if not wsum:
            continue
        val = sum(al * w for al, w in seg) / wsum
        # accepted <= drafted always holds, so a >ceiling value means a parse error
        # (counts swapped / format drift). Drop it; the caller falls back + warns.
        if val > ceiling + 1e-6:
            print(
                f"WARNING: {cfg} conc {conc}: server accept length {val:.3f} exceeds "
                f"ceiling {ceiling} — likely a server-log parse error; dropping.",
                file=sys.stderr,
            )
            continue
        out[conc] = val
    return out


def load_metrics(sweep: Path) -> dict[str, dict[int, dict]]:
    """For each (config, concurrency), merge benchmark_summary.json with the
    p50 row of benchmark_percentile.json (re-exposed as ``Median <field>`` keys).

    ``Accept Length`` is the server-true spec-decode value when a server log is
    available, else evalscope's client-side ``Decoded Tok/Iter`` (kept verbatim
    under that key for reference)."""
    out: dict[str, dict[int, dict]] = {}
    for cfg in CONFIGS:
        steps: list[tuple[float, dict]] = []
        for d in (sweep / cfg).glob("parallel_*"):
            summary_path = d / "benchmark_summary.json"
            # evalscope creates the parallel_* dir before writing its summary,
            # so a still-running sweep has dirs without a summary yet — skip them
            # rather than crash (lets `metrics` work mid-run too).
            if not summary_path.is_file():
                continue
            s = json.loads(summary_path.read_text())
            pct_path = d / "benchmark_percentile.json"
            if pct_path.exists():
                for row in json.loads(pct_path.read_text()):
                    if row.get("Percentiles") == "50%":
                        for k, v in row.items():
                            if k != "Percentiles":
                                s[f"Median {k}"] = v
                        break
            steps.append((summary_path.stat().st_mtime, s))
        if not steps:
            continue
        accept = server_accept_lengths(
            sweep, cfg, [(mt, int(s["Concurrency"])) for mt, s in steps]
        )
        for _, s in steps:
            conc = int(s["Concurrency"])
            s["Accept Length"] = accept.get(conc, s.get("Decoded Tok/Iter"))
            s["Accept Length Source"] = "server" if conc in accept else "evalscope"
            out.setdefault(cfg, {})[conc] = s
    return out


def server_startup_times(sweep: Path) -> dict[str, float]:
    """Per-config server startup seconds (process launch -> first healthy /health),
    read from the ``server_logs/<config>.startup`` sidecar the sweep writes.
    One value per config (a server is launched once per config, not per
    concurrency). Missing sidecar (older sweeps / a mirror pulled without it) ->
    the config is simply absent, rendered ``n/a``."""
    out: dict[str, float] = {}
    log_dir = sweep / "server_logs"
    for cfg in CONFIGS:
        f = log_dir / f"{cfg}.startup"
        if f.is_file():
            try:
                out[cfg] = float(f.read_text().strip())
            except ValueError:
                pass
    return out


def print_metric_table(
    label: str,
    key: str,
    fmt: str,
    data: dict[str, dict[int, dict]],
    configs: list[str],
) -> None:
    print(f"{label} — conc {', '.join(str(c) for c in CONC_ORDER)}")
    for cfg in configs:
        cells = data.get(cfg, {})
        row = ", ".join(
            fmt % cells[c][key] if c in cells else "n/a" for c in CONC_ORDER
        )
        print(f"- {cfg}: {row}")
    print()


def cmd_latest(args: argparse.Namespace) -> None:
    print(latest_sweep().name)


def cmd_metrics(args: argparse.Namespace) -> None:
    sweep_dir = resolve_sweep(args.ts)
    data = load_metrics(sweep_dir)
    # Warn when a server log IS present but the accept length still fell back
    # to evalscope's chunk value (missing/renamed log or a spec-decode format
    # drift) — that silently reverts to the value that can exceed the ceiling.
    # A sweep pulled as a mirror with no log falls back silently (expected).
    fell_back = sorted(
        f"{cfg}@conc{c}"
        for cfg, concs in data.items()
        for c, s in concs.items()
        if s.get("Accept Length Source") != "server"
        and (sweep_dir / "server_logs" / f"{cfg}.log").is_file()
    )
    if fell_back:
        print(
            "WARNING: accept length fell back to evalscope's chunk-based value "
            "despite a server log being present for: "
            f"{', '.join(fell_back)}. These can exceed the physical ceiling — "
            "check the run's server_logs/<config>.log.",
            file=sys.stderr,
        )
    # Only print configs that actually have data (a tp4-only run stays clean).
    present = [c for c in CONFIGS if c in data]
    configs = present or CONFIGS

    # TTFT uses the p50 (cold start taxes the first request of each step); TPOT /
    # ITL / output throughput are steady-state decode metrics, so the per-step
    # mean from benchmark_summary.json is representative and is used directly.
    for label, key, fmt in [
        ("TTFT (Median ms)", "Median TTFT (ms)", "%.1f"),
        ("TPOT (mean ms)", "TPOT (ms)", "%.2f"),
        ("ITL (mean ms)", "ITL (ms)", "%.2f"),
        ("Output Throughput (tok/s)", "Output Throughput (tok/s)", "%.1f"),
        ("Accept Length (spec-decode tok/iter)", "Accept Length", "%.3f"),
        ("KV Cache Hit Rate (%)", "KV Cache Hit Rate (%)", "%.2f"),
    ]:
        print_metric_table(label, key, fmt, data, configs)

    # Server startup (process launch -> ready): one value per config. Emit only
    # when the sweep recorded it, so sweeps without the .startup sidecar produce
    # no (empty) block.
    startup = server_startup_times(sweep_dir)
    if startup:
        print("Server Startup (s) — launch to ready")
        for cfg in configs:
            v = startup.get(cfg)
            print(f"- {cfg}: {'%.0f' % v if v is not None else 'n/a'}")
        print()


def cmd_logs(args: argparse.Namespace) -> None:
    log_dir = resolve_sweep(args.ts) / "server_logs"
    print(log_dir)
    if log_dir.is_dir():
        for f in sorted(log_dir.glob("*.log")):
            print(f"  {f.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("latest", help="print latest sweep timestamp")
    sp.set_defaults(func=cmd_latest)

    sp = sub.add_parser("metrics", help="print TTFT / Accept-Len / KV-hit tables")
    sp.add_argument("--ts", default=None, help="sweep ts or path (default: latest)")
    sp.set_defaults(func=cmd_metrics)

    sp = sub.add_parser("logs", help="print server log directory + files")
    sp.add_argument("ts", nargs="?", default=None)
    sp.set_defaults(func=cmd_logs)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
