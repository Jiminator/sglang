"""Controlled GLM 2P2D validation ramp driver with fail-closed evidence.

Runs the staged validation ramp (dense triton baseline -> +DSA -> +EAGLE
3-1-4 -> +HiCache prefill -> 32-session concurrent final run) against a
2 prefill + 2 decode fleet, retaining audit-grade evidence per stage and
gating every stage — and the whole run — on ``validate_ramp_evidence.py``.

    python3 ramp_driver.py --ramp-dir RUNS_DIR --fleet-config FLEET_JSON \
        [--start-stage N] [--skip-relaunch]

The fleet config (environment-specific, NOT tracked; see
``fleet_config.example.json``) supplies rank->role/IP mapping, the remote
execution command, the router URL, and scratch paths. The tracked
``ramp_run_spec.json`` supplies the run identity (dataset sha + semantics,
model, topology, sglang version).

Fail-closed preflight runs before ANY kill/launch/benchmark on BOTH full and
resume paths (including a direct stage-5 resume): retained dataset sha +
semantics vs the spec, per-rank worker-script digest vs the TRACKED launch
script, per-rank sglang version + package RECORD digest vs the spec, and (on
resume) an existing ``manifest.json`` stamped with the current HEAD. A full
run regenerates ``manifest.json`` fresh — old manifests are never mutated
into apparent freshness. Stage-5 resume additionally re-runs the health,
router, and real-traffic gates before touching evidence.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TRACKED_DIR = Path(__file__).resolve().parent
REPO = TRACKED_DIR.parents[1]
VALIDATOR = TRACKED_DIR / "validate_ramp_evidence.py"
TRACKED_WORKER_SCRIPT = TRACKED_DIR / "launch_glm_pd_worker.sh"
WIRE_PROBE = TRACKED_DIR / "wire_probe.py"
DATASET_NAME = "recovery_agent_fixed_32.json"

_PKG_IDENTITY_SNIPPET = (
    "import importlib.metadata as md, hashlib; "
    "d = md.distribution('sglang'); "
    "print(d.version); "
    "print(hashlib.sha256(d.read_text('RECORD').encode()).hexdigest())"
)


class Fleet:
    """Environment-specific topology + remote execution, from the fleet
    config JSON. Rank 0 is local; other ranks run via the configured remote
    command with the rank appended."""

    def __init__(self, config, ramp_dir):
        self.ramp_dir = ramp_dir
        self.ranks = {int(r): (v["role"], v["ip"]) for r, v in config["ranks"].items()}
        self.router_url = config["router_url"]
        self.remote_cmd = config["remote_cmd"]
        self.wrapper = config["wrapper"]
        self.source_dataset = config["source_dataset"]
        self.worker_script = config["worker_script"]
        self.router_python = config["router_python"]
        self.worker_port = config.get("worker_port", 30001)
        self.bootstrap_port = config.get("bootstrap_port", 8998)
        self.model_path = config["model_path"]
        self.prefill = {
            f"P{r}": f"http://{ip}:{self.worker_port}"
            for r, (role, ip) in self.ranks.items()
            if role == "prefill"
        }
        self.decode = {
            f"D{r}": f"http://{ip}:{self.worker_port}"
            for r, (role, ip) in self.ranks.items()
            if role == "decode"
        }

    def sh(self, cmd, timeout=180, check=True):
        return subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def on_rank(self, rank, script, timeout=180, check=True):
        if rank == 0:
            return self.sh(["bash", "-c", self.wrapper + script], timeout, check)
        return self.sh(
            self.remote_cmd + [str(rank), "--", "bash", "-c", self.wrapper + script],
            timeout,
            check,
        )


def load_spec():
    return json.loads((TRACKED_DIR / "ramp_run_spec.json").read_text())


def head_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()


def dataset_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_package_identity(fleet, rank):
    result = fleet.on_rank(
        rank, f"python3 -c \"{_PKG_IDENTITY_SNIPPET}\"", timeout=120, check=False
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(lines) < 2:
        return None
    return {"sglang_version": lines[-2], "record_sha256": lines[-1]}


def rank_script_digest(fleet, rank):
    result = fleet.on_rank(
        rank, f"sha256sum {fleet.worker_script}", timeout=60, check=False
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout.split()[0]


EXPECTED_RANK_ROLES = {0: "prefill", 1: "prefill", 2: "decode", 3: "decode"}


def preflight(fleet, spec, start_stage):
    """Verify every identity the evidence gate will later require, BEFORE any
    fleet mutation; raise on any mismatch. All local (side-effect-free)
    checks run first, then the remote identity checks; only after everything
    passes does a full run write its fresh manifest.json. A resume requires
    the existing manifest to match the live identities AND every prior
    stage's evidence to validate."""
    full_run = start_stage == 1
    # --- local checks, no side effects -----------------------------------
    if fleet.model_path != spec["model_path"]:
        raise RuntimeError(
            f"fleet model {fleet.model_path} != spec {spec['model_path']}"
        )
    if {r: role for r, (role, _ip) in fleet.ranks.items()} != EXPECTED_RANK_ROLES:
        raise RuntimeError(
            f"fleet ranks/roles {fleet.ranks} != expected {EXPECTED_RANK_ROLES}"
        )
    ips = [ip for _role, ip in fleet.ranks.values()]
    if len(set(ips)) != len(ips):
        raise RuntimeError(f"fleet worker IPs are not unique: {ips}")
    topology = spec["topology"]
    if (
        len(fleet.prefill) != topology["prefill_workers"]
        or len(fleet.decode) != topology["decode_workers"]
    ):
        raise RuntimeError(
            f"fleet pools {len(fleet.prefill)}P/{len(fleet.decode)}D != spec "
            f"{topology['prefill_workers']}P/{topology['decode_workers']}D"
        )
    ledger_path = fleet.ramp_dir / "ledger.json"
    if full_run:
        stale = [
            p.name
            for p in [ledger_path, *(fleet.ramp_dir / f"stage{s}" for s in range(1, 6))]
            if p.exists()
        ]
        if stale:
            raise RuntimeError(
                f"full run refused: evidence dir is not fresh ({stale}); use a "
                "new --ramp-dir or resume with --start-stage"
            )
    elif not ledger_path.exists():
        raise RuntimeError("resume refused: no ledger.json in the evidence dir")
    commit = head_commit()
    retained = fleet.ramp_dir / DATASET_NAME
    if not retained.exists():
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(Path(fleet.source_dataset).read_bytes())
    digest = dataset_digest(retained)
    if digest != spec["dataset_sha256"]:
        raise RuntimeError(
            f"retained dataset sha {digest[:16]}… != spec "
            f"{spec['dataset_sha256'][:16]}…; refusing all fleet action"
        )
    # --- remote identity checks ------------------------------------------
    tracked_digest = dataset_digest(TRACKED_WORKER_SCRIPT)
    ranks_identity = {}
    for rank in fleet.ranks:
        remote_digest = rank_script_digest(fleet, rank)
        if remote_digest != tracked_digest:
            raise RuntimeError(
                f"rank {rank} worker script {fleet.worker_script} digest "
                f"{str(remote_digest)[:16]}… != tracked script "
                f"{tracked_digest[:16]}…; sync it before running"
            )
        identity = rank_package_identity(fleet, rank)
        if identity is None or identity["sglang_version"] != spec["sglang_version"]:
            raise RuntimeError(
                f"rank {rank} sglang identity {identity} != spec "
                f"{spec['sglang_version']}; refusing all fleet action"
            )
        ranks_identity[str(rank)] = identity
    if len({v["record_sha256"] for v in ranks_identity.values()}) != 1:
        raise RuntimeError(
            f"package RECORD digests differ across ranks: {ranks_identity}"
        )
    manifest_path = fleet.ramp_dir / "manifest.json"
    manifest = {
        "commit": commit,
        "dataset_sha256": digest,
        "worker_script_sha256": tracked_digest,
        "ranks": ranks_identity,
    }
    if full_run:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        existing = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        # The ENTIRE manifest must match the live identities: a resume that
        # only checks the commit can silently mix packages or scripts.
        if existing != manifest:
            raise RuntimeError(
                f"resume refused: manifest {existing} does not match the live "
                f"identities {manifest}; start a full run in a fresh dir"
            )
        # Every prior stage's evidence must already validate — resuming on
        # top of an invalid stage would launder it into a completed ramp.
        for stage in range(1, start_stage):
            if not (fleet.ramp_dir / f"stage{stage}").is_dir():
                raise RuntimeError(f"resume refused: stage{stage} evidence missing")
            if not validate_stage(fleet.ramp_dir, stage):
                raise RuntimeError(
                    f"resume refused: stage{stage} evidence fails validation"
                )


def http_ok(url, timeout=5):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def wait_healthy(fleet, deadline_s=5400):
    start = time.time()
    urls = list(fleet.prefill.values()) + list(fleet.decode.values())
    while time.time() - start < deadline_s:
        if all(
            http_ok(f"{u}/health_generate") and http_ok(f"{u}/metrics") for u in urls
        ):
            return True
        time.sleep(20)
    return False


def counters(fleet):
    out = {}
    for name, url in {**fleet.prefill, **fleet.decode}.items():
        text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
        chat = prompt = cached = 0.0
        for line in text.splitlines():
            if line.startswith(
                'sglang:http_requests_total{endpoint="/v1/chat/completions"'
            ):
                chat = float(line.rsplit(" ", 1)[1])
            if line.startswith("sglang:prompt_tokens_total"):
                prompt = float(line.rsplit(" ", 1)[1])
            if line.startswith("sglang:cached_tokens_total") and "device" in line:
                cached = float(line.rsplit(" ", 1)[1])
        out[name] = {
            "chat_requests": chat,
            "prompt_tokens": prompt,
            "cached_tokens": cached,
        }
    return out


def request_delta(before, after):
    """Per-worker chat-request counter movement between two snapshots."""
    return {
        name: after[name]["chat_requests"] - before[name]["chat_requests"]
        for name in after
    }


def pool_hits(fleet, delta):
    """(prefill, decode) workers whose chat-request counter advanced."""
    prefill = [name for name in fleet.prefill if delta[name] > 0]
    decode = [name for name in fleet.decode if delta[name] > 0]
    return prefill, decode


def worker_launch_stage(role, stage):
    """Stage configuration a worker actually runs at: HiCache (stage 4)
    changes prefill only, so decode workers stay at their stage-3 config."""
    if role == "prefill" or stage < 4:
        return stage
    return 3


def kill_workers(fleet, ranks):
    # Kill the launch_server parents (HTTP frontends) too: compute-app-only
    # kills leave stale servers answering health checks with dead schedulers.
    script = (
        "for p in $(pgrep -f 'sglang.launch_server'); do "
        "case $(readlink /proc/$p/exe 2>/dev/null) in *python*) kill -9 $p;; esac; done; "
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 3"
    )
    for rank in ranks:
        fleet.on_rank(rank, script, check=False)


def launch_worker(fleet, rank, role, stage):
    log = f"/scratch/pd_logs/{role}_stage{stage}.log"
    script = (
        f"mkdir -p /scratch/pd_logs; "
        f"setsid bash -c 'ROLE={role} STAGE={stage} MODEL_PATH={fleet.model_path} "
        f"bash {fleet.worker_script}' </dev/null > {log} 2>&1 &"
    )
    if rank == 0:
        subprocess.Popen(["bash", "-c", fleet.wrapper + script])
    else:
        fleet.on_rank(rank, script, timeout=120)


def collect_stage_evidence(fleet, stage_dir, launch_stage_by_rank):
    """Retain startup log + resolved server_args for ALL four workers, each
    read from the log of the stage that worker was actually launched at.
    Collection is fail-open here; validate_stage() below is the gate."""
    for rank, (role, _ip) in fleet.ranks.items():
        log = f"/scratch/pd_logs/{role}_stage{launch_stage_by_rank[rank]}.log"
        result = fleet.on_rank(rank, f"cat {log}", timeout=120, check=False)
        (stage_dir / f"startup_rank{rank}_{role}.log").write_text(
            result.stdout[-400_000:]
        )
        args_line = [l for l in result.stdout.splitlines() if "server_args=" in l]
        (stage_dir / f"server_args_rank{rank}_{role}.txt").write_text(
            args_line[-1] if args_line else "MISSING"
        )


def validate_stage(ramp_dir, stage=None):
    """Fail-closed evidence gate; identity checks run in per-stage mode too.
    Without a stage this is the global final gate, retained as JSON."""
    cmd = [sys.executable, str(VALIDATOR), "--ramp-dir", str(ramp_dir)]
    if stage is None:
        out = ramp_dir / "evidence_validation.out"
        cmd += ["--json", str(ramp_dir / "evidence_validation.json")]
    else:
        out = ramp_dir / f"stage{stage}" / "evidence_validation.out"
        cmd += ["--stage", str(stage)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out.write_text(result.stdout + result.stderr)
    return result.returncode == 0


def bench(fleet, session_offset, num_sessions, concurrency, out_json):
    cmd = [
        sys.executable, "-m", "sglang.benchmark.serving",
        "--backend", "sglang-oai-chat", "--base-url", fleet.router_url,
        "--model", fleet.model_path, "--dataset-name", "recovery-agent",
        "--recovery-profile", "agent-short",
        # ONLY the retained, preflight-verified copy is ever benchmarked.
        "--dataset-path", str(fleet.ramp_dir / DATASET_NAME),
        "--num-sessions", str(num_sessions),
        "--dataset-offset", str(session_offset),
        "--max-concurrency", str(concurrency),
        "--warmup-requests", "0", "--seed", "42", "--disable-tqdm",
        "--cache-report", "--output-file", str(out_json),
    ]
    env = {
        "PYTHONPATH": str(REPO / "python"),
        "HF_HUB_OFFLINE": "1",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
    }
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)


def run_stage_sessions(fleet, stage_dir):
    """Eight sequential single-session runs: the stage's benchmark AND its
    per-session affinity attribution (raw counter deltas retained per
    session so the validator can derive the verdict)."""
    verdicts = []
    for session in range(8):
        before = counters(fleet)
        result = bench(fleet, session, 1, 1, stage_dir / f"bench_s{session}.jsonl")
        (stage_dir / f"bench_s{session}.out").write_text(
            result.stdout[-40_000:] + result.stderr[-10_000:]
        )
        after = counters(fleet)
        delta = request_delta(before, after)
        hit_p, hit_d = pool_hits(fleet, delta)
        verdicts.append(
            {
                "session": session,
                "bench_rc": result.returncode,
                "delta": delta,
                "prefill_workers": hit_p,
                "decode_workers": hit_d,
                "sticky": len(hit_p) == 1 and len(hit_d) == 1,
            }
        )
    sticky = all(v["sticky"] and v["bench_rc"] == 0 for v in verdicts)
    spread = (
        len({w for v in verdicts for w in v["prefill_workers"]}) >= 2
        and len({w for v in verdicts for w in v["decode_workers"]}) >= 2
    )
    summary = {"all_sticky": sticky, "spread_ok": spread, "sessions": verdicts}
    (stage_dir / "affinity.json").write_text(json.dumps(summary, indent=2))
    return sticky and spread


def stage5_attribution(fleet, stage_dir):
    """Deterministic per-key router attribution: two sequential tiny probes
    per session routing key with counter deltas; two identical successful
    single-pair probes are the determinism consistent hashing promises."""
    dataset = json.loads((fleet.ramp_dir / DATASET_NAME).read_text())
    seed = dataset["metadata"]["seed"]
    keys = [
        "session-" + hashlib.sha1(f"{seed}:{i}".encode()).hexdigest()[:16]
        for i in range(len(dataset["conversations"]))
    ]

    def probe_once(key):
        before = counters(fleet)
        body = json.dumps(
            {
                "model": fleet.model_path,
                "messages": [{"role": "user", "content": "attribution probe"}],
                "max_completion_tokens": 4,
            }
        ).encode()
        request = urllib.request.Request(
            f"{fleet.router_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "X-SMG-Routing-Key": key},
            method="POST",
        )
        ok = True
        try:
            urllib.request.urlopen(request, timeout=120).read()
        except Exception:
            ok = False
        after = counters(fleet)
        hit_p, hit_d = pool_hits(fleet, request_delta(before, after))
        return {"ok": ok, "prefill_workers": hit_p, "decode_workers": hit_d}

    records = []
    for key in keys:
        probes = [probe_once(key), probe_once(key)]
        sticky = all(
            p["ok"] and len(p["prefill_workers"]) == 1 and len(p["decode_workers"]) == 1
            for p in probes
        )
        stable = (
            probes[0]["prefill_workers"] == probes[1]["prefill_workers"]
            and probes[0]["decode_workers"] == probes[1]["decode_workers"]
        )
        records.append(
            {"routing_key": key, "probes": probes, "sticky": sticky, "stable": stable}
        )
    summary = {
        "all_sticky": all(r["sticky"] and r["stable"] for r in records),
        "keys": records,
    }
    (stage_dir / "attribution.json").write_text(json.dumps(summary, indent=2))
    return summary["all_sticky"]


def wire_probe(fleet, stage_dir):
    result = subprocess.run(
        [
            sys.executable, str(WIRE_PROBE), fleet.router_url, fleet.model_path,
            str(stage_dir / "wire_probe.json"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={
            "PYTHONPATH": str(REPO / "python"),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
    )
    (stage_dir / "wire_probe.out").write_text(result.stdout + result.stderr)
    return result.returncode == 0


def router_serves_traffic(fleet, deadline_s=420):
    """After a fleet change the router's circuit breakers take time to close;
    do not start a stage until a real request round-trips."""
    probe = json.dumps(
        {
            "model": fleet.model_path,
            "messages": [{"role": "user", "content": "ready?"}],
            "max_completion_tokens": 4,
        }
    ).encode()
    start = time.time()
    while time.time() - start < deadline_s:
        request = urllib.request.Request(
            f"{fleet.router_url}/v1/chat/completions",
            data=probe,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if json.loads(response.read()).get("choices"):
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def expected_router_cmd(fleet):
    prefill_args = " ".join(
        f"--prefill {url} {fleet.bootstrap_port}" for url in fleet.prefill.values()
    )
    decode_args = " ".join(f"--decode {url}" for url in fleet.decode.values())
    port = fleet.router_url.rsplit(":", 1)[1]
    return (
        f"setsid {fleet.router_python} -m sglang_router.launch_router "
        "--pd-disaggregation --policy consistent_hashing "
        "--prefill-policy consistent_hashing --decode-policy consistent_hashing "
        f"{prefill_args} {decode_args} --host 0.0.0.0 --port {port} "
        f"> {fleet.ramp_dir}/router.log 2>&1 &"
    )


def live_router_processes():
    result = subprocess.run(
        ["pgrep", "-af", "sglang_router.launch_router"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def ensure_router(fleet):
    """The router artifact must describe the LIVE router this run uses: a
    merely-healthy process of unknown configuration is killed and replaced by
    one we launched ourselves, and the retained config/process artifacts are
    written from that launch — never inherited from a previous run."""
    cmd = expected_router_cmd(fleet)
    config_path = fleet.ramp_dir / "router_config.txt"
    port = fleet.router_url.rsplit(":", 1)[1]
    if http_ok(f"{fleet.router_url}/health"):
        live = live_router_processes()
        bound = (
            config_path.exists()
            and config_path.read_text() == cmd
            and f"--port {port}" in live
            and all(url in live for url in fleet.prefill.values())
            and all(url in live for url in fleet.decode.values())
        )
        if bound:
            return
        # Healthy but unverifiable/mismatched: replace it with our own.
        subprocess.run(
            ["pkill", "-f", "sglang_router.launch_router"],
            capture_output=True,
            check=False,
        )
        time.sleep(3)
    subprocess.Popen(["bash", "-c", cmd])
    config_path.write_text(cmd)
    for _ in range(30):
        if http_ok(f"{fleet.router_url}/health"):
            (fleet.ramp_dir / "router_process.txt").write_text(
                live_router_processes()
            )
            return
        time.sleep(2)
    raise RuntimeError("router failed to start")


def gate_fleet_ready(fleet):
    """Health + router + real-traffic gates; used before EVERY stage,
    including a direct stage-5 resume."""
    if not wait_healthy(fleet):
        return "workers not healthy"
    ensure_router(fleet)
    if not router_serves_traffic(fleet):
        return "router circuits open"
    return None


def init_ledger(ramp_dir, start_stage):
    """Structured per-stage status ledger: the run's identity plus one
    status/reason record per stage and the derived last_passing_stage. A
    full run creates it fresh; a resume keeps prior stages' records."""
    path = ramp_dir / "ledger.json"
    if start_stage > 1 and path.exists():
        return
    spec_digest = hashlib.sha256(
        (TRACKED_DIR / "ramp_run_spec.json").read_bytes()
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "run_id": time.strftime("%Y-%m-%d_%H-%M-%S"),
                "commit": head_commit(),
                "spec_sha256": spec_digest,
                "stages": {
                    str(stage): {"status": "pending", "reason": None}
                    for stage in range(1, 6)
                },
                "last_passing_stage": None,
            },
            indent=2,
        )
    )


def update_ledger(ramp_dir, stage, status, reason=None):
    path = ramp_dir / "ledger.json"
    ledger = json.loads(path.read_text())
    ledger["stages"][str(stage)] = {"status": status, "reason": reason}
    passed = [
        int(s) for s, v in ledger["stages"].items() if v["status"] == "passed"
    ]
    ledger["last_passing_stage"] = max(passed) if passed else None
    path.write_text(json.dumps(ledger, indent=2))


def run_small_stage(fleet, ramp_dir, stage, roles_to_launch, prior, reuse_fleet):
    """One staged configuration: (re)launch, gate, benchmark, retain, and
    validate. Returns None on success or a failure reason string."""
    stage_dir = ramp_dir / f"stage{stage}"
    stage_dir.mkdir(exist_ok=True)
    if reuse_fleet:
        print(f"=== stage {stage}: reusing in-flight fleet", flush=True)
        for rank, (role, _ip) in fleet.ranks.items():
            prior[rank] = worker_launch_stage(role, stage)
    else:
        print(f"=== stage {stage}: relaunching {sorted(roles_to_launch)}", flush=True)
        kill_ranks = [
            r for r, (role, _) in fleet.ranks.items() if role in roles_to_launch
        ]
        kill_workers(fleet, kill_ranks)
        time.sleep(8)
        for rank, (role, _ip) in fleet.ranks.items():
            if role in roles_to_launch:
                worker_stage = worker_launch_stage(role, stage)
                launch_worker(fleet, rank, role, worker_stage)
                prior[rank] = worker_stage
    failure = gate_fleet_ready(fleet)
    if failure:
        return failure
    collect_stage_evidence(fleet, stage_dir, prior)
    (stage_dir / "counters_pre.json").write_text(json.dumps(counters(fleet), indent=2))
    affinity_ok = run_stage_sessions(fleet, stage_dir)
    (stage_dir / "counters_post.json").write_text(
        json.dumps(counters(fleet), indent=2)
    )
    probe_ok = wire_probe(fleet, stage_dir)
    evidence_ok = validate_stage(ramp_dir, stage)
    print(
        f"stage {stage}: affinity_ok={affinity_ok} wire_ok={probe_ok} "
        f"evidence_ok={evidence_ok}",
        flush=True,
    )
    if not evidence_ok:
        return "evidence validation failed"
    if not (affinity_ok and probe_ok):
        return f"affinity_ok={affinity_ok} wire_ok={probe_ok}"
    return None


def run_final_stage(fleet, ramp_dir, prior):
    """Final configuration, 32 concurrent sessions after a cache flush.
    Returns None on success or a failure reason string."""
    stage_dir = ramp_dir / "stage5"
    stage_dir.mkdir(exist_ok=True)
    failure = gate_fleet_ready(fleet)
    if failure:
        return failure
    collect_stage_evidence(fleet, stage_dir, prior)
    pre = counters(fleet)
    for url in list(fleet.prefill.values()) + list(fleet.decode.values()):
        try:
            urllib.request.urlopen(
                urllib.request.Request(f"{url}/flush_cache", method="POST"),
                timeout=30,
            )
        except Exception as error:
            print(f"flush_cache {url}: {error}", flush=True)
    (stage_dir / "counters_pre.json").write_text(json.dumps(pre, indent=2))
    result = bench(fleet, 0, 32, 8, stage_dir / "bench_32s.jsonl")
    (stage_dir / "bench_32s.out").write_text(
        result.stdout[-120_000:] + result.stderr[-20_000:]
    )
    (stage_dir / "counters_post.json").write_text(
        json.dumps(counters(fleet), indent=2)
    )
    attribution_ok = stage5_attribution(fleet, stage_dir)
    probe_ok = wire_probe(fleet, stage_dir)
    evidence_ok = validate_stage(ramp_dir, 5)
    print(
        f"stage 5: bench_rc={result.returncode} attribution_ok={attribution_ok} "
        f"wire_ok={probe_ok} evidence_ok={evidence_ok}",
        flush=True,
    )
    if result.returncode != 0:
        return f"benchmark exited {result.returncode}"
    if not evidence_ok:
        return "evidence validation failed"
    if not (attribution_ok and probe_ok):
        return f"attribution_ok={attribution_ok} wire_ok={probe_ok}"
    return None


STAGE_PLAN = {
    1: {"prefill", "decode"},
    2: {"prefill", "decode"},
    3: {"prefill", "decode"},
    4: {"prefill"},  # HiCache changes prefill only; decode stays at stage 3
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramp-dir", required=True)
    parser.add_argument("--fleet-config", required=True)
    parser.add_argument(
        "--start-stage", type=int, default=1, choices=[1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--skip-relaunch",
        action="store_true",
        help="The first stage's fleet is already launching (resume after a "
        "premature health timeout); wait for it instead of relaunching.",
    )
    args = parser.parse_args()
    ramp_dir = Path(args.ramp_dir)
    ramp_dir.mkdir(parents=True, exist_ok=True)
    fleet = Fleet(json.loads(Path(args.fleet_config).read_text()), ramp_dir)
    spec = load_spec()
    start_stage = args.start_stage
    preflight(fleet, spec, start_stage)
    init_ledger(ramp_dir, start_stage)

    # rank -> stage the worker was actually launched at (evidence lives in
    # that stage's log file). A direct stage-5 resume reuses the final fleet:
    # prefill launched at stage 4, decode at stage 3.
    prior = (
        {r: worker_launch_stage(role, 5) for r, (role, _ip) in fleet.ranks.items()}
        if start_stage == 5
        else {}
    )
    overall_ok = True
    # Attempt every required stage, recording each outcome — a stage failure
    # is a FAILED ledger entry with a reason, not an abort. (Preflight is the
    # only global interlock; each stage relaunches its own configuration.)
    for stage, base_roles in STAGE_PLAN.items():
        if stage < start_stage:
            continue
        roles = base_roles
        if stage == 4 and start_stage == 4:
            # Resuming at stage 4 still needs the decode fleet at stage-3
            # config, so relaunch both roles.
            roles = {"prefill", "decode"}
        update_ledger(ramp_dir, stage, "attempted")
        reason = run_small_stage(
            fleet,
            ramp_dir,
            stage,
            roles,
            prior,
            reuse_fleet=args.skip_relaunch and stage == start_stage,
        )
        if reason:
            print(f"STAGE {stage} FAILED: {reason}", flush=True)
            (ramp_dir / f"stage{stage}" / "STAGE_FAILED").write_text(reason)
            update_ledger(ramp_dir, stage, "failed", reason)
            overall_ok = False
        else:
            update_ledger(ramp_dir, stage, "passed")
    update_ledger(ramp_dir, 5, "attempted")
    reason = run_final_stage(fleet, ramp_dir, prior)
    if reason:
        print(f"STAGE 5 FAILED: {reason}", flush=True)
        (ramp_dir / "stage5" / "STAGE_FAILED").write_text(reason)
        update_ledger(ramp_dir, 5, "failed", reason)
        overall_ok = False
    else:
        update_ledger(ramp_dir, 5, "passed")

    # Final global gate: all stages + full run identity together.
    global_ok = validate_stage(ramp_dir)
    print(f"global evidence gate: {'PASS' if global_ok else 'FAIL'}", flush=True)
    overall_ok &= global_ok
    ledger = json.loads((ramp_dir / "ledger.json").read_text())
    print(f"last passing stage: {ledger['last_passing_stage']}", flush=True)
    print(f"RAMP {'COMPLETE' if overall_ok else 'FAILED'}", flush=True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
