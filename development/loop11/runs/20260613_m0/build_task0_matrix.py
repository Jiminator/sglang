#!/usr/bin/env python3
"""Loop 11 task0 (R1): build durable per-probe evidence extracts + the unified capacity matrix.

Reads every probe serve.log under probe_logs/ (R0 `pNN_*` + R1 fill, plus the rs16k
context-length supplementary set), parses the authoritative boot-stage memory lines and the
served ServerArgs, and emits:
  - probe_logs/<name>_evidence.txt   durable, untruncated per-probe extract (tracked .txt)
  - task0_matrix.tsv                 one row per (variant, indexer, envelope, fraction)
  - task0_ceilings.md                per-config boot-ceiling summary (highest-pass + first-fail)

The matrix reports the BOOT/CAPTURE/SMOKE ceiling per config (an upper bound on the servable
fraction); the sustained-stable served fraction comes from the task4/M2 ladders under real load.
Source serve.logs are gitignored (repo policy: *.log); these extracts are the tracked evidence.
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "probe_logs")

# Parse the canonical config from a probe filename.
# R0:   p03_fp16_on_def_080_serve.log / p05_fp16_on_rs16k_080_serve.log
# R1:   fp16_on_rs_085_serve.log
NAME_RE = re.compile(
    r"^(?:p\d+_)?(fp16|int8|tf)_(on|off)_(def|rs|rs16k)_(\d{3})_serve\.log$"
)


def grep1(text, pat):
    m = re.search(pat, text)
    return m.group(0) if m else None


def grepN(text, pat, flags=0):
    return re.findall(pat, text, flags)


def parse_log(path):
    with open(path, "r", errors="replace") as f:
        txt = f.read()
    rec = {}
    # Authoritative served args (single ServerArgs line).
    sa = grep1(txt, r"server_args=ServerArgs\([^\n]*")
    def arg(name):
        if not sa:
            return None
        m = re.search(rf"{name}=([^,\)]+)", sa)
        return m.group(1) if m else None
    rec["mem_fraction_static"] = arg("mem_fraction_static")
    rec["cuda_graph_max_bs"] = arg("cuda_graph_max_bs")
    rec["enable_double_sparsity"] = arg("enable_double_sparsity")
    # max_running_requests / context_len are None in the initial ServerArgs for the default
    # envelope (derived at pool init); prefer the effective values from the final boot line.
    m = re.search(r"max_running_requests=(\d+)", txt)
    rec["max_running_requests"] = m.group(1) if m else arg("max_running_requests")
    m = re.search(r"context_len=(\d+)", txt)
    rec["context_length"] = m.group(1) if m else arg("context_length")
    sig = re.search(r'"signature_dtype":\s*"(\w+)"', txt)
    rec["signature_dtype"] = sig.group(1) if sig else None
    # TP0 boot-stage lines (durable proof set).
    def tp0(pat):
        for ln in txt.splitlines():
            if "TP0]" in ln and re.search(pat, ln):
                return ln.strip()
        return None
    rec["load_weight_end"] = tp0(r"Load weight end")
    rec["kv_alloc"] = tp0(r"KV Cache is allocated")
    rec["pool_end"] = tp0(r"Memory pool end")
    rec["table"] = next((ln.strip() for ln in txt.splitlines()
                         if "token_label_table:" in ln), None)
    rec["capture_begin"] = tp0(r"Capture cuda graph begin")
    rec["capture_end"] = tp0(r"Capture cuda graph end")
    rec["final_line"] = tp0(r"max_total_num_tokens=")
    rec["server_ready"] = "The server is fired up" in txt
    # Derived scalars.
    m = re.search(r"max_total_num_tokens=(\d+)", txt)
    rec["max_total_num_tokens"] = int(m.group(1)) if m else None
    m = re.search(r"available_gpu_mem=([0-9.]+) GB", txt)
    rec["ready_gb"] = float(m.group(1)) if m else None
    m = re.search(r"token_label_table: ([0-9.]+) GB", txt)
    rec["table_gb"] = float(m.group(1)) if m else None
    m = re.search(r"KV size: ([0-9.]+) GB", txt)
    rec["kv_gb"] = float(m.group(1)) if m else None
    rec["capture_ok"] = rec["capture_end"] is not None
    return rec


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOGDIR, "*_serve.log"))):
        base = os.path.basename(path)
        m = NAME_RE.match(base)
        if not m:
            continue
        variant, idx, env, f3 = m.groups()
        name = base[:-len("_serve.log")]
        rec = parse_log(path)
        frac = f"0.{f3[1:]}" if f3.startswith("0") else f"0.{f3}"
        cap = rec["max_total_num_tokens"]
        bs = cap // 4608 if cap else 0
        # boot/capture/smoke status: ready iff server fired up + capture end seen.
        status = "OK" if (rec["server_ready"] and rec["capture_ok"]) else "BOOT_FAIL"
        # Durable extract.
        ev = os.path.join(LOGDIR, f"{name}_evidence.txt")
        with open(ev, "w") as out:
            out.write(f"# Durable evidence extract for probe {name}\n")
            out.write(f"# config: variant={variant} indexer={idx} envelope={env} "
                      f"mem_fraction={frac}\n")
            out.write("# Source serve.log is gitignored (repo policy *.log); this extract is "
                      "the tracked proof. Loop11 R1.\n\n")
            out.write("## served args (from ServerArgs)\n")
            for k in ("mem_fraction_static", "max_running_requests", "cuda_graph_max_bs",
                      "context_length", "enable_double_sparsity", "signature_dtype"):
                out.write(f"  {k} = {rec[k]}\n")
            args_file = os.path.join(LOGDIR, f"{name}_args.txt")
            if os.path.exists(args_file):
                out.write("\n## launch line + probe env (R1 fill)\n")
                with open(args_file) as af:
                    out.write("  " + af.read().strip().replace("\n", "\n  ") + "\n")
            out.write("\n## TP0 boot-stage memory lines (durable proof set)\n")
            for k in ("load_weight_end", "kv_alloc", "pool_end", "table",
                      "capture_begin", "capture_end", "final_line"):
                out.write(f"  {rec[k]}\n")
            out.write(f"\n## derived\n")
            out.write(f"  max_total_num_tokens = {cap}\n")
            out.write(f"  bs_cap (floor(cap/4608)) = {bs}\n")
            out.write(f"  table_GB = {rec['table_gb']}\n")
            out.write(f"  kv_pool_GB = {rec['kv_gb']}\n")
            out.write(f"  ready_GB (available_gpu_mem at server-ready) = {rec['ready_gb']}\n")
            out.write(f"  capture_ok = {rec['capture_ok']}\n")
            out.write(f"  server_ready = {rec['server_ready']}\n")
            out.write(f"  status = {status}\n")
        rows.append(dict(name=name, variant=variant, idx=idx, env=env, frac=frac,
                         f3=f3, status=status, cap=cap or 0, bs=bs,
                         table_gb=rec["table_gb"], ready_gb=rec["ready_gb"]))

    # Unified matrix tsv (sorted by config then fraction).
    rows.sort(key=lambda r: (r["variant"], r["idx"], r["env"], r["f3"]))
    tsv = os.path.join(HERE, "task0_matrix.tsv")
    with open(tsv, "w") as out:
        out.write("probe\tvariant\tindexer\tenvelope\tfraction\tstatus\t"
                  "max_total_num_tokens\tbs_cap_4608\ttable_GB\tready_GB\n")
        for r in rows:
            out.write(f"{r['name']}\t{r['variant']}\t{r['idx']}\t{r['env']}\t{r['frac']}\t"
                      f"{r['status']}\t{r['cap']}\t{r['bs']}\t{r['table_gb']}\t{r['ready_gb']}\n")

    # Per-config ceiling summary (12 grid configs; rs16k kept separate as supplementary).
    grid_envs = {"def", "rs"}
    configs = {}
    for r in rows:
        if r["env"] not in grid_envs:
            continue
        key = (r["variant"], r["idx"], r["env"])
        configs.setdefault(key, []).append(r)
    md = os.path.join(HERE, "task0_ceilings.md")
    with open(md, "w") as out:
        out.write("# task0 boot/capture/smoke ceilings (12-config grid)\n\n")
        out.write("Boot ceiling = highest mem_fraction that boots + captures graphs + answers the "
                  "smoke. **Upper bound on the servable fraction, not the sustained-stable served "
                  "fraction** (the latter is established on the task4/M2 ladders under real "
                  "4096-ISL load). bounded-selector-width axis (q2) UNMEASURED (needs code, not a "
                  "config knob) — kept queued. rs16k rows are a separate context-length set.\n\n")
        out.write("| variant | indexer | envelope | highest PASS (frac/bs/ready GB) | "
                  "first FAIL (frac) | bs>=64 cleared at |\n")
        out.write("|---|---|---|---|---|---|\n")
        for key in sorted(configs):
            variant, idx, env = key
            rs = sorted(configs[key], key=lambda r: r["f3"])
            passes = [r for r in rs if r["status"] == "OK"]
            fails = [r for r in rs if r["status"] != "OK"]
            if passes:
                top = passes[-1]
                hp = f"{top['frac']} / bs{top['bs']} / {top['ready_gb']}"
            else:
                hp = "—"
            ff = fails[0]["frac"] if fails else "≥grid-top (no fail in sweep)"
            cleared = next((r["frac"] for r in rs if r["status"] == "OK" and r["bs"] >= 64), "—")
            out.write(f"| {variant} | {idx} | {env} | {hp} | {ff} | {cleared} |\n")

    print(f"wrote {len(rows)} evidence extracts, {tsv}, {md}")
    # Echo the matrix + ceilings for the run log.
    with open(tsv) as f:
        print("\n=== task0_matrix.tsv ===\n" + f.read())
    with open(md) as f:
        print("\n=== task0_ceilings.md ===\n" + f.read())


if __name__ == "__main__":
    main()
