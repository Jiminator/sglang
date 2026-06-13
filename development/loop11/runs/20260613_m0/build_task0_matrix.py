#!/usr/bin/env python3
"""Loop 11 task0: build durable per-probe evidence extracts + the unified capacity matrix.

Reads every probe serve.log under probe_logs/ (R0 `pNN_*`, R1 fill, R2 bounded `bnd_`/`ctl_`,
plus the rs16k context-length supplementary set), parses the authoritative boot-stage memory
lines + served ServerArgs from the log, and MERGES the smoke result + first-fail reason from the
tracked driver TSVs (probes.tsv / probes_fill.tsv / probes_bounded.tsv). Emits:
  - probe_logs/<name>_evidence.txt   durable, untruncated per-probe extract (tracked .txt),
                                     including graph_capture + smoke + note (R2)
  - task0_matrix.tsv                 one row per probe, with graph_capture/smoke/note columns
  - task0_ceilings.md                per-config boot-ceiling summary (highest-pass + first-fail)
  - task0_bounded_compare.md         bounded vs unbounded right-sized ready-GB delta (R2)

The matrix reports the BOOT/CAPTURE/SMOKE ceiling per config (an upper bound on the servable
fraction); the sustained-stable served fraction comes from the task4/M2 ladders under real load.
Source serve.logs are gitignored (repo policy: *.log); these extracts are the tracked evidence.
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "probe_logs")

# Parse the canonical config from a probe filename. Optional prefix:
#   pNN_  R0 anchor rows;  bnd_  R2 bounded fail_closed;  ctl_  R2 full_fallback control.
# R1 fill rows have no prefix (fp16_on_rs_085_serve.log).
NAME_RE = re.compile(
    r"^(?:(p\d+|bnd|ctl)_)?(fp16|int8|tf)_(on|off)_(def|rs|rs16k)_(\d{3})_serve\.log$"
)


def parse_log(path):
    with open(path, "r", errors="replace") as f:
        txt = f.read()
    rec = {}
    sa = re.search(r"server_args=ServerArgs\([^\n]*", txt)
    sa = sa.group(0) if sa else None

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
    # The DS selector-width ladder the runner actually built (proves bounded vs full).
    sw = re.search(r"DS selector-width graph variants[^\n]*", txt)
    rec["ds_width_ladder"] = sw.group(0).split("]")[0] + "]" if sw else None

    def tp0(pat):
        for ln in txt.splitlines():
            if "TP0]" in ln and re.search(pat, ln):
                return ln.strip()
        return None

    rec["load_weight_end"] = tp0(r"Load weight end")
    rec["kv_alloc"] = tp0(r"KV Cache is allocated")
    rec["pool_end"] = tp0(r"Memory pool end")
    rec["table"] = next(
        (ln.strip() for ln in txt.splitlines() if "token_label_table:" in ln), None
    )
    rec["capture_begin"] = tp0(r"Capture cuda graph begin")
    rec["capture_end"] = tp0(r"Capture cuda graph end")
    rec["final_line"] = tp0(r"max_total_num_tokens=")
    rec["server_ready"] = "The server is fired up" in txt
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


def load_driver_fields():
    """Merge smoke + note (+ policy) per probe from the tracked driver TSVs, by header name."""
    fields = {}
    for fn in ("probes.tsv", "probes_fill.tsv", "probes_bounded.tsv", "probes_bounded_fill.tsv"):
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        if not lines:
            continue
        hdr = lines[0].split("\t")
        idx = {name: i for i, name in enumerate(hdr)}
        for ln in lines[1:]:
            cols = ln.split("\t")

            def col(name):
                i = idx.get(name)
                return cols[i] if i is not None and i < len(cols) else "-"

            probe = col("probe")
            if probe and probe != "-":
                fields[probe] = {
                    "graph_capture": col("graph_capture"),
                    "smoke": col("smoke"),
                    "note": col("note"),
                    "policy": col("policy"),
                }
    return fields


def main():
    driver = load_driver_fields()
    rows = []
    for path in sorted(glob.glob(os.path.join(LOGDIR, "*_serve.log"))):
        base = os.path.basename(path)
        m = NAME_RE.match(base)
        if not m:
            continue
        prefix, variant, idx, env, f3 = m.groups()
        name = base[: -len("_serve.log")]
        rec = parse_log(path)
        frac = f"0.{f3[1:]}"
        cap = rec["max_total_num_tokens"]
        bs = cap // 4608 if cap else 0
        status = "OK" if (rec["server_ready"] and rec["capture_ok"]) else "BOOT_FAIL"
        d = driver.get(name, {})
        smoke = d.get("smoke", "-")
        note = d.get("note", "-")
        graph_capture = d.get("graph_capture", "yes" if rec["capture_ok"] else "NO")
        policy = d.get("policy", "-")
        kind = (
            "bounded" if prefix in ("bnd", "ctl")
            else "ctxlen" if env == "rs16k"
            else "grid"
        )

        ev = os.path.join(LOGDIR, f"{name}_evidence.txt")
        with open(ev, "w") as out:
            out.write(f"# Durable evidence extract for probe {name}\n")
            out.write(
                f"# config: variant={variant} indexer={idx} envelope={env} "
                f"mem_fraction={frac} kind={kind} width_policy={policy}\n"
            )
            out.write(
                "# Source serve.log is gitignored (repo policy *.log); this extract is the "
                "tracked proof. Loop11 task0 (R0-R2).\n\n"
            )
            out.write("## served args (from ServerArgs)\n")
            for k in (
                "mem_fraction_static", "max_running_requests", "cuda_graph_max_bs",
                "context_length", "enable_double_sparsity", "signature_dtype",
            ):
                out.write(f"  {k} = {rec[k]}\n")
            if rec["ds_width_ladder"]:
                out.write(f"  {rec['ds_width_ladder']}\n")
            args_file = os.path.join(LOGDIR, f"{name}_args.txt")
            if os.path.exists(args_file):
                out.write("\n## launch line + probe env\n")
                with open(args_file) as af:
                    out.write("  " + af.read().strip().replace("\n", "\n  ") + "\n")
            out.write("\n## TP0 boot-stage memory lines (durable proof set)\n")
            for k in (
                "load_weight_end", "kv_alloc", "pool_end", "table",
                "capture_begin", "capture_end", "final_line",
            ):
                out.write(f"  {rec[k]}\n")
            out.write("\n## derived + outcome\n")
            out.write(f"  max_total_num_tokens = {cap}\n")
            out.write(f"  bs_cap (floor(cap/4608)) = {bs}\n")
            out.write(f"  table_GB = {rec['table_gb']}\n")
            out.write(f"  kv_pool_GB = {rec['kv_gb']}\n")
            out.write(f"  ready_GB (available_gpu_mem at server-ready) = {rec['ready_gb']}\n")
            out.write(f"  graph_capture = {graph_capture}\n")
            out.write(f"  smoke = {smoke}\n")
            out.write(f"  note (first-fail reason if BOOT_FAIL) = {note}\n")
            out.write(f"  status = {status}\n")
        rows.append(
            dict(
                name=name, variant=variant, idx=idx, env=env, frac=frac, f3=f3,
                kind=kind, policy=policy, status=status, cap=cap or 0, bs=bs,
                table_gb=rec["table_gb"], ready_gb=rec["ready_gb"],
                graph_capture=graph_capture, smoke=smoke, note=note,
            )
        )

    rows.sort(key=lambda r: (r["kind"], r["variant"], r["idx"], r["env"], r["f3"]))
    tsv = os.path.join(HERE, "task0_matrix.tsv")
    with open(tsv, "w") as out:
        out.write(
            "probe\tkind\tvariant\tindexer\tenvelope\twidth_policy\tfraction\tstatus\t"
            "max_total_num_tokens\tbs_cap_4608\ttable_GB\tready_GB\tgraph_capture\tsmoke\tnote\n"
        )
        for r in rows:
            out.write(
                f"{r['name']}\t{r['kind']}\t{r['variant']}\t{r['idx']}\t{r['env']}\t"
                f"{r['policy']}\t{r['frac']}\t{r['status']}\t{r['cap']}\t{r['bs']}\t"
                f"{r['table_gb']}\t{r['ready_gb']}\t{r['graph_capture']}\t{r['smoke']}\t{r['note']}\n"
            )

    # Per-config ceiling summary — grid kind only (the unbounded {default, rs} cross-product).
    configs = {}
    for r in rows:
        if r["kind"] != "grid":
            continue
        configs.setdefault((r["variant"], r["idx"], r["env"]), []).append(r)
    md = os.path.join(HERE, "task0_ceilings.md")
    with open(md, "w") as out:
        out.write("# task0 boot/capture/smoke ceilings (12-config unbounded grid)\n\n")
        out.write(
            "Boot ceiling = highest mem_fraction that boots + captures graphs + answers the "
            "smoke. **Upper bound on the servable fraction, not the sustained-stable served "
            "fraction** (task4/M2 ladders confirm under real 4096-ISL load). These rows use the "
            "default full_fallback selector-width ladder ({compact, full}); the bounded "
            "selector-width rows are in task0_bounded_compare.md. rs16k = separate context-length "
            "set.\n\n"
        )
        out.write(
            "| variant | indexer | envelope | highest PASS (frac/bs/ready GB) | "
            "first FAIL (frac/reason) | bs>=64 cleared at |\n"
        )
        out.write("|---|---|---|---|---|---|\n")
        for key in sorted(configs):
            variant, idx, env = key
            rs = sorted(configs[key], key=lambda r: r["f3"])
            passes = [r for r in rs if r["status"] == "OK"]
            fails = [r for r in rs if r["status"] != "OK"]
            hp = (
                f"{passes[-1]['frac']} / bs{passes[-1]['bs']} / {passes[-1]['ready_gb']}"
                if passes else "—"
            )
            ff = f"{fails[0]['frac']} ({fails[0]['note']})" if fails else "≥grid-top"
            cleared = next(
                (r["frac"] for r in rs if r["status"] == "OK" and r["bs"] >= 64), "—"
            )
            out.write(f"| {variant} | {idx} | {env} | {hp} | {ff} | {cleared} |\n")

        # Canonical BOUNDED right-sized ceiling table (fail_closed [4608], rs envelope) —
        # the bounded analog of the unbounded grid above, per the plan's right-sized axis.
        bconfigs = {}
        for r in rows:
            if r["kind"] == "bounded" and r["policy"] == "fail_closed":
                bconfigs.setdefault((r["variant"], r["idx"]), []).append(r)
        out.write(
            "\n## bounded right-sized ceilings (fail_closed [4608], rs envelope)\n\n"
            "Same boot/capture/smoke ceiling, with the bounded selector-width feature "
            "(no full-width DS graph). Compare ready GB to the `rs` rows above (the unbounded "
            "control); the bounded gain is ~0.3 GB — see task0_bounded_compare.md.\n\n"
            "| variant | indexer | envelope | highest PASS (frac/bs/ready GB) | "
            "first FAIL (frac/reason) | bs>=64 cleared at |\n"
            "|---|---|---|---|---|---|\n"
        )
        for key in sorted(bconfigs):
            variant, idx = key
            rs = sorted(bconfigs[key], key=lambda r: r["f3"])
            passes = [r for r in rs if r["status"] == "OK"]
            fails = [r for r in rs if r["status"] != "OK"]
            hp = (
                f"{passes[-1]['frac']} / bs{passes[-1]['bs']} / {passes[-1]['ready_gb']}"
                if passes else "—"
            )
            ff = f"{fails[0]['frac']} ({fails[0]['note']})" if fails else "≥grid-top"
            cleared = next(
                (r["frac"] for r in rs if r["status"] == "OK" and r["bs"] >= 64), "—"
            )
            out.write(f"| {variant} | {idx} | rs(bounded) | {hp} | {ff} | {cleared} |\n")

    # Bounded vs unbounded right-sized comparison (R2): match each bounded probe to the
    # unbounded grid row at the same (variant, indexer, rs, fraction) and report the ready delta.
    grid_by_key = {
        (r["variant"], r["idx"], r["frac"]): r
        for r in rows if r["kind"] == "grid" and r["env"] == "rs"
    }
    bounded = [r for r in rows if r["kind"] == "bounded"]
    if bounded:
        bmd = os.path.join(HERE, "task0_bounded_compare.md")
        with open(bmd, "w") as out:
            out.write("# task0 bounded selector-width vs unbounded right-sized (R2)\n\n")
            out.write(
                "Bounded = `selector_width_overflow_policy=fail_closed`, "
                "`selector_width_buckets=[4608]` → the DS graph captures ONLY the 4608 width "
                "(no full 202752-width DS scratch). Unbounded = the R1 right-sized row at the "
                "same point (default full_fallback ladder {5120, full}). `ready_GB` delta is the "
                "reclaimed full-width DS graph scratch. `ctl_` rows are full_fallback with "
                "buckets=[4608] ({4608, full}) — the matched control isolating the full-width "
                "drop.\n\n"
            )
            out.write(
                "| probe | variant | idx | frac | policy | bs_cap | ready GB | "
                "unbounded ready GB | delta GB | smoke |\n"
            )
            out.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(bounded, key=lambda r: (r["variant"], r["idx"], r["f3"], r["policy"])):
                u = grid_by_key.get((r["variant"], r["idx"], r["frac"]))
                ub = u["ready_gb"] if u else None
                delta = (
                    round(r["ready_gb"] - ub, 2)
                    if (r["ready_gb"] is not None and ub is not None) else "—"
                )
                out.write(
                    f"| {r['name']} | {r['variant']} | {r['idx']} | {r['frac']} | {r['policy']} "
                    f"| {r['bs']} | {r['ready_gb']} | {ub} | {delta} | {r['smoke']} |\n"
                )

    print(f"wrote {len(rows)} evidence extracts, {tsv}, {md}")
    for f in (tsv, md):
        with open(f) as fh:
            print(f"\n=== {os.path.basename(f)} ===\n" + fh.read())
    bmd = os.path.join(HERE, "task0_bounded_compare.md")
    if os.path.exists(bmd):
        with open(bmd) as fh:
            print(f"\n=== task0_bounded_compare.md ===\n" + fh.read())


if __name__ == "__main__":
    main()
