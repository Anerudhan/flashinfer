#!/usr/bin/env python3
"""Collect the MoE-EP configuration matrix into a report table + CSV.

Reads the per-arm benchmark_serving JSONs written by run_matrix.sh
(``<model>_<arm>_ep<N>_conc<C>_mnbt<M>.json``) plus the optional GSM8K
sidecar (``...gsm8k.json``), and prints one row per configuration.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

# arm -> (compute backend label, transport label)
ARM_LABELS = {
    "split_cutedsl_fia2a": ("cuTeDSL split", "FlashInfer all2all"),
    "split_cutedsl_nixl": ("cuTeDSL split", "NIXL EP"),
    "split_cutedsl_deepep": ("cuTeDSL split", "DeepEP LL"),
    "split_trtllm_fia2a": ("TRTLLM routed", "FlashInfer all2all"),
    "split_trtllm_nixl": ("TRTLLM routed", "NIXL EP"),
    "split_trtllm_deepep": ("TRTLLM routed", "DeepEP LL"),
    "split_cutedsl_deepep_ht": ("cuTeDSL split", "DeepEP HT"),
    "split_trtllm_deepep_ht": ("TRTLLM routed", "DeepEP HT"),
    "mega_fi_cutedsl": ("FI MegaMoE (cuTeDSL)", "fused (in-kernel)"),
    "mega_fi_deepgemm": ("FI MegaMoE (deep_gemm)", "fused (in-kernel)"),
    "mega_native_deepgemm": ("Native MegaMoE (deep_gemm)", "fused (in-kernel)"),
}

ARM_LABELS.update(
    {
        # SGLang arms (run_sg_matrix.sh)
        "sg_split_cutedsl_fia2a": ("cuTeDSL split", "FlashInfer all2all"),
        "sg_split_cutedsl_nixl": ("cuTeDSL split", "NIXL EP"),
        "sg_split_cutedsl_deepep": ("cuTeDSL split", "DeepEP"),
        "sg_trtllm_routed_fia2a": ("TRTLLM routed", "FlashInfer all2all"),
        "sg_trtllm_routed_nixl": ("TRTLLM routed", "NIXL EP"),
        "sg_trtllm_routed_deepep": ("TRTLLM routed", "DeepEP"),
        "sg_megamoe": ("FI MegaMoE (cuTeDSL)", "fused (in-kernel)"),
        "sg_megamoe_ikr": ("FI MegaMoE +ikr", "fused (in-kernel)"),
        "sg_megamoe_cmb_nvfp4": ("FI MegaMoE +cmb nvfp4", "fused (in-kernel)"),
        "sg_megamoe_cmb_mxfp8": ("FI MegaMoE +cmb mxfp8", "fused (in-kernel)"),
    }
)

FNAME_RE = re.compile(
    r"^(?P<model>pro|flash)_(?P<arm>[a-z0-9_]+)_ep(?P<ep>\d+)"
    r"_conc(?P<conc>\d+)_mnbt(?P<mnbt>\d+)\.json$"
)

# SGLang: sg_<model>_<arm>_ep<N>_conc<C>.jsonl (bench_serving --output-file)
SG_FNAME_RE = re.compile(
    r"^sg_(?P<model>pro|flash)_(?P<arm>[a-z0-9_]+)_ep(?P<ep>\d+)"
    r"_conc(?P<conc>\d+)\.jsonl$"
)


def _load_sglang(path, info):
    """SGLang bench_serving appends one JSON object per run; take the last."""
    last = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    if last is None:
        return None
    ngpu = int(info["ep"])
    tok_s = last.get(
        "total_token_throughput",
        last.get("output_throughput", 0.0) + last.get("input_throughput", 0.0),
    )
    return {
        "model": info["model"],
        "arm": info["arm"],
        "compute": ARM_LABELS.get(info["arm"], (info["arm"], "?"))[0],
        "transport": ARM_LABELS.get(info["arm"], ("?", "?"))[1],
        "ep": ngpu,
        "conc": int(info["conc"]),
        "mnbt": 0,
        "completed": last.get("completed", 0),
        "tok_s": tok_s,
        "tok_s_gpu": tok_s / ngpu,
        "ttft_ms": last.get("median_ttft_ms", 0.0),
        "tpot_ms": last.get("median_tpot_ms", 0.0),
        "itl_ms": last.get("median_itl_ms", 0.0),
        "gsm8k": None,
    }


def load_rows(result_dirs):
    rows = []
    for d in result_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            m = SG_FNAME_RE.match(os.path.basename(path))
            if not m:
                continue
            row = _load_sglang(path, m.groupdict())
            if row is not None:
                rows.append(row)
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            base = os.path.basename(path)
            if base.endswith(".gsm8k.json"):
                continue
            m = FNAME_RE.match(base)
            if not m:
                continue
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! unreadable {base}: {exc}", file=sys.stderr)
                continue

            acc = None
            acc_path = path[: -len(".json")] + ".gsm8k.json"
            if os.path.exists(acc_path):
                try:
                    with open(acc_path) as fh:
                        a = json.load(fh)
                    # eval_gsm8k.py writes {"summary": {...}, "results": [...]}
                    summary = a.get("summary", a)
                    acc = summary.get("accuracy", summary.get("acc"))
                except Exception:  # noqa: BLE001
                    pass

            info = m.groupdict()
            ngpu = int(info["ep"])
            rows.append(
                {
                    "model": info["model"],
                    "arm": info["arm"],
                    "compute": ARM_LABELS.get(info["arm"], (info["arm"], "?"))[0],
                    "transport": ARM_LABELS.get(info["arm"], ("?", "?"))[1],
                    "ep": ngpu,
                    "conc": int(info["conc"]),
                    "mnbt": int(info["mnbt"]),
                    "completed": data.get("completed", 0),
                    "tok_s": data.get("total_token_throughput", 0.0),
                    "tok_s_gpu": data.get("total_token_throughput", 0.0) / ngpu,
                    "ttft_ms": data.get("median_ttft_ms", 0.0),
                    "tpot_ms": data.get("median_tpot_ms", 0.0),
                    "itl_ms": data.get("median_itl_ms", 0.0),
                    "gsm8k": acc,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories")
    ap.add_argument("--csv", help="also write a CSV here")
    ap.add_argument(
        "--baseline",
        default="mega_fi_cutedsl",
        help="arm used as the 1.00x reference within each (model, conc, mnbt)",
    )
    args = ap.parse_args()

    rows = load_rows(args.dirs)
    if not rows:
        print("no result files found", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: (r["model"], r["mnbt"], r["conc"], r["arm"]))

    hdr = (
        f"{'model':<6} {'compute':<22} {'transport':<19} {'conc':>5} "
        f"{'done':>5} {'tok/s':>10} {'tok/s/GPU':>10} {'TTFT ms':>9} "
        f"{'TPOT ms':>8} {'ITL ms':>7} {'vs base':>8} {'GSM8K':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["conc"], r["mnbt"]), []).append(r)

    for key in sorted(groups):
        grp = groups[key]
        base = next(
            (g for g in grp if g["arm"] == args.baseline and g["completed"] > 0), None
        )
        for r in grp:
            if r["completed"] < 1:
                print(
                    f"{r['model']:<6} {r['compute']:<22} {r['transport']:<19} "
                    f"{r['conc']:>5} {'FAIL':>5}   (0 completed requests)"
                )
                continue
            ratio = (
                f"{r['tok_s'] / base['tok_s']:.3f}x" if base and base["tok_s"] else "--"
            )
            acc = f"{r['gsm8k']:.3f}" if isinstance(r["gsm8k"], (int, float)) else "--"
            print(
                f"{r['model']:<6} {r['compute']:<22} {r['transport']:<19} "
                f"{r['conc']:>5} {r['completed']:>5} {r['tok_s']:>10.1f} "
                f"{r['tok_s_gpu']:>10.1f} {r['ttft_ms']:>9.1f} {r['tpot_ms']:>8.2f} "
                f"{r['itl_ms']:>7.2f} {ratio:>8} {acc:>7}"
            )
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
