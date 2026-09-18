#!/usr/bin/env python3
"""Render the section-8 size-sweep table.

The result filename carries conc/batch but not ISL/OSL, and four scenarios
share batch 16 — so the scenario is recovered from the benchmark's own
``input_lens`` / ``output_lens`` (``--random-range-ratio 0.8`` means the max
sampled length is the nominal ISL/OSL).
"""

import glob
import json
import os
import re
import sys

ARM = {
    "split_trtllm_fia2a": "TRTLLM routed",
    "split_cutedsl_fia2a": "FlashInfer cuTeDSL",
    "mega_native_deepgemm": "Native MegaMoE",
}
# (ISL, OSL, batch) -> scenario
SCEN = {
    (1024, 1024, 128): "Chat",
    (8192, 1024, 64): "RAG",
    (16384, 1024, 16): "Summarization",
    (4096, 2048, 16): "Code gen",
    (16384, 4096, 32): "Agentic",
    (32768, 4096, 16): "Agentic (long)",
    (4096, 16384, 16): "Reasoning",
}
ORDER = [
    "Chat",
    "RAG",
    "Summarization",
    "Code gen",
    "Agentic",
    "Agentic (long)",
    "Reasoning",
]
NOMINAL = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def nominal(v):
    return min(NOMINAL, key=lambda n: abs(n - v))


def main(dirs):
    rows = []
    for d in dirs:
        for path in glob.glob(os.path.join(d, "flash_*.json")):
            base = os.path.basename(path)
            if base.endswith(".gsm8k.json"):
                continue
            m = re.match(r"flash_([a-z0-9_]+)_ep\d+_conc(\d+)b(\d+)_", base)
            if not m or m.group(1) not in ARM:
                continue
            with open(path) as fh:
                d_ = json.load(fh)
            il, ol = d_.get("input_lens"), d_.get("output_lens")
            if not il or not ol:
                continue
            key = (nominal(max(il)), nominal(max(ol)), int(m.group(3)))
            acc = None
            ap = path[: -len(".json")] + ".gsm8k.json"
            if os.path.exists(ap):
                with open(ap) as fh:
                    acc = json.load(fh).get("summary", {}).get("accuracy")
            rows.append(
                dict(
                    scen=SCEN.get(key, f"? {key}"),
                    backend=ARM[m.group(1)],
                    isl=key[0],
                    osl=key[1],
                    batch=key[2],
                    tok_s=d_.get("total_token_throughput", 0.0),
                    ttft=d_.get("median_ttft_ms", 0.0),
                    tpot=d_.get("median_tpot_ms", 0.0),
                    itl=d_.get("median_itl_ms", 0.0),
                    acc=acc,
                )
            )

    hdr = (
        f"{'Scenario':<15} {'ISL/OSL/B':>16} {'Backend':<19} {'tok/s':>9} "
        f"{'tok/s/GPU':>10} {'TTFT ms':>9} {'TPOT ms':>8} {'ITL ms':>7} "
        f"{'vs Mega':>8} {'GSM8K':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in ORDER:
        grp = [r for r in rows if r["scen"] == s]
        if not grp:
            continue
        base = next((g for g in grp if g["backend"] == "Native MegaMoE"), None)
        for r in sorted(grp, key=lambda r: -r["tok_s"]):
            ratio = (
                f"{r['tok_s'] / base['tok_s']:.3f}x" if base and base["tok_s"] else "--"
            )
            acc = f"{r['acc']:.3f}" if isinstance(r["acc"], (int, float)) else "--"
            shape = f"{r['isl']}/{r['osl']}/{r['batch']}"
            print(
                f"{s:<15} {shape:>16} {r['backend']:<19} {r['tok_s']:>9.1f} "
                f"{r['tok_s'] / 4:>10.1f} {r['ttft']:>9.1f} {r['tpot']:>8.2f} "
                f"{r['itl']:>7.2f} {ratio:>8} {acc:>6}"
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
