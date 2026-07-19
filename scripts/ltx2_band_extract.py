#!/usr/bin/env python3
"""Extract the per-step loss band from a training log (MJ-1041 band gate).

Works on: musubi train.log (tqdm postfix `loss_v=X` per step) and the Mojo
trainer log (`loss X` on the progress line). Prints median / frac>thr / max.

Usage: ltx2_band_extract.py <log> [--key loss_v|loss] [--thr 0.30]
"""
import re
import statistics
import sys

path = sys.argv[1]
key = "loss_v"
thr = 0.30
for i, a in enumerate(sys.argv):
    if a == "--key":
        key = sys.argv[i + 1]
    if a == "--thr":
        thr = float(sys.argv[i + 1])

text = open(path, errors="replace").read().replace("\r", "\n")
vals = []
if key == "loss_v":
    # musubi tqdm postfix: take the LAST loss_v per step index
    per_step = {}
    for m in re.finditer(r"steps:\s+\d+%\|[^|]*\|\s+(\d+)/\d+.*?loss_v=([0-9.eE+-]+)", text):
        per_step[int(m.group(1))] = float(m.group(2))
    vals = [v for _, v in sorted(per_step.items())]
else:
    # Mojo progress line: "| loss 0.4560 |"
    for m in re.finditer(r"\|\s*loss\s+([0-9.eE+-]+)\s*\|", text):
        vals.append(float(m.group(1)))

if not vals:
    print("NO LOSS VALUES FOUND")
    sys.exit(2)
vals_sorted = sorted(vals)
frac = sum(1 for v in vals if v > thr) / len(vals)
print(f"n={len(vals)} median={statistics.median(vals):.4f} "
      f"frac>{thr:.2f}={frac:.2%} max={max(vals):.4f} min={min(vals):.4f} "
      f"first5={[round(v,4) for v in vals[:5]]} last5={[round(v,4) for v in vals[-5:]]}")
