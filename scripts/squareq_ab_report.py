#!/usr/bin/env python3
"""Chunk-5 A/B gate report: parse the three 300-step train logs (bf16 / fp8 /
squareq_w4) and emit the comparison table + gate verdicts.

Gates:
  - squareq loss curve within the fp8-vs-bf16 envelope (per-window mean loss
    |squareq - bf16| <= 2x |fp8 - bf16| + 0.005 slack, windows of 50 steps)
  - s/step(squareq) <= 1.3x s/step(fp8)
  - peak VRAM(squareq) < peak VRAM(fp8)

Usage: python scripts/squareq_ab_report.py [--out evidence/squareq/klein4b_ab.md]
"""
import argparse
import json
import re
import statistics

ARMS = ["bf16", "fp8", "squareq"]
OPT_ARMS = ["squareq_g32", "squareq_g32mse"]  # included when their logs exist
STEP_RE = re.compile(
    r"step (\d+)/300 \| epoch \S+ \| loss ([0-9.]+) \| grad_norm ([0-9.]+)"
)


def parse(arm: str):
    path = f"output/klein4b_ab_{arm}/train.log"
    losses, grads = {}, {}
    perf = None
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                losses[int(m.group(1))] = float(m.group(2))
                grads[int(m.group(1))] = float(m.group(3))
            if line.startswith("[training-perf-json]"):
                perf = json.loads(line.split(" ", 1)[1])
    return losses, grads, perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/squareq/klein4b_ab.md")
    a = ap.parse_args()
    import os
    arms = list(ARMS)
    for arm in OPT_ARMS:
        if os.path.exists(f"output/klein4b_ab_{arm}/train.log"):
            arms.append(arm)
    data = {arm: parse(arm) for arm in arms}
    for arm in arms:
        assert len(data[arm][0]) == 300, f"{arm}: {len(data[arm][0])} steps parsed"

    lines = ["# Chunk 5 — Klein-4B 300-step A/B (bf16 vs fp8_e4m3 vs squareq_w4)", ""]
    lines.append("Same cache (8 samples), same seeds, 512px, rank-16 LoRA, AdamW,")
    lines.append("inline 1024x1024 samples at steps 100/200/300.")
    lines.append("")
    hdr = "| window | " + " | ".join(f"{a2} mean loss" for a2 in arms) + " | |sq-bf16| | |sq_g32-bf16| |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(arms) + 3))
    envelope_ok = True
    envelope_g32_ok = True
    for w0 in range(1, 301, 50):
        ws = range(w0, w0 + 50)
        m = {arm: statistics.mean(data[arm][0][s] for s in ws) for arm in arms}
        d_sq = abs(m["squareq"] - m["bf16"])
        d_f8 = abs(m["fp8"] - m["bf16"])
        d_g32 = abs(m.get("squareq_g32mse", m.get("squareq_g32", m["squareq"])) - m["bf16"])
        if d_sq > 2 * d_f8 + 0.005:
            envelope_ok = False
        if d_g32 > 2 * d_f8 + 0.005:
            envelope_g32_ok = False
        row = "| " + f"{w0}-{w0+49} | " + " | ".join(f"{m[a2]:.4f}" for a2 in arms)
        lines.append(row + f" | {d_sq:.4f} | {d_g32:.4f} |")
    lines.append("")
    lines.append("| arm | s/step | peak VRAM (GB) | fwd s | bwd s |")
    lines.append("|---|---|---|---|---|")
    perf = {arm: data[arm][2] for arm in arms}
    for arm in arms:
        p = perf[arm]
        lines.append(
            f"| {arm} | {p['total_seconds_per_step']:.3f} |"
            f" {p['peak_vram_bytes']/2**30:.2f} |"
            f" {p['phases']['forward_seconds']:.1f} | {p['phases']['backward_seconds']:.1f} |"
        )
    speed_ratio = (
        perf["squareq"]["total_seconds_per_step"] / perf["fp8"]["total_seconds_per_step"]
    )
    vram_ok = perf["squareq"]["peak_vram_bytes"] < perf["fp8"]["peak_vram_bytes"]
    lines.append("")
    lines.append(f"- loss-envelope gate (squareq g64): {'PASS' if envelope_ok else 'FAIL'}")
    if "squareq_g32" in arms:
        lines.append(f"- loss-envelope gate (squareq g32): {'PASS' if envelope_g32_ok else 'FAIL'}")
    lines.append(
        f"- s/step gate (<=1.3x fp8): {'PASS' if speed_ratio <= 1.3 else 'FAIL'}"
        f" (ratio {speed_ratio:.3f})"
    )
    lines.append(
        f"- VRAM gate (< fp8): {'PASS' if vram_ok else 'FAIL'}"
        f" ({perf['squareq']['peak_vram_bytes']/2**30:.2f} vs"
        f" {perf['fp8']['peak_vram_bytes']/2**30:.2f} GB)"
    )
    lines.append("")
    lines.append("Sample images: compare output sample dirs of the three arms at")
    lines.append("steps 100/200/300 (human visual gate — see arm output dirs).")
    out = "\n".join(lines) + "\n"
    with open(a.out, "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()
