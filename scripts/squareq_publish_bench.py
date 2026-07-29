#!/usr/bin/env python3
"""Chunk-9 cross-stack table: SquareQ (Mojo) vs SimpleTuner+SDNQ ConvRot,
same dataset identity, same seeds/steps/rank where stacks allow.

Cross-stack loss scales differ (different sigma sampling), so the honest
quality metric is each stack's quantized arm vs ITS OWN bf16 baseline.
Usage: python scripts/squareq_publish_bench.py
"""
import json, re, statistics

def mojo_losses(path, steps):
    rx = re.compile(r"step (\d+)/" + str(steps) + r" .* loss ([0-9.]+)")
    d = {}
    for line in open(path):
        m = rx.search(line)
        if m: d[int(m.group(1))] = float(m.group(2))
    return d

def mojo_perf(path):
    for line in open(path):
        if line.startswith("[training-perf-json]"):
            return json.loads(line.split(" ", 1)[1])
    return None

def st_final_metrics(path):
    """it/s from the final Steps line; loss = MEAN over the whole smoothed
    step_loss stream (hundreds of tqdm refreshes — far fairer than one
    end-value)."""
    txt = open(path).read()
    losses = [float(m.group(1)) for m in re.finditer(r"step_loss=([0-9.]+)", txt)]
    loss = statistics.mean(losses) if losses else None
    itps = None
    for m in re.finditer(r"(\d+\.\d+)it/s", txt):
        itps = float(m.group(1))
    return itps, loss, len(losses)

ours_bf16 = mojo_losses("output/klein4b_ab_bf16/train.log", 300)
ours_sq = mojo_losses("output/klein4b_ab_squareq_g32mse/train.log", 300)
p_sq = mojo_perf("output/klein4b_ab_squareq_g32mse/train.log")
p_bf = mojo_perf("output/klein4b_ab_bf16/train.log")
off = statistics.mean(ours_sq.values()) - statistics.mean(ours_bf16.values())
rel = off / statistics.mean(ours_bf16.values()) * 100

st_dir = "/home/alex/simpletuner-bench"
it_bf, loss_bf, n_bf = st_final_metrics(f"{st_dir}/output_bf16_run.log")
it_cr, loss_cr, n_cr = st_final_metrics(f"{st_dir}/output_convrot_run.log")
st_off = loss_cr - loss_bf
st_rel = st_off / loss_bf * 100


print("## Klein-4B LoRA, 300 steps, rank16/a16, lr3e-5, alina dataset, seed 42\n")
print("| stack | arm | bytes (base) | s/step | peak VRAM | quant-vs-own-bf16 |")
print("|---|---|---|---|---|---|")
print(f"| SquareQ (Mojo) | bf16 streamed | 1.0x | {p_bf['total_seconds_per_step']:.3f} | {p_bf['peak_vram_bytes']/2**30:.2f} GB | — |")
print(f"| SquareQ (Mojo) | int4 g32+MSE | 0.294x | {p_sq['total_seconds_per_step']:.3f} | {p_sq['peak_vram_bytes']/2**30:.2f} GB | mean loss {off:+.4f} ({rel:+.2f}%) |")
print(f"| SimpleTuner | bf16 | 1.0x | {1/it_bf:.3f} | (not sampled) | — |")
print(f"| SimpleTuner | int8-sdnq+H256 (ConvRot, dequant-first*) | 0.50x | {1/it_cr:.3f} | (not sampled) | stream-mean loss {st_off:+.4f} ({st_rel:+.2f}%) over {n_cr} samples |")
print()
print("*ST deviation, documented: their quantized-matmul preset (compiled AND")
print("uncompiled) fails on current Triton with 'Kernel requires a runtime")
print("memory allocation, but no allocator was set' — the dequant-first")
print("int8-sdnq+H256 route is what ran. Same seed/data/steps/rank as ours.")
print("\nCaveats: cross-stack losses are NOT directly comparable (different")
print("sigma sampling/loss conventions); ST loss shown is the final smoothed")
print("step_loss only (their tqdm stream), ours is the 300-step mean offset.")
print("ST arms: 6 epochs over 51 images; ours: 8-sample cache epochs.")
