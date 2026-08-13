# serenitymojo/models/ltx2/parity/ltx2_av_step_budget_probe.mojo
#
# WHERE DOES THE AV STEP GO? The AV arm ran at 117-122 s/step; after the
# scalar-byte-loop fix (632b603) it runs at 23.6 s/step. Re-point OBSERVED_STEP_S
# whenever the step time changes, or the shares printed below are nonsense. Recompute is NOT the suspect: gate 0 measured one block
# forward at 35.16 ms, so 48 recomputes = 1.69 s, ~1.4% of the step.
#
# The HYPOTHESIS to test is that per-block weight materialisation dominates:
# `ltx2_av_stack.mojo` calls `LTX2AVBlockWeights.load(ckpt, i, cfg, ctx).to_f32(ctx)`
# once per block in the FORWARD and again per block in the BACKWARD = 96
# materialisations/step, off the 39.13 GB bf16-dequant AV checkpoint.
#
# Measure, don't assert (the Klein lesson: the "LoRA GEMM storm" was SDPA). This
# probe splits the two phases so the step budget is attributable:
#
#   T_load  = LTX2AVBlockWeights.load(...)      bf16 read off the checkpoint
#   T_f32   = .to_f32(ctx)                      dtype conversion
#   T_fwd   = ltx2_block_forward_av_train       (gate-0 cross-check, 35.16 ms)
#
# then reports 96*(T_load+T_f32) + 96*T_fwd against the observed step time, and
# what fraction is left unexplained.
#
# Measurement only, one block at a time. No training run.
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_step_budget_probe.mojo \
#   -o /tmp/ltx2_av_step_budget && /tmp/ltx2_av_step_budget

from max.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import env_or, serenity_checkpoint
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights

comptime CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime NUM_LAYERS = 48
comptime REPS = 5
comptime OBSERVED_STEP_S = Float64(23.6)   # median steps 2-4, post scalar-loop fix (632b603)


def _median(var v: List[Float64]) -> Float64:
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    return v[len(v) // 2]


def _fmt(v: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0: s += ", "
        s += String(v[i])
    return s + "]"


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt = env_or("LTX2_AV_CKPT", serenity_checkpoint(String(CKPT_NAME)))
    print("=== AV STEP BUDGET: where do the ~119 s go? ===")
    print("  checkpoint:", ckpt)
    print("  blocks=", NUM_LAYERS, " reps=", REPS)

    var st = ShardedSafeTensors.open(ckpt)
    _ = st

    # warmup (first touch pages the file / warms the allocator)
    var w0 = LTX2AVBlockWeights.load(ckpt, 0, cfg, ctx)
    var w0f = w0^.to_f32(ctx)
    _ = w0f^
    ctx.synchronize()

    # ── (1) LOAD: bf16 materialisation off the checkpoint ────────────────────
    var load_ms = List[Float64]()
    for r in range(REPS):
        var bi = r % NUM_LAYERS
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var w = LTX2AVBlockWeights.load(ckpt, bi, cfg, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        load_ms.append(Float64(t1 - t0) / 1.0e6)
        _ = w^
    print("  (1) LOAD  per block reps_ms=", _fmt(load_ms),
          " MEDIAN=", _median(load_ms.copy()), "ms")

    # ── (2) to_f32: dtype conversion of an already-loaded block ──────────────
    var f32_ms = List[Float64]()
    for r in range(REPS):
        var bi = r % NUM_LAYERS
        var w = LTX2AVBlockWeights.load(ckpt, bi, cfg, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var wf = w^.to_f32(ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        f32_ms.append(Float64(t1 - t0) / 1.0e6)
        _ = wf^
    print("  (2) to_f32 per block reps_ms=", _fmt(f32_ms),
          " MEDIAN=", _median(f32_ms.copy()), "ms")

    # ── budget ───────────────────────────────────────────────────────────────
    var t_load = _median(load_ms.copy())
    var t_f32 = _median(f32_ms.copy())
    var t_mat = t_load + t_f32
    # 96 materialisations = 48 forward + 48 backward (the backward reloads each
    # block's weights to run its own backward; the recompute REUSES that same w).
    var mat_s = Float64(2 * NUM_LAYERS) * t_mat / 1000.0
    var fwd_s = Float64(2 * NUM_LAYERS) * 35.16 / 1000.0   # gate-0 block forward
    print("")
    print("  per-block materialise (load + to_f32) =", t_mat, "ms")
    print("  x96 (48 fwd + 48 bwd)                 =", mat_s, "s")
    print("  block compute x96 (gate-0 35.16ms)    =", fwd_s, "s")
    print("  observed step                         =", OBSERVED_STEP_S, "s")
    print("  materialise share of step             =",
          mat_s / OBSERVED_STEP_S * 100.0, "%")
    print("  unexplained remainder                 =",
          OBSERVED_STEP_S - mat_s - fwd_s, "s")
    print("")
    if mat_s > OBSERVED_STEP_S * 0.5:
        print("  VERDICT: weight materialisation DOMINATES the step.")
        print("    -> the fix is a resident/cached block store, not a compute change.")
        print("    -> headroom available: 24564 - 7954 = ~16.6 GiB at current peak.")
    else:
        print("  VERDICT: materialisation does NOT dominate — the step time is")
        print("    elsewhere. Do NOT build a resident block store on this evidence;")
        print("    profile the remainder first.")
    print("ltx2_av_step_budget_probe DONE")
