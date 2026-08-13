# models/wan22/parity/wan22_lora_fwd_hostvsdev_bench.mojo
#
# MEASUREMENT (not a parity gate): how much of the wan step is the HOST LoRA
# forward, and is the device-resident sibling bit-equal to it?
#
# WHY: the V2_GRAPH step is 14.0 s/step, but nsys says only ~1.6 s/step is CUDA
# API and ~0.69 s/step is GPU kernels — ~12 s/step is host CPU work OUTSIDE the
# CUDA API. The prime suspect is `_add_lora_delta` → `klein_lora_fwd`
# (lora_block.mojo:~250), which per projection does:
#   x_h.copy() → from_host → GEMM → to_host → from_host → GEMM → to_host
#   → host loop `out.append(scale*dy[i])` over M*out_f floats → from_host → add
# At the real wan dims that host loop alone is M*out_f = 256*5120 = 1.31M
# element appends, and the step runs 400 of these (10 projections × 40 blocks)
# in the forward and 400 more in the graph backward's recompute.
#
# This times BOTH paths at the REAL attention-projection dims and compares their
# outputs bit-for-bit, so the fix (device LoRA forward) is justified by a number
# and its numerics class is known BEFORE it is wired in.
#
# NOTE on scale: this config is rank=16 / alpha=16 → scale == 1.0, where the
# host's F32 multiply-then-narrow and the device's bf16 mul_scalar CANNOT differ.
# The bench therefore ALSO runs a scale != 1.0 case, which is where a rounding
# difference would actually show up. Do not generalize a scale==1.0 bit result.
#
# Build (rm -f serenitymojo.mojopkg first):
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/wan22/parity/wan22_lora_fwd_hostvsdev_bench.mojo -o /tmp/wan_lora_bench
#
# Mojo 1.0.0b1, NVIDIA.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.klein.lora_block import (
    lora_adapter_to_device, klein_lora_fwd_device_resident_unfused,
)
from serenitymojo.models.wan22.wan22_block import _add_lora_delta

comptime M = 256          # S
comptime IN_F = 5120      # dim
comptime OUT_F = 5120     # dim (attention projection)
comptime RANK = 16
comptime ITERS = 40       # one "block" of 10 projections ≈ 10 iters; 40 = 400/10


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32((s >> 33) & UInt64(0x7FFFFF)) / Float32(8388608.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _adapter(lscale: Float32) -> LoraAdapter:
    return LoraAdapter(
        _randn(RANK * IN_F, 11, 0.07), _randn(OUT_F * RANK, 12, 0.05),
        RANK, IN_F, OUT_F, lscale,
        _zeros(RANK * IN_F), _zeros(RANK * IN_F),
        _zeros(OUT_F * RANK), _zeros(OUT_F * RANK),
    )


def _run(lscale: Float32, ctx: DeviceContext) raises:
    print("---- scale =", lscale, " dims M=", M, " in=", IN_F, " out=", OUT_F,
          " rank=", RANK, " iters=", ITERS, "----")
    var x = Tensor.from_host(_randn(M * IN_F, 1, 1.0), [M, IN_F], STDtype.BF16, ctx)
    var w = Tensor.from_host(_randn(OUT_F * IN_F, 2, 0.06), [OUT_F, IN_F], STDtype.BF16, ctx)
    var lo = Optional[LoraAdapter](_adapter(lscale))
    var lo_dev = lora_adapter_to_device(lo.value(), ctx)

    # ── warm both paths (JIT/alloc) ──
    var nb0 = Optional[Tensor](None)
    var warm = linear(x, w, nb0^, ctx)
    var xh0 = x.to_host(ctx)
    var wy = _add_lora_delta(warm, xh0, lo, M, ctx)
    _ = wy.to_host(ctx)
    var nb1 = Optional[Tensor](None)
    var warm2 = linear(x, w, nb1^, ctx)
    var wd = klein_lora_fwd_device_resident_unfused(x, lo_dev, M, ctx)
    _ = add(warm2, wd, ctx).to_host(ctx)
    ctx.synchronize()

    # ── HOST path (what the trainer does today), incl. the x.to_host readback
    #    that record_wan_proj_lora / wan22_block_lora_forward both perform ──
    var t0 = perf_counter_ns()
    var y_host = Tensor.from_host(_zeros(1), [1], STDtype.BF16, ctx)
    for _ in range(ITERS):
        var nb = Optional[Tensor](None)
        var base = linear(x, w, nb^, ctx)
        var x_h = x.to_host(ctx)                       # the readback
        y_host = _add_lora_delta(base, x_h, lo, M, ctx)
    ctx.synchronize()
    var host_ns = perf_counter_ns() - t0

    # ── DEVICE path (no host round trip) ──
    var t1 = perf_counter_ns()
    var y_dev = Tensor.from_host(_zeros(1), [1], STDtype.BF16, ctx)
    for _ in range(ITERS):
        var nb = Optional[Tensor](None)
        var base = linear(x, w, nb^, ctx)
        var delta = klein_lora_fwd_device_resident_unfused(x, lo_dev, M, ctx)
        y_dev = add(base, delta, ctx)
    ctx.synchronize()
    var dev_ns = perf_counter_ns() - t1

    var host_ms = Float64(host_ns) / 1.0e6
    var dev_ms = Float64(dev_ns) / 1.0e6
    print("  HOST   total=", host_ms, "ms   per-call=", host_ms / Float64(ITERS), "ms")
    print("  DEVICE total=", dev_ms, "ms   per-call=", dev_ms / Float64(ITERS), "ms")
    print("  speedup=", host_ms / dev_ms, "x   saved/call=",
          (host_ms - dev_ms) / Float64(ITERS), "ms")
    # 400 projections/pass (10 per block × 40 blocks); fwd + bwd recompute = 800
    print("  EXTRAPOLATED per step: 400 proj/pass, fwd+recompute = 800 calls ->",
          " host=", (host_ms / Float64(ITERS)) * 800.0 / 1000.0, "s   device=",
          (dev_ms / Float64(ITERS)) * 800.0 / 1000.0, "s")

    # ── bit comparison of the two outputs ──
    var hh = y_host.to_host(ctx)
    var hd = y_dev.to_host(ctx)
    var bad = 0
    var maxdiff = Float32(0.0)
    if len(hh) != len(hd):
        bad = -1
    else:
        for i in range(len(hh)):
            if hh[i] != hd[i]:
                bad += 1
                var d = hh[i] - hd[i]
                if d < Float32(0.0):
                    d = -d
                if d > maxdiff:
                    maxdiff = d
    print("  BIT host-vs-device: n_mismatch=", bad, "/", len(hh),
          "  max_abs_diff=", maxdiff)
    if bad == 0:
        print("  => BIT-EQUAL: the device forward is a drop-in at this scale.")
    else:
        print("  => NOT bit-equal at this scale (rounding of the scale multiply).")


def main() raises:
    var ctx = DeviceContext()
    print("==== wan LoRA forward: HOST (klein_lora_fwd) vs DEVICE (resident, unfused) ====")
    _run(Float32(1.0), ctx)     # the shipped config (alpha 16 / rank 16)
    _run(Float32(0.5), ctx)     # scale != 1 — where rounding could differ
