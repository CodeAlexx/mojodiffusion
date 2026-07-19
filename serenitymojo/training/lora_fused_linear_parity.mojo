# training/lora_fused_linear_parity.mojo — GATE for the fused LoRA kernels.
#
# Compares the fused path (training/lora_fused_linear.mojo, via the
# lora_block.mojo dispatchers) against the legacy unfused chain
# (linear + mul_scalar + linear_backward_dx/dw) on identical device data.
#
# EXPECTATION (not bit-equality): identical products, different accumulation
# order. Two noise sources (both the accepted class — see fused-AdamW m/v
# contract): (a) pure F32 reorder ~1e-7 relative; (b) t / d_t values landing
# exactly on a bf16 RNE boundary flip by ±1 bf16 quantum when the F32 sum
# differs by 1 ulp, which propagates ~quantum-sized ABSOLUTE shifts into a few
# downstream elements (MEASURED first run: max_abs 3.9e-5 vs ref rms 1.7e-2,
# cos 0.9999999988). Elementwise max-rel is the WRONG metric here (near-zero
# denominators + legitimate quantum flips).
# BARS (MEASURED 2026-06-11, first full run):
#   d_a / d_b — consume the UNROUNDED F32 t/d_t: pure reorder noise only,
#     measured 12-13 nines → bar cosine ≥ 0.99999999 (8 nines).
#   delta / d_x — consume bf16(t)/bf16(d_t): RNE tie flips at quantization
#     boundaries add ±1-quantum shifts (measured d_x cos 0.99999998-0.9999999,
#     i.e. the tie class, NOT a math bug: d_a/d_b sharing the same buffers are
#     12+ nines) → bar cosine ≥ 0.9999999 (7 nines) at the raw-contribution
#     level. The LoRA contribution is a small additive term inside the block
#     (contribution rms «<< base rms), so the BLOCK torch gates remain the
#     8-nines oracle and MUST be re-run after wiring (see scoreboard).
#   All outputs: max_abs ≤ 1e-2 × ref_rms. mismatch_rate (any nonzero diff —
#   measured ~10% at 1-ulp, ~98% on d_a/d_b at 1e-12 relative) is info-only.
#
# Build (GPU):
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/lora_fused_linear_parity.mojo -o /tmp/lora_fused_par
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.lora_fused_linear import lora_fused_bwd, lora_fused_fwd
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_bwd_device_resident_tensors_unfused,
    klein_lora_fwd_device_resident_unfused,
)


comptime COS_BAR_F32 = 0.99999999   # unrounded-F32 outputs: d_a, d_b
comptime COS_BAR_BF16 = 0.9999999   # bf16-tie-exposed outputs: delta, d_x
comptime ABS_OVER_RMS_BAR = 1.0e-2


struct _Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_f32(mut self) -> Float32:
        self.state = (
            self.state * UInt64(6364136223846793005)
            + UInt64(1442695040888963407)
        )
        var bits = (self.state >> 33) % UInt64(2000000)
        return Float32(Int(bits)) / Float32(1.0e6) - Float32(1.0)


def _rand_list(mut rng: _Lcg, n: Int, amp: Float32) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(rng.next_f32() * amp)
    return o^


struct _Cmp(Movable):
    var cos: Float64
    var max_abs: Float64
    var ref_rms: Float64
    var mismatch_rate: Float64

    def __init__(
        out self,
        cos: Float64,
        max_abs: Float64,
        ref_rms: Float64,
        mismatch_rate: Float64,
    ):
        self.cos = cos
        self.max_abs = max_abs
        self.ref_rms = ref_rms
        self.mismatch_rate = mismatch_rate


def _compare(test: List[Float32], refv: List[Float32]) raises -> _Cmp:
    if len(test) != len(refv):
        raise Error("compare: length mismatch")
    var dot = Float64(0.0)
    var n_t = Float64(0.0)
    var n_r = Float64(0.0)
    var max_abs = Float64(0.0)
    var mismatches = 0
    for i in range(len(refv)):
        var tv = Float64(test[i])
        var rv = Float64(refv[i])
        if not (tv == tv) or not (rv == rv):
            raise Error("compare: NaN at " + String(i))
        dot += tv * rv
        n_t += tv * tv
        n_r += rv * rv
        var ad = tv - rv
        if ad < 0:
            ad = -ad
        if ad > max_abs:
            max_abs = ad
        if ad > 0:
            mismatches += 1
    if n_r == 0.0:
        raise Error("compare: reference is all-zero (degenerate)")
    var cos = dot / (sqrt(n_t) * sqrt(n_r))
    var rms = sqrt(n_r / Float64(len(refv)))
    return _Cmp(cos, max_abs, rms, Float64(mismatches) / Float64(len(refv)))


def _check(name: String, c: _Cmp, cos_bar: Float64) raises:
    print(
        "  ", name, ": cos=", c.cos, " max_abs=", c.max_abs,
        " ref_rms=", c.ref_rms, " abs/rms=", c.max_abs / c.ref_rms,
        " mismatch_rate=", c.mismatch_rate,
    )
    if c.cos < cos_bar:
        raise Error(name + ": cosine below bar")
    if c.max_abs > ABS_OVER_RMS_BAR * c.ref_rms:
        raise Error(name + ": max_abs above bar")


def _run_case(
    M: Int, in_f: Int, out_f: Int, scale: Float32,
    seed: UInt64, ctx: DeviceContext,
) raises:
    print("case M=", M, " in=", in_f, " out=", out_f, " scale=", scale)
    var rank = 16
    var rng = _Lcg(seed)
    var x_h = _rand_list(rng, M * in_f, 1.0)
    var dc_h = _rand_list(rng, M * out_f, 0.05)
    var a_h = _rand_list(rng, rank * in_f, 0.02)
    var b_h = _rand_list(rng, out_f * rank, 0.02)

    var lo = LoraAdapterDevice(
        ArcPointer[Tensor](
            Tensor.from_host(a_h.copy(), [rank, in_f], STDtype.BF16, ctx)
        ),
        ArcPointer[Tensor](
            Tensor.from_host(b_h.copy(), [out_f, rank], STDtype.BF16, ctx)
        ),
        rank, in_f, out_f, scale,
    )
    var x = Tensor.from_host(x_h.copy(), [M, in_f], STDtype.F32, ctx)
    var dc = Tensor.from_host(dc_h.copy(), [M, out_f], STDtype.F32, ctx)

    # forward: fused entry (called directly — the production dispatch switch
    # LORA_FUSED_ENABLED is currently off) vs explicit unfused chain.
    var d_fused = lora_fused_fwd(
        x, lo.a[], lo.b[], lo.rank, lo.in_f, lo.out_f, lo.scale, ctx
    )
    var d_ref = klein_lora_fwd_device_resident_unfused(x, lo, M, ctx)
    _check(
        "fwd delta", _compare(d_fused.to_host(ctx), d_ref.to_host(ctx)),
        COS_BAR_BF16,
    )

    # backward: fused entry vs explicit unfused chain.
    var g_fused = lora_fused_bwd(
        dc, x, lo.a[], lo.b[], lo.rank, lo.in_f, lo.out_f, lo.scale, ctx
    )
    var g_ref = klein_lora_bwd_device_resident_tensors_unfused(dc, x, lo, M, ctx)
    _check(
        "bwd d_a",
        _compare(g_fused.d_a[].to_host(ctx), g_ref.d_a[].to_host(ctx)),
        COS_BAR_F32,
    )
    _check(
        "bwd d_b",
        _compare(g_fused.d_b[].to_host(ctx), g_ref.d_b[].to_host(ctx)),
        COS_BAR_F32,
    )
    _check(
        "bwd d_x",
        _compare(g_fused.d_x[].to_host(ctx), g_ref.d_x[].to_host(ctx)),
        COS_BAR_BF16,
    )


def main() raises:
    comptime if not has_accelerator():
        print("lora_fused_linear_parity: GPU required")
        raise Error("no accelerator")
    else:
        var ctx = DeviceContext()
        # Klein-9B product shapes (512px training: M = 1536 rows).
        _run_case(1536, 3072, 3072, 1.0, 11, ctx)     # proj-style slot
        _run_case(1536, 3072, 9216, 0.5, 22, ctx)     # fused-qkv slot
        _run_case(1536, 3072, 12288, 2.0, 33, ctx)    # mlp-up slot
        _run_case(1536, 12288, 3072, 0.25, 44, ctx)   # mlp-down slot
        # edge shapes: M / dims not multiples of tiles.
        _run_case(37, 130, 70, 1.5, 55, ctx)
        _run_case(255, 257, 511, 0.75, 66, ctx)
        print("lora_fused_linear_parity: ALL PASS")
