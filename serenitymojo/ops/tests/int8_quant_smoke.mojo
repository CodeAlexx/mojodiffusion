# ops/tests/int8_quant_smoke.mojo — verify int8_quant.mojo compiles + round-trips
# within int8 precision (per-row absmax symmetric, == reference trainer quantize_int8_axiswise).
#
# Build:
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . \
#     serenitymojo/ops/tests/int8_quant_smoke.mojo -o /tmp/int8_quant_smoke
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, pi
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.int8_quant import (
    int8_dequant_perrow_to_bf16,
    int8_dequant_perrow_to_bf16_into,
    int8_encode_perrow,
    int8_rowscale,
)


def _gaussian(n: Int, seed: Int) -> List[Float32]:
    # cheap LCG Box-Muller (deterministic; no Math.random in Mojo).
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 12345)
    for _i in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64(st >> 11) + 1.0) / Float64(1 << 53)
        st = st * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(st >> 11) / Float64(1 << 53)
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(2.0 * pi * u2)))
    return out^


def main() raises:
    var ctx = DeviceContext()
    comptime rows = 64
    comptime cols = 256
    var h = _gaussian(rows * cols, 7)
    # force a known absmax on row 0 so we can check the scale exactly
    h[0] = Float32(2.54)   # absmax → scale 2.54/127 = 0.02
    var w = Tensor.from_host(h.copy(), [rows, cols], STDtype.BF16, ctx)

    var scale = int8_rowscale(w, ctx)
    var q = int8_encode_perrow(w, scale, ctx)
    if q.dtype() != STDtype.I8:
        print("FAIL: q dtype", q.dtype().name()); return

    var w_h = w.to_host(ctx)       # bf16-rounded reference (upcast F32)
    var s_h = scale.to_host(ctx)
    # read int8 bytes raw (to_host has no I8 path) and decode two's-complement.
    var nb = q.numel()
    var qhost = ctx.enqueue_create_host_buffer[DType.uint8](nb)
    ctx.enqueue_copy(dst_buf=qhost, src_buf=q.buf)
    ctx.synchronize()
    var q_h = List[Float32]()
    var qp = qhost.unsafe_ptr()
    for i in range(nb):
        var b = Int(qp[i])
        var sv = b if b < 128 else b - 256
        q_h.append(Float32(sv))

    # Correct int8 metrics: (a) per-element ABSOLUTE error <= one quant step
    # (scale[row]); tiny elements rounding to 0 is expected (that's int8). (b)
    # cosine similarity of the whole dequant vs original — the real fidelity gate.
    var max_abs_over_step: Float32 = 0.0
    var dot: Float64 = 0.0
    var nw: Float64 = 0.0
    var ng: Float64 = 0.0
    for i in range(rows * cols):
        var r = i // cols
        var got = q_h[i] * s_h[r]
        var want = w_h[i]
        var d = got - want
        var ad = d if d >= 0 else -d
        var over = ad / s_h[r]                 # error in units of quant steps
        if over > max_abs_over_step:
            max_abs_over_step = over
        dot += Float64(got) * Float64(want)
        nw += Float64(want) * Float64(want)
        ng += Float64(got) * Float64(got)
    var cos = dot / (sqrt(nw) * sqrt(ng) + 1e-30)
    print("int8 round-trip [64x256]: max_abs_err/step =", max_abs_over_step,
          "  cosine =", cos)
    print("row0 scale =", s_h[0], " (expect ~0.020)")
    # each element must be within ~one step (0.5 rounding + bf16 slack), and the
    # whole-tensor cosine must be ~1.0 — the int8 W8A8 fidelity contract.
    if max_abs_over_step > Float32(1.0):
        print("FAIL: element exceeds one quant step", max_abs_over_step); return
    if cos < 0.999:
        print("FAIL: cosine too low", cos); return

    # ── GPU dequant (decode half, 2026-08-04) vs the host decode above:
    # BIT-EXACT. Kernel computes bf16(Float32(q) * scale[row]); the host
    # replays the identical F32 multiply + RNE bf16 cast, so any mismatch is
    # a kernel bug (sign extension, row indexing), not precision. ──
    var deq = int8_dequant_perrow_to_bf16(q, scale, ctx)
    if deq.dtype() != STDtype.BF16:
        print("FAIL: deq dtype", deq.dtype().name()); return
    var deq_h = deq.to_host(ctx)                    # exact F32 upcast of bf16
    var n_mismatch = 0
    for i in range(rows * cols):
        var r = i // cols
        var exp = (q_h[i] * s_h[r]).cast[DType.bfloat16]().cast[DType.float32]()
        if deq_h[i] != exp:
            n_mismatch += 1
    if n_mismatch > 0:
        print("FAIL: GPU dequant != host decode at", n_mismatch, "elements"); return
    # `_into` variant (caller-owned dst, the resident-store hot path): same bytes.
    var dst_buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * 2)
    var dst_shape: List[Int] = [rows, cols]
    var dst = Tensor(dst_buf^, dst_shape^, STDtype.BF16)
    int8_dequant_perrow_to_bf16_into(q, scale, dst, ctx)
    var dst_h = dst.to_host(ctx)
    var n_into_mismatch = 0
    for i in range(rows * cols):
        if dst_h[i] != deq_h[i]:
            n_into_mismatch += 1
    if n_into_mismatch > 0:
        print("FAIL: _into dequant != allocating dequant at", n_into_mismatch,
              "elements"); return
    print("PASS: int8 quantizer compiles + round-trips (cos>=0.999, <=1 step)"
          " + dequant kernels bit-exact vs host decode (both wrappers)")
