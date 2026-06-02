# vec_rms_norm_parity.mojo — gate the VECTORIZED RMSNorm (fwd+bwd) AGAINST the
# EXISTING SCALAR kernels (ops/norm.mojo rms_norm + ops/norm_backward.mojo
# rms_norm_backward). Reference = the scalar kernel's own GPU output (NOT a
# torch oracle): the vectorized output must match the scalar output cos>=0.999.
#
# Also a microbench: scalar-ms vs vec-ms over N reps at Klein-ish dims.
# Bitrot demo: pass BITROT=1 as an arg to corrupt the reference and prove the
# gate exits NONZERO.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/ops/parity/vec_rms_norm_parity.mojo
#  bitrot: ... vec_rms_norm_parity.mojo BITROT

from sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.vec_rms_norm import vec_rms_norm, vec_rms_norm_backward


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def main() raises:
    var bitrot = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("BITROT"):
            bitrot = True

    var ctx = DeviceContext()
    var rows = 4608     # Klein-ish token count (N_IMG+N_TXT ballpark)
    var d = 128         # Klein Dh (multiple of 4)
    var eps = Float32(1e-6)
    print("=== vec_rms_norm parity vs SCALAR (rows=", rows, " D=", d, ") ===")

    var x_h = _fill(rows * d, 11, 2.0)
    var g_h = _fill(d, 22, 1.0)
    var go_h = _fill(rows * d, 33, 1.5)

    var x = Tensor.from_host(x_h.copy(), [rows, d], STDtype.F32, ctx)
    var g = Tensor.from_host(g_h.copy(), [d], STDtype.F32, ctx)
    var go = Tensor.from_host(go_h.copy(), [rows, d], STDtype.F32, ctx)

    # ── parity: vec vs scalar (scalar output is the reference) ──
    var y_scalar = rms_norm(x, g, eps, ctx).to_host(ctx)
    var y_vec = vec_rms_norm(x, g, eps, ctx)

    var sg = rms_norm_backward(go, x, g, eps, ctx)
    var dx_scalar = sg.d_x.to_host(ctx)
    var dg_scalar = sg.d_g.to_host(ctx)
    var vg = vec_rms_norm_backward(go, x, g, eps, ctx)

    if bitrot:
        # Deliberately corrupt the reference so a CORRECT vec kernel fails the
        # gate — proves the gate actually asserts on values.
        for i in range(len(y_scalar)):
            y_scalar[i] = y_scalar[i] + 1.0

    var h = ParityHarness(0.999)
    var r_y = h.compare(y_vec, y_scalar, ctx)
    var r_dx = h.compare(vg.d_x, dx_scalar, ctx)
    var r_dg = h.compare(vg.d_g, dg_scalar, ctx)
    print("    fwd y :", r_y)
    print("    bwd dx:", r_dx)
    print("    bwd dg:", r_dg)

    # ── microbench ──
    var reps = 200
    # scalar fwd
    _ = rms_norm(x, g, eps, ctx).to_host(ctx)  # warmup + sync
    var t0 = perf_counter_ns()
    for _ in range(reps):
        var yy = rms_norm(x, g, eps, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t1 = perf_counter_ns()
    # vec fwd
    _ = vec_rms_norm(x, g, eps, ctx).to_host(ctx)
    var t2 = perf_counter_ns()
    for _ in range(reps):
        var yy = vec_rms_norm(x, g, eps, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var scal_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var vec_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench fwd] scalar=", scal_ms, "ms  vec=", vec_ms,
          "ms  speedup=", scal_ms / vec_ms, "x")

    # ── second microbench at a big feature dim (D=4096) where vec4 helps ──
    var d2 = 4096
    var rows2 = 1024
    var x2 = Tensor.from_host(_fill(rows2 * d2, 7, 2.0), [rows2, d2], STDtype.F32, ctx)
    var g2 = Tensor.from_host(_fill(d2, 9, 1.0), [d2], STDtype.F32, ctx)
    _ = rms_norm(x2, g2, eps, ctx).to_host(ctx)
    var u0 = perf_counter_ns()
    for _ in range(reps):
        var yy = rms_norm(x2, g2, eps, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var u1 = perf_counter_ns()
    _ = vec_rms_norm(x2, g2, eps, ctx).to_host(ctx)
    var u2 = perf_counter_ns()
    for _ in range(reps):
        var yy = vec_rms_norm(x2, g2, eps, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var u3 = perf_counter_ns()
    var s2 = Float64(u1 - u0) / 1.0e6 / Float64(reps)
    var v2 = Float64(u3 - u2) / 1.0e6 / Float64(reps)
    print("    [microbench fwd D=4096 rows=1024] scalar=", s2, "ms  vec=", v2,
          "ms  speedup=", s2 / v2, "x")

    if r_y.passed and r_dx.passed and r_dg.passed:
        print("PASS: vec_rms_norm matches scalar cos>=0.999 (fwd+dx+dg)")
    else:
        raise Error("vec_rms_norm_parity gate FAILED")
