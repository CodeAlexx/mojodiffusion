# serenitymojo/models/krea2/parity/krea2_b2_gemmshape_gate.mojo
#
# PROOF test for the row-stacking root cause: does a linear (fwd) and linear_backward_dx
# (bwd) on [1, 2L, .] produce a DIFFERENT [0:L] slice than the same op on [1, L, .]
# (same values in [0:L])? d_x math is per-row (reduces over out/in, not M), so a
# difference == the bf16 GEMM tiles the M dimension differently for M=L vs M=2L =
# SHAPE-DETERMINISTIC rounding, which is the whole story if the deltas are ~1 bf16 eps.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.tensor_algebra import concat, slice

comptime IN = 6144
comptime OUT = 6144
comptime L = 1408


def _r(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _stats(a: List[Float32], b: List[Float32]) -> Tuple[Float64, Float64]:
    var n = len(a)
    var dot = Float64(0.0); var na = Float64(0.0); var nb = Float64(0.0); var mad = Float64(0.0)
    for i in range(n):
        dot += Float64(a[i]) * Float64(b[i]); na += Float64(a[i]) * Float64(a[i]); nb += Float64(b[i]) * Float64(b[i])
        var dd = Float64(a[i]) - Float64(b[i])
        if dd < Float64(0.0):
            dd = -dd
        if dd > mad:
            mad = dd
    var c = Float64(0.0)
    if na > Float64(0.0) and nb > Float64(0.0):
        c = dot / (sqrt(na) * sqrt(nb))
    return (c, mad)


def _rows(t: Tensor, r0: Int, r1: Int, cols: Int, ctx: DeviceContext) raises -> List[Float32]:
    var h = t.to_host(ctx)
    var out = List[Float32]()
    for r in range(r0, r1):
        for c in range(cols):
            out.append(h[r * cols + c])
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_b2_gemmshape_gate (linear fwd + linear_backward_dx: [1,L] vs [1,2L][0:L]) ====")
    print("  IN=", IN, " OUT=", OUT, " L=", L)
    var w = _r(_s2(OUT, IN), 1, ctx)
    var x0 = _r(_s3(1, L, IN), 2, ctx)
    var garb = _r(_s3(1, L, IN), 3, ctx)
    var dy0 = _r(_s3(1, L, OUT), 4, ctx)
    var garb_y = _r(_s3(1, L, OUT), 5, ctx)

    # ── FORWARD linear ──────────────────────────────────────────────────────────
    var nb1 = Optional[Tensor](None)
    var y_std = linear(x0, w, nb1^, ctx)                       # [1,L,OUT]
    var x2 = concat(1, ctx, x0, garb)                          # [1,2L,IN]
    var nb2 = Optional[Tensor](None)
    var y_bat = linear(x2, w, nb2^, ctx)                       # [1,2L,OUT]
    var fr = _stats(_rows(y_bat, 0, L, OUT, ctx), _rows(y_std, 0, L, OUT, ctx))
    print("")
    print("  FORWARD  linear [1,2L][0:L] vs [1,L]:   cos =", fr[0], "   max_abs =", fr[1])

    # ── BACKWARD linear_backward_dx (d_x math is per-row → M-independent in exact math) ─
    var dx_std = linear_backward_dx(dy0, w, L, IN, OUT, ctx)   # [L, IN]
    var dy2 = concat(1, ctx, dy0, garb_y)                      # [1,2L,OUT]
    var dx_bat = linear_backward_dx(dy2, w, 2 * L, IN, OUT, ctx)  # [2L, IN]
    var br = _stats(_rows(dx_bat, 0, L, IN, ctx), _rows(dx_std, 0, L, IN, ctx))
    print("  BACKWARD linear_bwd_dx [2L][0:L] vs [L]: cos =", br[0], "   max_abs =", br[1])

    print("")
    if fr[1] > Float64(0.0) or br[1] > Float64(0.0):
        print("  PROVEN: batched-M GEMM rounds the [0:L] rows differently than single-M (shape-deterministic bf16).")
    else:
        print("  BIT-IDENTICAL: GEMM is M-invariant → the row-stacking-GEMM story is WRONG, look elsewhere.")
