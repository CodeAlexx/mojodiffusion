# serenitymojo/models/krea2/parity/krea2_b2_dxleak_gate.mojo
#
# CONVICTION test (team-lead b2-block-diff): the op-by-op probe showed the b2 block
# backward LEAKS nonzero values into the PAD rows [real_len:L] of every LoRA-linear
# d_x (d_sw/d_xm2/d_gated/d_xm), where b1 stays EXACTLY 0. Both call the SAME per-row
# linear_backward_dx (d_x = grad_y @ W) — a zero grad_y row MUST give a zero d_x row.
# So the only way b2 leaks is if linear_backward_dx does NOT fully overwrite its
# (uninitialized _new_f32) output at M=2L — the matmul leaves stale/garbage in rows
# its tiling skips. This isolates that: zero the pad rows of grad_y, run M=L vs M=2L,
# and check whether the pad rows of d_x come back ZERO.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.tensor_algebra import concat, slice, mul_scalar

comptime IN = 6144
comptime OUT = 6144
comptime L = 1408
comptime RL = 1198          # real_len (valid prefix); rows [RL:L] are pad → zeroed


def _r(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return randn(shape^, seed, STDtype.BF16, ctx)
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


# zero rows [RL:L] of a [1,L,C] bf16 tensor: keep valid prefix, concat an exact-zero pad.
def _zero_pad_rows(t: Tensor, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var valid = slice(t, 1, 0, RL, ctx)
    var padz = mul_scalar(randn(_s3(1, L - RL, cols), 999, STDtype.BF16, ctx), Float32(0.0), ctx)
    return concat(1, ctx, valid, padz)


def _pad_norm(t: Tensor, cols: Int, ctx: DeviceContext) raises -> Float64:
    # norm of rows [RL:L] of a [1,L,cols] (or [L,cols]) tensor.
    var h = t.to_host(ctx)
    var s = Float64(0.0)
    for r in range(RL, L):
        var b = r * cols
        for c in range(cols):
            var v = Float64(h[b + c])
            s += v * v
    return sqrt(s)


def _valid_norm(t: Tensor, cols: Int, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var s = Float64(0.0)
    for r in range(0, RL):
        var b = r * cols
        for c in range(cols):
            var v = Float64(h[b + c])
            s += v * v
    return sqrt(s)


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_b2_dxleak_gate (linear_backward_dx zero-pad-row leak: M=L vs M=2L) ====")
    print("  IN=", IN, " OUT=", OUT, " L=", L, " RL=", RL)

    var w = _r(_s2(OUT, IN), 1, ctx)                 # [OUT,IN] bf16
    var dy0 = _r(_s3(1, L, OUT), 2, ctx)
    var dy0z = _zero_pad_rows(dy0, OUT, ctx)         # [1,L,OUT] pad rows zeroed

    # ── M = L (single) ──────────────────────────────────────────────────────────
    var dxL = linear_backward_dx(dy0z, w, L, IN, OUT, ctx)   # [L,IN]
    print("")
    print("  M=L   grad_y pad-rows-norm =", _pad_norm(dy0z, OUT, ctx),
          "  ->  d_x pad-rows-norm =", _pad_norm(dxL, IN, ctx),
          "   (valid d_x norm =", _valid_norm(dxL, IN, ctx), ")")

    # ── M = 2L (row-stacked, EXACT b2 pattern: [sample0 | sample1], both pad-zeroed) ─
    var dy2 = concat(1, ctx, dy0z, dy0z)             # [1,2L,OUT]
    var dx2 = linear_backward_dx(dy2, w, 2 * L, IN, OUT, ctx)  # [2L,IN]
    var dx2_h0 = slice(dx2, 0, 0, L, ctx)            # sample-0 [L,IN]
    print("  M=2L  grad_y[0:L] pad-rows-norm =", _pad_norm(dy0z, OUT, ctx),
          "  ->  d_x[0:L] pad-rows-norm =", _pad_norm(dx2_h0, IN, ctx),
          "   (valid d_x[0:L] norm =", _valid_norm(dx2_h0, IN, ctx), ")")

    print("")
    var padL = _pad_norm(dxL, IN, ctx)
    var pad2 = _pad_norm(dx2_h0, IN, ctx)
    if pad2 > Float64(1.0e-9) and padL <= Float64(1.0e-9):
        print("  CONVICTED: linear_backward_dx LEAKS at M=2L — zero grad_y rows produce",
              "NONZERO d_x rows (uninitialized-output not fully overwritten). M=L is clean.")
    elif pad2 <= Float64(1.0e-9) and padL <= Float64(1.0e-9):
        print("  CLEAN: both M=L and M=2L zero the pad d_x rows → leak is NOT in linear_backward_dx.")
    else:
        print("  M=L padnorm=", padL, " M=2L padnorm=", pad2, " — unexpected; inspect.")
