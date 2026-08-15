# sdpa_backward_dynamic_gate — the runtime-dim SDPA backward must be
# BIT-IDENTICAL to the comptime instantiation (same kernels, same layouts —
# only the dimension plumbing changed). Checked in bf16 (the H3 train dtype,
# which routes the comptime path through sdpa_backward_rect) at two shapes.
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_dynamic,
)

comptime H = 8
comptime Dh = 64
comptime S1 = 384
comptime S2 = 173  # deliberately odd runtime length


def _max_abs_diff(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("length mismatch")
    var m = Float64(0)
    for i in range(len(ah)):
        var d = Float64(ah[i]) - Float64(bh[i])
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _std(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var s = Float64(0)
    var s2 = Float64(0)
    for i in range(len(h)):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var n = Float64(len(h))
    var mean = s / n
    return (s2 / n - mean * mean) ** 0.5


def main() raises:
    var ctx = DeviceContext()
    var scale = Float32(0.125)

    # arm 1: bit-compare vs the comptime path at S=384
    var sh: List[Int] = [1, S1, H, Dh]
    var q = randn(sh.copy(), UInt64(1), STDtype.BF16, ctx)
    var k = randn(sh.copy(), UInt64(2), STDtype.BF16, ctx)
    var v = randn(sh.copy(), UInt64(3), STDtype.BF16, ctx)
    var go = randn(sh.copy(), UInt64(4), STDtype.BF16, ctx)
    var cref = sdpa_backward[1, S1, H, Dh](q, k, v, go, scale, ctx)
    var dyn = sdpa_backward_dynamic(q, k, v, go, scale, ctx)
    ctx.synchronize()
    var dq = _max_abs_diff(cref.d_q, dyn.d_q, ctx)
    var dk = _max_abs_diff(cref.d_k, dyn.d_k, ctx)
    var dv = _max_abs_diff(cref.d_v, dyn.d_v, ctx)
    print("S=384 max_abs d_q/d_k/d_v:", dq, dk, dv)
    if dq != 0.0 or dk != 0.0 or dv != 0.0:
        raise Error("sdpa_backward_dynamic: not bit-identical at S=384")

    # arm 2: odd runtime length runs and produces finite, non-degenerate grads
    var sh2: List[Int] = [1, S2, H, Dh]
    var q2 = randn(sh2.copy(), UInt64(5), STDtype.BF16, ctx)
    var k2 = randn(sh2.copy(), UInt64(6), STDtype.BF16, ctx)
    var v2 = randn(sh2.copy(), UInt64(7), STDtype.BF16, ctx)
    var go2 = randn(sh2.copy(), UInt64(8), STDtype.BF16, ctx)
    var g2 = sdpa_backward_dynamic(q2, k2, v2, go2, scale, ctx)
    ctx.synchronize()
    var s_dq = _std(g2.d_q, ctx)
    var s_dk = _std(g2.d_k, ctx)
    var s_dv = _std(g2.d_v, ctx)
    print("S=173 grad std d_q/d_k/d_v:", s_dq, s_dk, s_dv)
    if not (s_dq > 0 and s_dk > 0 and s_dv > 0):
        raise Error("sdpa_backward_dynamic: degenerate grads at S=173")

    print("PASS: sdpa_backward_dynamic bit-identical + runtime-length clean")
