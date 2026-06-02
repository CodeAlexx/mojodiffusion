# training/grad_clip_smoke.mojo — gate for clip_grads_by_global_norm (item 1b).
#
# Builds 5 toy F32 grads with a KNOWN global L2 norm, clips at a max_norm below
# it, then asserts:
#   (1) the returned (pre-clip) total_norm matches the hand-computed value
#       to 1e-5, and
#   (2) the post-clip recomputed global norm == max_norm to 1e-4 (clipping
#       fired because max_norm < total_norm).
# Also runs a no-op case (max_norm above the norm) and asserts the grads are
# unchanged and the returned norm still equals the true norm.
#
# Hand-computed reference:
#   g0 = [3, 4]            -> sumsq 25
#   g1 = [0, 0, 0]         -> sumsq 0     (a legitimately-zero grad; must not
#                                          break the norm or the scaling)
#   g2 = [1, 2, 2]         -> sumsq 9
#   g3 = [5]               -> sumsq 25
#   g4 = [-1, 1, -1, 1]    -> sumsq 4
#   total sumsq = 25+0+9+25+4 = 63 ; total_norm = sqrt(63) = 7.9372539...
#   clip at max_norm = 2.0 (< 7.937) -> scale = 2/7.9372539 ; post-clip norm = 2.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/grad_clip_smoke.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.optim import clip_grads_by_global_norm, TArc


def _grad(vals: List[Float32], ctx: DeviceContext) raises -> TArc:
    var shape = List[Int]()
    shape.append(len(vals))
    return TArc(Tensor.from_host(vals, shape^, STDtype.F32, ctx))


# recompute the global L2 norm of a list of grads via host readback (independent
# of the library, so this is a true cross-check).
def _global_norm(grads: List[TArc], ctx: DeviceContext) raises -> Float64:
    var ss = Float64(0.0)
    for i in range(len(grads)):
        var h = grads[i][].to_host(ctx)
        for j in range(len(h)):
            ss += Float64(h[j]) * Float64(h[j])
    return sqrt(ss)


def _build(ctx: DeviceContext) raises -> List[TArc]:
    var v0 = List[Float32](); v0.append(3.0); v0.append(4.0)
    var v1 = List[Float32](); v1.append(0.0); v1.append(0.0); v1.append(0.0)
    var v2 = List[Float32](); v2.append(1.0); v2.append(2.0); v2.append(2.0)
    var v3 = List[Float32](); v3.append(5.0)
    var v4 = List[Float32](); v4.append(-1.0); v4.append(1.0); v4.append(-1.0); v4.append(1.0)
    var g = List[TArc]()
    g.append(_grad(v0, ctx)); g.append(_grad(v1, ctx)); g.append(_grad(v2, ctx))
    g.append(_grad(v3, ctx)); g.append(_grad(v4, ctx))
    return g^


def main() raises:
    var ctx = DeviceContext()
    var ok = True

    var expected_norm = sqrt(Float64(63.0))   # 7.937253933...
    print("hand-computed total_norm = sqrt(63) =", expected_norm)

    # ── case 1: clip fires (max_norm < total_norm) ────────────────────────────
    var grads = _build(ctx)
    var max_norm = Float32(2.0)
    var returned = clip_grads_by_global_norm(grads, max_norm, ctx)
    print("case1 clip max_norm=2.0 -> returned pre-clip norm =", returned)

    if abs(Float64(returned) - expected_norm) > 1e-5:
        print("FAIL returned norm != hand-computed (|d|=", abs(Float64(returned) - expected_norm), ")"); ok = False
    else:
        print("PASS returned pre-clip norm matches hand-computed to 1e-5")

    var post = _global_norm(grads, ctx)
    print("case1 post-clip recomputed norm =", post)
    if abs(post - Float64(2.0)) > 1e-4:
        print("FAIL post-clip norm != max_norm (|d|=", abs(post - Float64(2.0)), ")"); ok = False
    else:
        print("PASS post-clip recomputed norm == max_norm (2.0) to 1e-4")

    # ── case 2: no-op (max_norm > total_norm) ─────────────────────────────────
    var grads2 = _build(ctx)
    var big = Float32(100.0)
    var returned2 = clip_grads_by_global_norm(grads2, big, ctx)
    print("case2 no-clip max_norm=100 -> returned norm =", returned2)
    if abs(Float64(returned2) - expected_norm) > 1e-5:
        print("FAIL case2 returned norm != hand-computed"); ok = False
    else:
        print("PASS case2 returned norm == hand-computed (no clip applied)")
    var post2 = _global_norm(grads2, ctx)
    if abs(post2 - expected_norm) > 1e-4:
        print("FAIL case2 grads were modified (post norm=", post2, ")"); ok = False
    else:
        print("PASS case2 grads UNCHANGED (post norm == pre norm)")

    if not ok:
        raise Error("grad_clip_smoke FAILED")
    print("grad_clip_smoke gate PASS")
