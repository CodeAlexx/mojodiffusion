# group_norm_mixed_bwd_smoke.mojo -- F32 activations with BF16/F16 GN weights.
#
# Guards the frozen-checkpoint path used by SDXL ResBlocks/final GN: F32
# activations and upstream grads, BF16/F16 checkpoint norm weights, F32 d_x, and
# parameter grads returned in the original norm-weight dtype.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/group_norm_mixed_bwd_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm_backward import group_norm_backward


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _fill(n: Int, scale: Float32, offset: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32((i % 11) - 5) * scale + offset)
    return out^


def _all_finite(t: Tensor, ctx: DeviceContext, label: String) raises:
    var h = t.to_host(ctx)
    for i in range(len(h)):
        if h[i] != h[i]:
            raise Error(String("group_norm_mixed_bwd_smoke nonfinite ") + label)


def _run(dtype: STDtype, label: String, ctx: DeviceContext) raises:
    comptime N = 1
    comptime H = 2
    comptime W = 3
    comptime C = 4
    comptime G = 2
    var x = Tensor.from_host(_fill(N * H * W * C, Float32(0.01), Float32(0.1)), _shape4(N, H, W, C), STDtype.F32, ctx)
    var go = Tensor.from_host(_fill(N * H * W * C, Float32(0.02), Float32(-0.05)), _shape4(N, H, W, C), STDtype.F32, ctx)
    var weight_f32 = Tensor.from_host(_fill(C, Float32(0.03), Float32(1.0)), _shape1(C), STDtype.F32, ctx)
    var weight = cast_tensor(weight_f32, dtype, ctx)
    var g = group_norm_backward(go, x, weight, G, Float32(1.0e-5), ctx)
    if g.d_x.dtype() != STDtype.F32:
        raise Error(String("group_norm_mixed_bwd_smoke d_x dtype changed for ") + label)
    if g.d_g.dtype() != dtype or g.d_b.dtype() != dtype:
        raise Error(String("group_norm_mixed_bwd_smoke param grad dtype changed for ") + label)
    _all_finite(g.d_x, ctx, String("d_x ") + label)
    _all_finite(g.d_g, ctx, String("d_g ") + label)
    _all_finite(g.d_b, ctx, String("d_b ") + label)


def main() raises:
    print("=== group norm mixed backward smoke ===")
    var ctx = DeviceContext()
    _run(STDtype.BF16, String("BF16"), ctx)
    _run(STDtype.F16, String("F16"), ctx)
    print("group_norm_mixed_bwd_smoke PASS")
