# linear_mixed_bias_smoke.mojo -- F32 activations with BF16/F16 checkpoint weights.
#
# This guards the narrow mixed-storage path used by SDXL and other trainers:
# F32 activations/grads, BF16/F16 frozen checkpoint weight+bias storage, F32
# output and returned grads. It is a runtime smoke, not a torch oracle.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/linear_mixed_bias_smoke.mojo

from std.collections import List, Optional
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _fill(n: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var v = Float32((i % 17) - 8) * scale
        out.append(v)
    return out^


def _all_finite(t: Tensor, ctx: DeviceContext, label: String) raises:
    var h = t.to_host(ctx)
    for i in range(len(h)):
        if h[i] != h[i]:
            raise Error(String("linear_mixed_bias_smoke nonfinite ") + label)


def _run(dtype: STDtype, label: String, ctx: DeviceContext) raises:
    comptime M = 3
    comptime IN = 4
    comptime OUT = 5
    var x = Tensor.from_host(_fill(M * IN, Float32(0.01)), _shape2(M, IN), STDtype.F32, ctx)
    var w_f32 = Tensor.from_host(_fill(OUT * IN, Float32(0.02)), _shape2(OUT, IN), STDtype.F32, ctx)
    var b_f32 = Tensor.from_host(_fill(OUT, Float32(0.03)), _shape1(OUT), STDtype.F32, ctx)
    var w = cast_tensor(w_f32, dtype, ctx)
    var b = cast_tensor(b_f32, dtype, ctx)
    if w.dtype() != dtype or b.dtype() != dtype:
        raise Error(String("linear_mixed_bias_smoke failed to create ") + label + String(" storage"))

    var y = linear(x, w, Optional[Tensor](b.clone(ctx)), ctx)
    if y.dtype() != STDtype.F32:
        raise Error(String("linear_mixed_bias_smoke output dtype changed for ") + label)
    _all_finite(y, ctx, String("y ") + label)

    var gy = Tensor.from_host(_fill(M * OUT, Float32(0.015)), _shape2(M, OUT), STDtype.F32, ctx)
    var g = linear_backward(gy, x, w, M, IN, OUT, ctx)
    if g.d_x.dtype() != STDtype.F32 or g.d_w.dtype() != STDtype.F32 or g.d_b.dtype() != STDtype.F32:
        raise Error(String("linear_mixed_bias_smoke grad dtype changed for ") + label)
    _all_finite(g.d_x, ctx, String("d_x ") + label)
    _all_finite(g.d_w, ctx, String("d_w ") + label)
    _all_finite(g.d_b, ctx, String("d_b ") + label)


def main() raises:
    print("=== linear mixed checkpoint-bias smoke ===")
    var ctx = DeviceContext()
    _run(STDtype.BF16, String("BF16"), ctx)
    _run(STDtype.F16, String("F16"), ctx)
    print("linear_mixed_bias_smoke PASS")
