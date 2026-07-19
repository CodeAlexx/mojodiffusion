# conv2d_mixed_dtype_smoke.mojo -- F32 activations with BF16/F16 conv weights.
#
# Guards the frozen-checkpoint path used by SDXL ResBlock convs: F32 activation
# tensors, BF16/F16 checkpoint filters and biases, F32 forward/d_x, and
# parameter grads returned in the original checkpoint dtype.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run -I . serenitymojo/ops/parity/conv2d_mixed_dtype_smoke.mojo

from std.collections import List, Optional
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.conv2d_backward import conv2d_backward


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _fill(n: Int, scale: Float32, offset: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32((i * 7) % 17 - 8) * scale + offset)
    return out^


def _all_finite(t: Tensor, ctx: DeviceContext, label: String) raises:
    var h = t.to_host(ctx)
    for i in range(len(h)):
        if h[i] != h[i]:
            raise Error(String("conv2d_mixed_dtype_smoke nonfinite ") + label)


def _run(dtype: STDtype, label: String, ctx: DeviceContext) raises:
    comptime N = 1
    comptime Hi = 5
    comptime Wi = 4
    comptime Cin = 3
    comptime Kh = 3
    comptime Kw = 3
    comptime Cout = 2
    comptime SH = 1
    comptime SW = 1
    comptime PH = 1
    comptime PW = 1
    comptime Ho = (Hi + 2 * PH - Kh) // SH + 1
    comptime Wo = (Wi + 2 * PW - Kw) // SW + 1

    var x = Tensor.from_host(_fill(N * Hi * Wi * Cin, Float32(0.01), Float32(0.1)), _shape4(N, Hi, Wi, Cin), STDtype.F32, ctx)
    var w_f32 = Tensor.from_host(_fill(Kh * Kw * Cin * Cout, Float32(0.02), Float32(-0.03)), _shape4(Kh, Kw, Cin, Cout), STDtype.F32, ctx)
    var b_f32 = Tensor.from_host(_fill(Cout, Float32(0.03), Float32(0.05)), _shape1(Cout), STDtype.F32, ctx)
    var gy = Tensor.from_host(_fill(N * Ho * Wo * Cout, Float32(0.015), Float32(-0.02)), _shape4(N, Ho, Wo, Cout), STDtype.F32, ctx)

    var w = cast_tensor(w_f32, dtype, ctx)
    var b = cast_tensor(b_f32, dtype, ctx)
    var y = conv2d[N, Hi, Wi, Cin, Kh, Kw, Cout, SH, SW, PH, PW](
        x, w.clone(ctx), Optional[Tensor](b.clone(ctx)), ctx
    )
    if y.dtype() != STDtype.F32:
        raise Error(String("conv2d_mixed_dtype_smoke forward dtype changed for ") + label)
    _all_finite(y, ctx, String("y ") + label)

    var g = conv2d_backward[N, Hi, Wi, Cin, Kh, Kw, Cout, SH, SW, PH, PW](
        x, w, gy, ctx
    )
    if g.d_x.dtype() != STDtype.F32:
        raise Error(String("conv2d_mixed_dtype_smoke d_x dtype changed for ") + label)
    if g.d_w.dtype() != dtype or g.d_b.dtype() != dtype:
        raise Error(String("conv2d_mixed_dtype_smoke param grad dtype changed for ") + label)
    _all_finite(g.d_x, ctx, String("d_x ") + label)
    _all_finite(g.d_w, ctx, String("d_w ") + label)
    _all_finite(g.d_b, ctx, String("d_b ") + label)


def main() raises:
    print("=== conv2d mixed dtype smoke ===")
    var ctx = DeviceContext()
    _run(STDtype.BF16, String("BF16"), ctx)
    _run(STDtype.F16, String("F16"), ctx)
    print("conv2d_mixed_dtype_smoke PASS")
