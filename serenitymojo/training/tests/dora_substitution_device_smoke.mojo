# tests/dora_substitution_device_smoke.mojo -- GPU direct DoRA substitution gate.
#
# Build/run:
#   cd /home/alex/mojodiffusion && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/dora_substitution_device_smoke.mojo -o /tmp/dora_device_smoke \
#   && /tmp/dora_device_smoke

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.dora_adapter import (
    DoRAAdapter, new_dora_adapter,
    dora_substitution_forward, dora_substitution_backward,
)
from serenitymojo.training.dora_substitution_device import (
    dora_device_from_host, dora_substitution_forward_device,
    dora_substitution_backward_device,
)


comptime COS_BAR = 0.999999
comptime BF16_COS_BAR = 0.99999
comptime NREL_BAR = 3.0e-4
comptime BF16_NREL_BAR = 8.0e-3


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale + bias)
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: length mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("nrel: length mismatch")
    var d = 0.0
    var n = 0.0
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _check(
    label: String, got: List[Float32], expected: List[Float32],
    nrel_bar: Float64 = NREL_BAR, cos_bar: Float64 = COS_BAR,
) raises:
    var c = _cos(got, expected)
    var n = _nrel(got, expected)
    print("  ", label, " cos=", c, " nrel=", n)
    if c < cos_bar or n > nrel_bar:
        raise Error(String("FAIL: ") + label)


def _case(ctx: DeviceContext, label: String, wd_on_out: Bool, dtype: STDtype) raises:
    var IN = 12
    var OUT = 16
    var R = 4
    var M = 5
    var alpha = Float32(2.0)
    var mlen = OUT if wd_on_out else IN

    var A = _randn(R * IN, 11, 0.3, 0.0)
    var B = _randn(OUT * R, 22, 0.3, 0.0)
    var mag = _randn(mlen, 33, 0.4, 1.0)
    var W = _randn(OUT * IN, 44, 0.5, 0.0)
    var x = _randn(M * IN, 100, 1.0, 0.0)
    var d_y = _randn(M * OUT, 200, 0.5, 0.0)
    var d = DoRAAdapter(
        A.copy(), B.copy(), mag.copy(), R, IN, OUT, alpha, Float32(0.0),
        _zeros(R * IN), _zeros(R * IN),
        _zeros(OUT * R), _zeros(OUT * R),
        _zeros(mlen), _zeros(mlen), wd_on_out,
    )

    print("[dora-device] ", label, " dtype=", dtype.name())
    var dev = dora_device_from_host(d, ctx)
    var x_dev = Tensor.from_host(x.copy(), [M, IN], dtype, ctx)
    var w_dev = Tensor.from_host(W.copy(), [OUT, IN], dtype, ctx)
    var y = dora_substitution_forward_device(x_dev, w_dev, dev, ctx)
    var y_ref = dora_substitution_forward(x.copy(), W.copy(), d, M)
    var bar = BF16_NREL_BAR if dtype != STDtype.F32 else NREL_BAR
    var cbar = BF16_COS_BAR if dtype != STDtype.F32 else COS_BAR
    _check(label + String(" forward"), y.to_host(ctx), y_ref, bar, cbar)

    var dev_b = dora_device_from_host(d, ctx)
    var x_dev_b = Tensor.from_host(x.copy(), [M, IN], dtype, ctx)
    var w_dev_b = Tensor.from_host(W.copy(), [OUT, IN], dtype, ctx)
    var dy_dev = Tensor.from_host(d_y.copy(), [M, OUT], dtype, ctx)
    var g = dora_substitution_backward_device(dy_dev, x_dev_b, w_dev_b, dev_b, ctx)
    var g_ref = dora_substitution_backward(d_y.copy(), x.copy(), W.copy(), d, M)
    _check(label + String(" d_A"), g.d_a.to_host(ctx), g_ref.d_a, bar, cbar)
    _check(label + String(" d_B"), g.d_b.to_host(ctx), g_ref.d_b, bar, cbar)
    _check(label + String(" d_m"), g.d_m.to_host(ctx), g_ref.d_m, bar, cbar)
    _check(label + String(" d_x"), g.d_x.to_host(ctx), g_ref.d_x, bar, cbar)


def _case_delta_zero_bf16_init(ctx: DeviceContext) raises:
    var IN = 12
    var OUT = 16
    var R = 4
    var M = 5
    var alpha = Float32(2.0)
    var W = _randn(OUT * IN, 44, 0.5, 0.0)
    var x = _randn(M * IN, 100, 1.0, 0.0)
    var d = new_dora_adapter(
        W.copy(), IN, OUT, R, alpha, UInt64(123), Float32(1.0e-7), False,
    )

    print("[dora-device] OneTrainer BF16 delta-zero init shortcut")
    var dev = dora_device_from_host(d, ctx)
    var x_dev = Tensor.from_host(x.copy(), [M, IN], STDtype.BF16, ctx)
    var w_dev = Tensor.from_host(W.copy(), [OUT, IN], STDtype.BF16, ctx)
    var y = dora_substitution_forward_device(x_dev, w_dev, dev, ctx)
    var y_ref = dora_substitution_forward(x.copy(), W.copy(), d, M)
    _check(
        String("OneTrainer BF16 delta-zero init forward"),
        y.to_host(ctx), y_ref, BF16_NREL_BAR, BF16_COS_BAR,
    )

    var dev_mixed = dora_device_from_host(d, ctx)
    var x_dev_mixed = Tensor.from_host(x.copy(), [M, IN], STDtype.BF16, ctx)
    var w_dev_mixed = Tensor.from_host(W.copy(), [OUT, IN], STDtype.F32, ctx)
    var y_mixed = dora_substitution_forward_device(x_dev_mixed, w_dev_mixed, dev_mixed, ctx)
    _check(
        String("OneTrainer BF16/F32 delta-zero init forward"),
        y_mixed.to_host(ctx), y_ref, BF16_NREL_BAR, BF16_COS_BAR,
    )

    var dev_f32 = dora_device_from_host(d, ctx)
    var x_dev_f32 = Tensor.from_host(x.copy(), [M, IN], STDtype.F32, ctx)
    var w_dev_f32 = Tensor.from_host(W.copy(), [OUT, IN], STDtype.BF16, ctx)
    var y_f32 = dora_substitution_forward_device(x_dev_f32, w_dev_f32, dev_f32, ctx)
    _check(
        String("OneTrainer F32/BF16 delta-zero init forward"),
        y_f32.to_host(ctx), y_ref, BF16_NREL_BAR, BF16_COS_BAR,
    )


def main() raises:
    var ctx = DeviceContext()
    _case(ctx, String("OneTrainer per-input magnitude"), False, STDtype.F32)
    _case(ctx, String("lycoris per-output magnitude"), True, STDtype.F32)
    _case(ctx, String("OneTrainer BF16 boundary"), False, STDtype.BF16)
    _case_delta_zero_bf16_init(ctx)
    print("PASS -- device DoRA direct substitution matches host reference")
