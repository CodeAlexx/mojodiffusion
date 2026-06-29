# tests/oft_onetrainer_device_smoke.mojo -- GPU OneTrainer-OFT block-size-4 gate.
#
# Build/run:
#   cd /home/alex/mojodiffusion && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/oft_onetrainer_device_smoke.mojo -o /tmp/oft_device_smoke \
#   && /tmp/oft_device_smoke

from std.collections import List
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.training.oft_onetrainer_device import (
    oft_ot_rotate_b4, oft_ot_rotate_backward_b4,
)


comptime COS_BAR = 0.99999
comptime NREL_BAR = 1.0e-5
comptime BF16_NREL_BAR = 6.0e-3


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _identity(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        for j in range(n):
            if i == j:
                out.append(Float32(1.0))
            else:
                out.append(Float32(0.0))
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


def _require_close(label: String, got: List[Float32], expected: List[Float32], nrel_bar: Float64) raises:
    var c = _cos(got, expected)
    var nr = _nrel(got, expected)
    print("  ", label, " cos=", c, " nrel=", nr)
    if c < COS_BAR or nr > nrel_bar:
        raise Error(String("FAIL: ") + label)


def main() raises:
    var ctx = DeviceContext()
    var IN = 12
    var M = 5
    var b = 4
    var r = IN // b
    var ne = b * (b - 1) // 2
    print("[oft-device-smoke] IN=", IN, " M=", M, " b=", b, " r=", r)

    var vec = _fill(r * ne, 11, 0.25)
    var x = _fill(M * IN, 33, 1.0)
    var d_x_rot = _fill(M * IN, 44, 0.5)
    var W = _identity(IN)

    var x_dev = Tensor.from_host(x.copy(), [M, IN], STDtype.F32, ctx)
    var vec_dev = Tensor.from_host(vec.copy(), [r, ne], STDtype.F32, ctx)
    var rot = oft_ot_rotate_b4(x_dev, vec_dev, ctx)
    var rot_h = rot.to_host(ctx)
    var rot_ref = oft_ot_forward(x.copy(), vec.copy(), W.copy(), M, IN, IN, b, r)
    _require_close(String("f32 forward rotate"), rot_h, rot_ref, NREL_BAR)

    var dxr_dev = Tensor.from_host(d_x_rot.copy(), [M, IN], STDtype.F32, ctx)
    var grads = oft_ot_rotate_backward_b4(dxr_dev, Tensor.from_host(x.copy(), [M, IN], STDtype.F32, ctx), vec_dev, ctx)
    var host_grads = oft_ot_backward(d_x_rot.copy(), x.copy(), vec.copy(), W.copy(), M, IN, IN, b, r)
    _require_close(String("f32 backward d_x"), grads.d_x.to_host(ctx), host_grads.d_x, NREL_BAR)
    _require_close(String("f32 backward d_vec"), grads.d_vec.to_host(ctx), host_grads.d_vec, NREL_BAR)

    var x_bf16 = Tensor.from_host(x.copy(), [M, IN], STDtype.BF16, ctx)
    var rot_bf16 = oft_ot_rotate_b4(x_bf16, vec_dev, ctx)
    if rot_bf16.dtype() != STDtype.BF16:
        raise Error("FAIL: BF16 rotation did not preserve storage dtype")
    _require_close(String("bf16 forward rotate"), rot_bf16.to_host(ctx), rot_ref, BF16_NREL_BAR)

    print("PASS -- device OneTrainer-OFT block-size-4 forward/backward matches host oracle")
