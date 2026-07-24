# ops/tests/leaky_relu_parity.mojo — unit gate for the leaky_relu forward op
# (Real-ESRGAN RRDBNet/SRVGGNetCompact use negative_slope=0.2). Checks the
# device kernel against a CPU reference for f32/bf16/f16 across positive AND
# negative inputs (the negative branch must scale by slope, not clamp to 0).
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/ops/tests/leaky_relu_parity.mojo -o /tmp/leaky_relu_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/leaky_relu_parity

from std.math import sin, cos
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.activations import leaky_relu


def _fill(n: Int, phase: Float32) -> List[Float32]:
    # Non-degenerate, symmetric around 0 so ~half the lanes hit the negative
    # (slope) branch and half hit the identity branch.
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(Float32(0.7) * Float32(i) + phase)
                   + Float32(0.4) * cos(Float32(1.9) * Float32(i)))
    return out^


def _ref_leaky(x: List[Float32], slope: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(x)):
        var v = x[i]
        out.append(v if v >= Float32(0.0) else slope * v)
    return out^


def _run(dt: STDtype, tag: String, tol: Float32, ctx: DeviceContext, mut allok: Bool) raises:
    var N = 4096
    comptime SLOPE = Float32(0.2)
    var xh = _fill(N, Float32(0.31))
    var x = Tensor.from_host(xh, [N], dt, ctx)
    var got = leaky_relu(x, ctx, SLOPE).to_host(ctx)
    var expected = _ref_leaky(xh, SLOPE)

    var max_abs = Float32(0.0)
    var neg_seen = False
    var nonzero = False
    for i in range(N):
        var d = abs(got[i] - expected[i])
        if d > max_abs:
            max_abs = d
        if xh[i] < Float32(0.0) and got[i] != Float32(0.0):
            neg_seen = True   # confirms negatives are scaled, not clamped to 0
        if got[i] != Float32(0.0):
            nonzero = True
    var ok = (max_abs <= tol) and neg_seen and nonzero
    if not ok:
        allok = False
    print("  ", "PASS" if ok else "FAIL", tag,
          " max_abs=", max_abs, " tol=", tol,
          " neg_scaled=", neg_seen, " nonzero=", nonzero)


def main() raises:
    var ctx = DeviceContext()
    var allok = True
    _run(STDtype.F32, "leaky_relu.f32", Float32(0.0), ctx, allok)
    _run(STDtype.BF16, "leaky_relu.bf16", Float32(8e-3), ctx, allok)
    _run(STDtype.F16, "leaky_relu.f16", Float32(2e-3), ctx, allok)
    print("RESULT:", "PASS" if allok else "FAIL")
