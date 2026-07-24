# ops/tests/prelu_parity.mojo — unit gate for the prelu forward op (torch.nn.PReLU,
# per-channel learnable slope; SRVGGNetCompact uses it after every conv). Checks
# the device kernel against a CPU reference for f32/bf16/f16 over an NHWC buffer
# with a distinct negative slope per channel, across positive AND negative inputs
# (each channel's negatives must scale by that channel's alpha, not clamp to 0).
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/ops/tests/prelu_parity.mojo -o /tmp/prelu_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/prelu_parity

from std.math import sin, cos
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.activations import prelu


def _fill(n: Int, phase: Float32) -> List[Float32]:
    # Non-degenerate, symmetric around 0 so ~half the lanes hit the negative
    # (per-channel slope) branch and half hit the identity branch.
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(Float32(0.7) * Float32(i) + phase)
                   + Float32(0.4) * cos(Float32(1.9) * Float32(i)))
    return out^


def _alphas(C: Int) -> List[Float32]:
    # Distinct slope per channel (all != leaky's 0.2, some >1, some tiny).
    var out = List[Float32]()
    for c in range(C):
        out.append(Float32(0.05) + Float32(0.9) * Float32(c) / Float32(C))
    return out^


def _ref_prelu(x: List[Float32], a: List[Float32], C: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(x)):
        var v = x[i]
        var s = a[i % C]
        out.append(v if v >= Float32(0.0) else s * v)
    return out^


def _run(dt: STDtype, tag: String, tol: Float32, ctx: DeviceContext, mut allok: Bool) raises:
    var Hh = 8
    var Ww = 8
    var C = 16
    var N = Hh * Ww * C
    var xh = _fill(N, Float32(0.31))
    var ah = _alphas(C)
    var x = Tensor.from_host(xh, [1, Hh, Ww, C], dt, ctx)       # NHWC
    var alpha = Tensor.from_host(ah, [C], STDtype.F32, ctx)     # alpha always F32
    var got = prelu(x, alpha, ctx).to_host(ctx)
    var expected = _ref_prelu(xh, ah, C)

    var max_abs = Float32(0.0)
    var neg_seen = False
    var nonzero = False
    for i in range(N):
        var d = abs(got[i] - expected[i])
        if d > max_abs:
            max_abs = d
        if xh[i] < Float32(0.0) and got[i] != Float32(0.0):
            neg_seen = True   # confirms negatives are scaled per channel, not clamped
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
    _run(STDtype.F32, "prelu.f32", Float32(0.0), ctx, allok)
    _run(STDtype.BF16, "prelu.bf16", Float32(8e-3), ctx, allok)
    _run(STDtype.F16, "prelu.f16", Float32(2e-3), ctx, allok)
    print("RESULT:", "PASS" if allok else "FAIL")
