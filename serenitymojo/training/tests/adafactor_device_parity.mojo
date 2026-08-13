# training/tests/adafactor_device_parity.mojo — device Adafactor vs the
# parity-gated HOST oracle (training/adafactor.mojo::adafactor_step_2d,
# torch/optim/_adafactor.py-exact).
#
# Trajectory gate (3 steps, fresh non-degenerate grads each step): the host
# oracle's p is RNE-rounded to bf16 after each step (the device carrier is
# bf16), so both walk the same discretized trajectory; residual diffs are
# reduction-order only. Gates: p cos >= 0.9999, row_var/col_var cos >= 0.99999.
# Plus the SR bound check: with sr_seed != 0, every element differs from the
# RNE result by at most one bf16 ulp.
#
# Build/run:
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . \
#     -I /home/alex/MOJO-libs -Xlinker -lm \
#     serenitymojo/training/tests/adafactor_device_parity.mojo -o /tmp/af_dev_parity
#   LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/af_dev_parity

from max.gpu.host import DeviceContext
from std.math import sqrt, sin
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.adafactor import AdafactorState, adafactor_step_2d
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device,
)

comptime ROWS = 96
comptime COLS = 64
comptime N = ROWS * COLS

comptime LR = Float64(3e-4)
comptime B2D = Float64(-0.8)
comptime EPS2 = Float64(1e-3)
comptime DD = Float64(1.0)
comptime WD = Float64(0.0)


def _bf16_rne(x: Float32) -> Float32:
    return Float32(SIMD[DType.float32, 1](x).cast[DType.bfloat16]().cast[DType.float32]()[0])


def _fill(n: Int, a: Float32, b: Float32, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(sin(a * Float32(i) + b) * c)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0; var na: Float64 = 0.0; var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def _maxabs(a: List[Float32], b: List[Float32]) -> Float32:
    var m: Float32 = 0.0
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0: d = -d
        if d > m: m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    print("==== adafactor_device_parity (device kernels vs host torch-oracle) ====")
    print("ROWS=", ROWS, " COLS=", COLS, " steps=3  lr=", LR)

    # bf16-representable start so host(F32) and device(bf16) see the same p0.
    var p_host = List[Float32]()
    var p0 = _fill(N, 0.0007, 0.11, 0.5)
    for i in range(N):
        p_host.append(_bf16_rne(p0[i]))
    var p_dev = Tensor.from_host(p_host.copy(), [ROWS, COLS], STDtype.BF16, ctx)

    var hstate = AdafactorState(ROWS, COLS)
    var dstate = AdafactorDeviceState(ROWS, COLS, ctx)

    var allok = True
    for step in range(1, 4):
        var g = _fill(N, 0.0011 * Float32(step), 0.07 + 0.3 * Float32(step), 0.05)
        var g_dev = Tensor.from_host(g.copy(), [ROWS, COLS], STDtype.F32, ctx)

        adafactor_step_2d(p_host, g, hstate, LR, B2D, Float64(-1.0), EPS2, DD, WD)
        # bf16 carrier: discretize the host trajectory like the device one.
        for i in range(N):
            p_host[i] = _bf16_rne(p_host[i])

        adafactor_step_device(
            p_dev, g_dev, dstate, step, LR, B2D, Float64(-1.0), EPS2, DD, WD,
            UInt64(0), ctx,   # RNE gate mode
        )
        ctx.synchronize()

    var p_dev_h = p_dev.to_host(ctx)
    var rv_h = dstate.row_var[].to_host(ctx)
    var cv_h = dstate.col_var[].to_host(ctx)

    var cp = _cos(p_dev_h, p_host)
    var crv = _cos(rv_h, hstate.row_var)
    var ccv = _cos(cv_h, hstate.col_var)
    print("cos(p) =", cp, "  max_abs(p) =", _maxabs(p_dev_h, p_host))
    print("cos(row_var) =", crv, "  cos(col_var) =", ccv)
    if cp < 0.9999 or crv < 0.99999 or ccv < 0.99999:
        allok = False

    # ── SR bound: sr result within 1 bf16 ulp of the RNE result ─────────────
    var p_a = Tensor.from_host(p_dev_h.copy(), [ROWS, COLS], STDtype.BF16, ctx)
    var p_b = Tensor.from_host(p_dev_h.copy(), [ROWS, COLS], STDtype.BF16, ctx)
    var g4 = _fill(N, 0.0017, 0.9, 0.05)
    var g4d = Tensor.from_host(g4.copy(), [ROWS, COLS], STDtype.F32, ctx)
    var g4d2 = Tensor.from_host(g4.copy(), [ROWS, COLS], STDtype.F32, ctx)
    # fresh states (same trajectory) for the pair
    var sa = AdafactorDeviceState(ROWS, COLS, ctx)
    var sb = AdafactorDeviceState(ROWS, COLS, ctx)
    adafactor_step_device(p_a, g4d, sa, 1, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(0), ctx)
    adafactor_step_device(p_b, g4d2, sb, 1, LR, B2D, Float64(-1.0), EPS2, DD, WD, UInt64(12345), ctx)
    ctx.synchronize()
    var pa = p_a.to_host(ctx)
    var pb = p_b.to_host(ctx)
    var sr_viol = 0
    var sr_moved = 0
    for i in range(N):
        var d = pa[i] - pb[i]
        if d < 0: d = -d
        if d > 0:
            sr_moved += 1
            # one bf16 ulp at |x|: 2^-8 relative (7-bit mantissa) + eps floor
            var mag = pa[i] if pa[i] >= 0 else -pa[i]
            var ulp = mag * Float32(1.0 / 128.0) + Float32(1e-7)
            if d > ulp:
                sr_viol += 1
    print("SR: moved", sr_moved, "/", N, " elements; >1ulp violations:", sr_viol)
    if sr_viol > 0:
        allok = False
    if sr_moved == 0:
        print("SR: WARNING zero elements moved (rounding never hit) — suspicious")
        allok = False

    if allok:
        print("VERDICT: PASS — device Adafactor matches the host torch-oracle (RNE) + SR bounded")
    else:
        print("VERDICT: FAIL")
