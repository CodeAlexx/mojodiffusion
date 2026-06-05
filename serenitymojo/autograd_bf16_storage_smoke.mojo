# autograd_bf16_storage_smoke.mojo -- focused BF16 storage gate for Tape grads.
#
# Run:
#   pixi run mojo run -I . serenitymojo/autograd_bf16_storage_smoke.mojo

from std.builtin.dtype import DType
from std.math import exp
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.autograd import Tape, backward, ones_like
from serenitymojo.ops.norm_backward import rms_norm_backward


comptime _RMS_EPS = Float32(1e-6)


def _bf16(v: Float32) -> Float32:
    return v.cast[DType.bfloat16]().cast[DType.float32]()


def _max_abs(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("max_abs: length mismatch")
    var m = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i] - b[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def _check_bf16(name: String, t: Tensor) raises:
    if t.dtype() != STDtype.BF16:
        raise Error(name + ": expected BF16 grad storage, got " + t.dtype().name())


def _vals(n: Int, mul: Int, mod: Int, center: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * mul) % mod) - Float32(center)) * scale)
    return out^


def _expect_close(name: String, got: List[Float32], expected: List[Float32], h: ParityHarness) raises:
    var r = h.compare_host(got, expected)
    var ma = _max_abs(got, expected)
    print(name, r, " max_abs=", ma)
    if not r.passed or ma > 0.02:
        raise Error(name + ": BF16 numerical sanity failed")


def _elementwise_chain(ctx: DeviceContext, h: ParityHarness) raises:
    comptime N = 16
    var sh = List[Int](); sh.append(N)
    var av = _vals(N, 7, 13, 6, 0.05)
    var bv = _vals(N, 5, 11, 5, 0.05)
    var cv = _vals(N, 3, 9, 4, 0.05)
    var a = Tensor.from_host(av.copy(), sh.copy(), STDtype.BF16, ctx)
    var b = Tensor.from_host(bv.copy(), sh.copy(), STDtype.BF16, ctx)
    var c = Tensor.from_host(cv.copy(), sh.copy(), STDtype.BF16, ctx)
    var tape = Tape()
    tape.track(a); tape.track(b); tape.track(c)
    var s = tape.record_add(a, b, ctx)
    var y = tape.record_mul(s, c, ctx)
    if s.dtype() != STDtype.BF16 or y.dtype() != STDtype.BF16:
        raise Error("elementwise chain did not preserve BF16 forward storage")
    var grads = backward(tape, y, ctx)
    _check_bf16("elementwise d_a", grads[a.id][])
    _check_bf16("elementwise d_b", grads[b.id][])
    _check_bf16("elementwise d_c", grads[c.id][])
    var exp_da = List[Float32]()
    var exp_db = List[Float32]()
    var exp_dc = List[Float32]()
    for i in range(N):
        var ab = _bf16(_bf16(av[i]) + _bf16(bv[i]))
        exp_da.append(_bf16(cv[i]))
        exp_db.append(_bf16(cv[i]))
        exp_dc.append(ab)
    _expect_close("bf16 tape elementwise d_a", grads[a.id][].to_host(ctx), exp_da, h)
    _expect_close("bf16 tape elementwise d_b", grads[b.id][].to_host(ctx), exp_db, h)
    _expect_close("bf16 tape elementwise d_c", grads[c.id][].to_host(ctx), exp_dc, h)


def _linear(ctx: DeviceContext, h: ParityHarness) raises:
    comptime M = 4
    comptime IN = 3
    comptime OUT = 2
    var xv = _vals(M * IN, 7, 13, 6, 0.05)
    var wv = _vals(OUT * IN, 5, 11, 5, 0.05)
    var bv = _vals(OUT, 3, 9, 4, 0.05)
    var x = Tensor.from_host(xv.copy(), [M, IN], STDtype.BF16, ctx)
    var w = Tensor.from_host(wv.copy(), [OUT, IN], STDtype.BF16, ctx)
    var b = Tensor.from_host(bv.copy(), [OUT], STDtype.BF16, ctx)
    var tape = Tape()
    tape.track(x); tape.track(w); tape.track(b)
    var y = tape.record_linear(x, w, b, ctx)
    if y.dtype() != STDtype.BF16:
        raise Error("linear did not preserve BF16 forward storage")
    var grads = backward(tape, y, ctx)
    _check_bf16("linear d_x", grads[x.id][])
    _check_bf16("linear d_w", grads[w.id][])
    _check_bf16("linear d_b", grads[b.id][])
    var exp_dx = List[Float32]()
    for _m in range(M):
        for i in range(IN):
            var s: Float32 = 0.0
            for o in range(OUT):
                s += _bf16(wv[o * IN + i])
            exp_dx.append(_bf16(s))
    var exp_dw = List[Float32]()
    for o in range(OUT):
        for i in range(IN):
            var s: Float32 = 0.0
            for m in range(M):
                s += _bf16(xv[m * IN + i])
            _ = o
            exp_dw.append(_bf16(s))
    var exp_db = List[Float32]()
    for _o in range(OUT):
        exp_db.append(_bf16(Float32(M)))
    _expect_close("bf16 tape linear d_x", grads[x.id][].to_host(ctx), exp_dx, h)
    _expect_close("bf16 tape linear d_w", grads[w.id][].to_host(ctx), exp_dw, h)
    _expect_close("bf16 tape linear d_b", grads[b.id][].to_host(ctx), exp_db, h)


def _silu_swiglu(ctx: DeviceContext, h: ParityHarness) raises:
    comptime N = 12
    var sh = List[Int](); sh.append(N)
    var xv = _vals(N, 7, 13, 6, 0.3)
    var uv = _vals(N, 5, 11, 5, 0.2)
    var x = Tensor.from_host(xv.copy(), sh.copy(), STDtype.BF16, ctx)
    var up = Tensor.from_host(uv.copy(), sh.copy(), STDtype.BF16, ctx)
    var tape = Tape()
    tape.track(x); tape.track(up)
    var sx = tape.record_silu(x, ctx)
    var y = tape.record_swiglu(x, up, ctx)
    if sx.dtype() != STDtype.BF16 or y.dtype() != STDtype.BF16:
        raise Error("silu/swiglu did not preserve BF16 forward storage")
    var gs = backward(tape, sx, ctx)
    var gw = backward(tape, y, ctx)
    _check_bf16("silu d_x", gs[x.id][])
    _check_bf16("swiglu d_gate", gw[x.id][])
    _check_bf16("swiglu d_up", gw[up.id][])
    var exp_silu = List[Float32]()
    var exp_dgate = List[Float32]()
    var exp_dup = List[Float32]()
    for i in range(N):
        var gv = _bf16(xv[i])
        var uu = _bf16(uv[i])
        var sig = Float32(1.0) / (Float32(1.0) + exp(-gv))
        var silu_g = gv * sig
        var dsilu = sig * (Float32(1.0) + gv * (Float32(1.0) - sig))
        exp_silu.append(_bf16(dsilu))
        exp_dup.append(_bf16(silu_g))
        exp_dgate.append(_bf16(uu * dsilu))
    _expect_close("bf16 tape silu d_x", gs[x.id][].to_host(ctx), exp_silu, h)
    _expect_close("bf16 tape swiglu d_gate", gw[x.id][].to_host(ctx), exp_dgate, h)
    _expect_close("bf16 tape swiglu d_up", gw[up.id][].to_host(ctx), exp_dup, h)


def _rms_mse(ctx: DeviceContext, h: ParityHarness) raises:
    comptime ROWS = 4
    comptime D = 6
    var xv = _vals(ROWS * D, 7, 13, 6, 0.1)
    var gv = List[Float32]()
    for i in range(D):
        gv.append(Float32(1.0) + _vals(D, 5, 11, 5, 0.05)[i])
    var x = Tensor.from_host(xv.copy(), [ROWS, D], STDtype.BF16, ctx)
    var gamma = Tensor.from_host(gv.copy(), [D], STDtype.BF16, ctx)
    var tape = Tape()
    tape.track(x); tape.track(gamma)
    var y = tape.record_rms_norm(x, gamma, ctx)
    var grads = backward(tape, y, ctx)
    _check_bf16("rms d_x", grads[x.id][])
    _check_bf16("rms d_gamma", grads[gamma.id][])
    var dy = ones_like(y, ctx)
    var reference = rms_norm_backward(dy, x, gamma, _RMS_EPS, ctx)
    _check_bf16("rms ref d_x", reference.d_x)
    _check_bf16("rms ref d_gamma", reference.d_g)
    _expect_close("bf16 tape rms d_x", grads[x.id][].to_host(ctx), reference.d_x.to_host(ctx), h)
    _expect_close("bf16 tape rms d_gamma", grads[gamma.id][].to_host(ctx), reference.d_g.to_host(ctx), h)

    comptime N = 12
    var predv = _vals(N, 7, 13, 6, 0.2)
    var tgtv = _vals(N, 5, 11, 5, 0.15)
    var pred = Tensor.from_host(predv.copy(), [N], STDtype.BF16, ctx)
    var target = Tensor.from_host(tgtv.copy(), [N], STDtype.BF16, ctx)
    var mse_tape = Tape()
    mse_tape.track(pred)
    var loss = mse_tape.mse_loss(pred, target, ctx)
    var mse_grads = backward(mse_tape, loss, ctx)
    _check_bf16("mse d_pred", mse_grads[pred.id][])
    var exp_dp = List[Float32]()
    for i in range(N):
        exp_dp.append(_bf16(Float32(2.0) * (_bf16(predv[i]) - _bf16(tgtv[i])) / Float32(N)))
    _expect_close("bf16 tape mse d_pred", mse_grads[pred.id][].to_host(ctx), exp_dp, h)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    _elementwise_chain(ctx, h)
    _linear(ctx, h)
    _silu_swiglu(ctx, h)
    _rms_mse(ctx, h)
    print("autograd BF16 storage smoke PASS")
