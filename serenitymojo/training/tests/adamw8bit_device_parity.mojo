# GPU AdamW8bit gate against the existing bnb-parity host oracle.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.adamw8bit import (
    Adam8bitState,
    adam8bit_create_dynamic_map,
    adam8bit_step_bnb,
    adamw8bit_device_state,
    adamw8bit_device_step,
)

comptime TArc = ArcPointer[Tensor]


def _values(n: Int, phase: Int) -> List[Float32]:
    var out = List[Float32](capacity=n)
    var x = UInt64(0x9E3779B97F4A7C15) + UInt64(phase * 7919)
    for _ in range(n):
        x = x * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float64(x >> 11) * (1.0 / 9007199254740992.0)
        out.append(Float32((u - 0.5) * 0.2))
    return out^


def _check_close(got: List[Float32], want: List[Float32], label: String) raises:
    if len(got) != len(want):
        raise Error(label + " length mismatch")
    var worst = Float32(0.0)
    for i in range(len(got)):
        var d = got[i] - want[i]
        if d < Float32(0.0):
            d = -d
        if d > worst:
            worst = d
    if worst > Float32(3.0e-7):
        raise Error(label + " max_abs=" + String(worst))
    print("PASS", label, "max_abs", worst)


def _check_codes(
    device_codes: UnsafePointer[UInt8, MutUntrackedOrigin],
    offset: Int,
    want: List[UInt8],
    label: String,
) raises:
    var bad = 0
    for i in range(len(want)):
        if device_codes[offset + i] != want[i]:
            bad += 1
    if bad != 0:
        raise Error(label + " code mismatches=" + String(bad))
    print("PASS", label, "codes exact")


def main() raises:
    var ctx = DeviceContext()
    var n0 = 512
    var n1 = 768
    var p0h = _values(n0, 1)
    var p1h = _values(n1, 2)
    var g0h = _values(n0, 3)
    var g1h = _values(n1, 4)
    var p0ref = p0h.copy()
    var p1ref = p1h.copy()
    var s0 = Adam8bitState(n0)
    var s1 = Adam8bitState(n1)
    var qs = adam8bit_create_dynamic_map(True)
    var qu = adam8bit_create_dynamic_map(False)

    var p0 = Tensor.from_host(p0h, [n0], STDtype.F32, ctx)
    var p1 = Tensor.from_host(p1h, [n1], STDtype.F32, ctx)
    var g0 = Tensor.from_host(g0h, [n0], STDtype.F32, ctx)
    var g1 = Tensor.from_host(g1h, [n1], STDtype.F32, ctx)
    var params: List[TArc] = [TArc(p0^), TArc(p1^)]
    var grads: List[TArc] = [TArc(g0^), TArc(g1^)]
    var state = adamw8bit_device_state(params, ctx)

    for step in range(1, 3):
        if step == 2:
            g0h = _values(n0, 5)
            g1h = _values(n1, 6)
            var ng0 = Tensor.from_host(g0h, [n0], STDtype.F32, ctx)
            var ng1 = Tensor.from_host(g1h, [n1], STDtype.F32, ctx)
            grads = [TArc(ng0^), TArc(ng1^)]
        adam8bit_step_bnb(
            p0ref, g0h, s0, qs, qu, step, Float32(5.0e-5),
            Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01),
        )
        adam8bit_step_bnb(
            p1ref, g1h, s1, qs, qu, step, Float32(5.0e-5),
            Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01),
        )
        adamw8bit_device_step(
            params, grads, state, step, Float32(5.0e-5),
            Float32(0.9), Float32(0.999), Float32(1.0e-8), Float32(0.01), ctx,
        )
        ctx.synchronize()
        _check_close(params[0][].to_host(ctx), p0ref, "param0 step " + String(step))
        _check_close(params[1][].to_host(ctx), p1ref, "param1 step " + String(step))

        var mh = ctx.enqueue_create_host_buffer[DType.uint8](state.total_padded)
        var vh = ctx.enqueue_create_host_buffer[DType.uint8](state.total_padded)
        ctx.enqueue_copy(dst_buf=mh, src_buf=state.m_codes[].buf)
        ctx.enqueue_copy(dst_buf=vh, src_buf=state.v_codes[].buf)
        ctx.synchronize()
        var mp = mh.unsafe_ptr().bitcast[UInt8]()
        var vp = vh.unsafe_ptr().bitcast[UInt8]()
        _check_codes(mp, 0, s0.m_codes, "m0 step " + String(step))
        _check_codes(vp, 0, s0.v_codes, "v0 step " + String(step))
        _check_codes(mp, n0, s1.m_codes, "m1 step " + String(step))
        _check_codes(vp, n0, s1.v_codes, "v1 step " + String(step))

        var ma = state.m_absmax[].to_host(ctx)
        var va = state.v_absmax[].to_host(ctx)
        var want_ma = List[Float32]()
        var want_va = List[Float32]()
        for x in s0.m_absmax:
            want_ma.append(x)
        for x in s1.m_absmax:
            want_ma.append(x)
        for x in s0.v_absmax:
            want_va.append(x)
        for x in s1.v_absmax:
            want_va.append(x)
        _check_close(ma, want_ma, "m_absmax step " + String(step))
        _check_close(va, want_va, "v_absmax step " + String(step))
    print("ALL PASS: device AdamW8bit matches host bnb-parity oracle")
