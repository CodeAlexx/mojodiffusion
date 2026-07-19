# chroma_padmask_parity.mojo — gate for the SerenityTrainer T5 pad-KEY mask in the
# chroma device attention seam (_chroma_sdpa_fwd/_chroma_sdpa_bwd, 2026-07-16).
#
# SEMANTIC CLAIM UNDER TEST: key-masked attention on the [txt(512)|img(1024)]
# joint sequence with valid txt length LT equals attention computed on the
# TRUNCATED sequence [txt_valid(LT)|img] with the pad rows REMOVED — for every
# valid row, in forward AND backward — and the pad rows' outputs/grads are
# EXACT zeros. Reference = math sdpa_nomask/sdpa_backward at S_T = LT+N_IMG
# (the masked arm is cuDNN flash, so the bar is the flash-vs-math value class,
# cos >= 0.999, same as the klein/zimage flash sign-off).
#
# The backward's incoming d_att pad rows are filled with GARBAGE to prove the
# masked path ignores them.
#
#   pixi run mojo run -I . serenitymojo/models/chroma/parity/chroma_padmask_parity.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt, sin, cos as fcos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice, concat
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.models.chroma.chroma_block_device import (
    _chroma_sdpa_fwd, _chroma_sdpa_bwd,
)

comptime H = 24
comptime Dh = 128
comptime N_TXT = 512
comptime N_IMG = 1024
comptime S = N_TXT + N_IMG        # 1536
comptime LT = 256                 # valid txt rows for this fixture
comptime S_T = LT + N_IMG         # 1280 truncated reference length


def _fill(n: Int, a: Float64, b: Float64) -> List[Float32]:
    # non-degenerate sinusoidal fill (repo rule: never modular (i*k)%m)
    var out = List[Float32]()
    for i in range(n):
        var x = Float64(i)
        out.append(Float32(sin(a * x + 0.37) * 0.5 + fcos(b * x + 1.1) * 0.31))
    return out^


def _bf16_t(vals: List[Float32], s: Int, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(
        Tensor.from_host(vals.copy(), [1, s, H, Dh], STDtype.F32, ctx),
        STDtype.BF16, ctx,
    )


def _host_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    return cast_tensor(t, STDtype.F32, ctx).to_host(ctx)


def _seg_cos(a: List[Float32], b: List[Float32], a0: Int, b0: Int, rows: Int) raises -> Float64:
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    var n = rows * H * Dh
    for i in range(n):
        var x = Float64(a[a0 * H * Dh + i])
        var y = Float64(b[b0 * H * Dh + i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (sqrt(na) * sqrt(nb))


def _seg_max_abs(a: List[Float32], a0: Int, rows: Int) -> Float32:
    var m = Float32(0.0)
    for i in range(rows * H * Dh):
        var x = a[a0 * H * Dh + i]
        if x < 0:
            x = -x
        if x > m:
            m = x
    return m


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def main() raises:
    var ctx = DeviceContext()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var q_h = _fill(S * H * Dh, 0.000913, 0.000407)
    var k_h = _fill(S * H * Dh, 0.000631, 0.000829)
    var v_h = _fill(S * H * Dh, 0.000517, 0.000733)
    # d_att with GARBAGE in the pad rows [LT, N_TXT) — must be ignored.
    var d_h = _fill(S * H * Dh, 0.000811, 0.000397)
    for r in range(LT, N_TXT):
        for i in range(H * Dh):
            d_h[r * H * Dh + i] = Float32(7.5)

    var q = _bf16_t(q_h, S, ctx)
    var k = _bf16_t(k_h, S, ctx)
    var v = _bf16_t(v_h, S, ctx)
    var d = _bf16_t(d_h, S, ctx)

    # truncated reference inputs: rows [0,LT) + [N_TXT,S)
    var qt = concat(1, ctx, slice(q, 1, 0, LT, ctx), slice(q, 1, N_TXT, N_IMG, ctx))
    var kt = concat(1, ctx, slice(k, 1, 0, LT, ctx), slice(k, 1, N_TXT, N_IMG, ctx))
    var vt = concat(1, ctx, slice(v, 1, 0, LT, ctx), slice(v, 1, N_TXT, N_IMG, ctx))
    var dt = concat(1, ctx, slice(d, 1, 0, LT, ctx), slice(d, 1, N_TXT, N_IMG, ctx))

    # ── forward ──
    var o_masked = _chroma_sdpa_fwd[H, Dh, S, True](q, k, v, scale, ctx, LT, N_TXT)
    var o_ref = sdpa_nomask[1, S_T, H, Dh](qt, kt, vt, scale, ctx)
    var om = _host_f32(o_masked, ctx)
    var orf = _host_f32(o_ref, ctx)
    var c_txt = _seg_cos(om, orf, 0, 0, LT)
    var c_img = _seg_cos(om, orf, N_TXT, LT, N_IMG)
    var pad_max = _seg_max_abs(om, LT, N_TXT - LT)
    print("[padmask] fwd cos txt_valid=", c_txt, " img=", c_img, " pad_max_abs=", pad_max)
    _require(c_txt >= 0.999, String("fwd txt cos below bar"))
    _require(c_img >= 0.999, String("fwd img cos below bar"))
    _require(pad_max == Float32(0.0), String("fwd pad rows not exactly zero"))

    # ── backward ──
    var g = _chroma_sdpa_bwd[H, Dh, S, True](q, k, v, d, scale, ctx, LT, N_TXT)
    var gr = sdpa_backward[1, S_T, H, Dh](qt, kt, vt, dt, scale, ctx)
    var names = List[String]()
    names.append(String("d_q"))
    names.append(String("d_k"))
    names.append(String("d_v"))
    var m_list = List[List[Float32]]()
    m_list.append(_host_f32(g.d_q, ctx))
    m_list.append(_host_f32(g.d_k, ctx))
    m_list.append(_host_f32(g.d_v, ctx))
    var r_list = List[List[Float32]]()
    r_list.append(_host_f32(gr.d_q, ctx))
    r_list.append(_host_f32(gr.d_k, ctx))
    r_list.append(_host_f32(gr.d_v, ctx))
    for gi in range(3):
        var ct = _seg_cos(m_list[gi], r_list[gi], 0, 0, LT)
        var ci = _seg_cos(m_list[gi], r_list[gi], N_TXT, LT, N_IMG)
        var pm = _seg_max_abs(m_list[gi], LT, N_TXT - LT)
        print("[padmask] bwd ", names[gi], " cos txt_valid=", ct, " img=", ci,
              " pad_max_abs=", pm)
        _require(ct >= 0.999, names[gi] + String(" txt cos below bar"))
        _require(ci >= 0.999, names[gi] + String(" img cos below bar"))
        _require(pm == Float32(0.0), names[gi] + String(" pad rows not exactly zero"))

    print("[padmask] PASS: masked flash == truncated math on all valid segments; pad rows exact 0")
