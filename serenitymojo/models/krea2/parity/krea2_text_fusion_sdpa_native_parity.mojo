# Krea2 TextFusion native SDPA isolate.
#
# Focus: the BF16 cuDNN SDPA training backend used by TextFusion LoRA at
# B=1,S=16,H=20,Dh=128. This is deliberately narrower than the full
# krea2_text_fusion_lora_parity gate: it checks the attention backend/order
# surface without touching model production code.
#
# Run:
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/parity/krea2_text_fusion_sdpa_native_parity.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.attention_flash import (
    sdpa_flash_backward,
    sdpa_flash_backward_native,
    sdpa_flash_train_fwd,
    sdpa_flash_train_fwd_native,
)
from serenitymojo.tensor import Tensor


comptime B = 1
comptime S = 16
comptime H = 20
comptime Dh = 128
comptime N = B * S * H * Dh


struct Agree(Copyable, Movable):
    var cosine: Float64
    var max_abs: Float64
    var rms: Float64

    def __init__(out self, cosine: Float64, max_abs: Float64, rms: Float64):
        self.cosine = cosine
        self.max_abs = max_abs
        self.rms = rms


def _shape() -> List[Int]:
    var out: List[Int] = [B, S, H, Dh]
    return out^


def _lcg_pattern(n: Int, seed: UInt64, amp: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * amp)
    return out^


def _layout_sentinel(kind: Int) -> List[Float32]:
    var out = List[Float32]()
    for b in range(B):
        for s in range(S):
            for h in range(H):
                for d in range(Dh):
                    var code = (b * 101 + s * 37 + h * 13 + d * 3 + kind * 17) % 29
                    var centered = Float32(code - 14) * Float32(0.035)
                    var slow_axes = Float32(s - 7) * Float32(0.002)
                    slow_axes += Float32(h - 9) * Float32(0.003)
                    out.append(centered + slow_axes)
    return out^


def _agree(a: List[Float32], b: List[Float32]) raises -> Agree:
    if len(a) != len(b):
        raise Error("_agree: length mismatch")
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var max_abs: Float64 = 0.0
    var sse: Float64 = 0.0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var diff = x - y
        var ad = diff
        if ad < 0.0:
            ad = -ad
        if ad > max_abs:
            max_abs = ad
        sse += diff * diff
    var denom = sqrt(na) * sqrt(nb)
    var cosine: Float64 = 1.0
    if denom > 0.0:
        cosine = dot / denom
    return Agree(cosine, max_abs, sqrt(sse / Float64(len(a))))


def _check_dtype(t: Tensor, dtype: STDtype, name: String) raises:
    if t.dtype() != dtype:
        raise Error(
            name + String(" dtype boundary changed: got=")
            + t.dtype().name() + String(" expected=") + dtype.name()
        )


def _gate_line(name: String, what: String, a: Agree, cos_bar: Float64) -> Bool:
    var ok = a.cosine >= cos_bar
    print(
        "GATE krea2_txtfusion_native_sdpa ", name, " ", what,
        " cos=", a.cosine, " max_abs=", a.max_abs, " rms=", a.rms,
        " ", "PASS" if ok else "FAIL",
    )
    return ok


def _referee_line(
    name: String, what: String, err_native: Float64, err_math: Float64
) -> Bool:
    var ok = err_native <= err_math * 1.5 + 1.0e-7
    print(
        "GATE krea2_txtfusion_native_sdpa ", name, " ", what,
        " rms_vs_f32: native=", err_native, " math=", err_math,
        " ", "PASS" if ok else "FAIL (native > 1.5x math)",
    )
    return ok


def _run_case(
    name: String,
    var q_h: List[Float32],
    var k_h: List[Float32],
    var v_h: List[Float32],
    var do_h: List[Float32],
    ctx: DeviceContext,
) raises -> Bool:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var shape = _shape()
    print("=== Krea2 TextFusion native SDPA case ", name, " [1,16,20,128] ===")

    var q = Tensor.from_host(q_h^, shape.copy(), STDtype.BF16, ctx)
    var k = Tensor.from_host(k_h^, shape.copy(), STDtype.BF16, ctx)
    var v = Tensor.from_host(v_h^, shape.copy(), STDtype.BF16, ctx)
    var d_o = Tensor.from_host(do_h^, shape.copy(), STDtype.BF16, ctx)
    _check_dtype(q, STDtype.BF16, name + String(".q"))
    _check_dtype(k, STDtype.BF16, name + String(".k"))
    _check_dtype(v, STDtype.BF16, name + String(".v"))
    _check_dtype(d_o, STDtype.BF16, name + String(".d_out"))

    # F32 referee uses the same BF16-rounded values, matching the local flash
    # parity pattern without creating an F32 storage boundary in the product path.
    var q32 = Tensor.from_host(q.to_host(ctx), shape.copy(), STDtype.F32, ctx)
    var k32 = Tensor.from_host(k.to_host(ctx), shape.copy(), STDtype.F32, ctx)
    var v32 = Tensor.from_host(v.to_host(ctx), shape.copy(), STDtype.F32, ctx)
    var do32 = Tensor.from_host(d_o.to_host(ctx), shape.copy(), STDtype.F32, ctx)

    var o_math = sdpa_nomask[B, S, H, Dh](q, k, v, scale, ctx)
    var o_ref = sdpa_nomask[B, S, H, Dh](q32, k32, v32, scale, ctx)
    var fwd_native = sdpa_flash_train_fwd_native[B, S, H, Dh](q, k, v, scale, ctx)
    var fwd_padded = sdpa_flash_train_fwd[B, S, H, Dh](q, k, v, scale, ctx)
    ctx.synchronize()

    _check_dtype(fwd_native.o, STDtype.BF16, name + String(".native.o"))
    _check_dtype(fwd_native.o_pad, STDtype.BF16, name + String(".native.o_pad"))
    _check_dtype(fwd_native.q_pad, STDtype.BF16, name + String(".native.q_pad"))
    _check_dtype(fwd_native.k_pad, STDtype.BF16, name + String(".native.k_pad"))
    _check_dtype(fwd_native.v_pad, STDtype.BF16, name + String(".native.v_pad"))
    _check_dtype(fwd_native.stats, STDtype.F32, name + String(".native.stats"))

    var ok = True
    var o_native_h = fwd_native.o.to_host(ctx)
    var o_padded_h = fwd_padded.o.to_host(ctx)
    var o_math_h = o_math.to_host(ctx)
    var o_ref_h = o_ref.to_host(ctx)
    ok = _gate_line(name, "fwd native-vs-math", _agree(o_native_h, o_math_h), 0.999) and ok
    ok = _gate_line(name, "fwd native-vs-padded-flash", _agree(o_native_h, o_padded_h), 0.999) and ok
    var o_native_ref = _agree(o_native_h, o_ref_h)
    var o_math_ref = _agree(o_math_h, o_ref_h)
    ok = _referee_line(name, "fwd", o_native_ref.rms, o_math_ref.rms) and ok

    var g_math = sdpa_backward[B, S, H, Dh](q, k, v, d_o, scale, ctx)
    var g_ref = sdpa_backward[B, S, H, Dh](q32, k32, v32, do32, scale, ctx)
    var g_native = sdpa_flash_backward_native[B, S, H, Dh](fwd_native, d_o, scale, ctx)
    var g_padded = sdpa_flash_backward[B, S, H, Dh](fwd_padded, d_o, scale, ctx)
    ctx.synchronize()

    _check_dtype(g_native.d_q, STDtype.BF16, name + String(".native.d_q"))
    _check_dtype(g_native.d_k, STDtype.BF16, name + String(".native.d_k"))
    _check_dtype(g_native.d_v, STDtype.BF16, name + String(".native.d_v"))

    var labels: List[String] = [String("d_q"), String("d_k"), String("d_v")]
    for which in range(3):
        var mh: List[Float32]
        var rh: List[Float32]
        var nh: List[Float32]
        var ph: List[Float32]
        if which == 0:
            mh = g_math.d_q.to_host(ctx)
            rh = g_ref.d_q.to_host(ctx)
            nh = g_native.d_q.to_host(ctx)
            ph = g_padded.d_q.to_host(ctx)
        elif which == 1:
            mh = g_math.d_k.to_host(ctx)
            rh = g_ref.d_k.to_host(ctx)
            nh = g_native.d_k.to_host(ctx)
            ph = g_padded.d_k.to_host(ctx)
        else:
            mh = g_math.d_v.to_host(ctx)
            rh = g_ref.d_v.to_host(ctx)
            nh = g_native.d_v.to_host(ctx)
            ph = g_padded.d_v.to_host(ctx)
        ok = _gate_line(name, "bwd " + labels[which] + " native-vs-math", _agree(nh, mh), 0.999) and ok
        ok = _gate_line(name, "bwd " + labels[which] + " native-vs-padded-flash", _agree(nh, ph), 0.999) and ok
        var n_ref = _agree(nh, rh)
        var m_ref = _agree(mh, rh)
        ok = _referee_line(name, "bwd " + labels[which], n_ref.rms, m_ref.rms) and ok

    return ok


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    ok = _run_case(
        String("lcg"),
        _lcg_pattern(N, 11, Float32(0.8)),
        _lcg_pattern(N, 22, Float32(0.8)),
        _lcg_pattern(N, 33, Float32(0.8)),
        _lcg_pattern(N, 44, Float32(0.8)),
        ctx,
    ) and ok
    ok = _run_case(
        String("bshd_sentinel"),
        _layout_sentinel(0),
        _layout_sentinel(1),
        _layout_sentinel(2),
        _layout_sentinel(3),
        ctx,
    ) and ok
    if not ok:
        raise Error("Krea2 TextFusion native SDPA isolate failed")
    print("PASS: Krea2 TextFusion native SDPA BF16 fwd/bwd isolate")
