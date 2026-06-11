# ops/tests/sdpa_flash_parity.mojo — numerics + speed gate for the cuDNN
# flash SDPA port (ops/attention_flash.mojo), per the approved sign-off
# (memory: sdpa-flash-signoff): flash is a DIFFERENT summation order than the
# math-mode path, so the gate is NOT bit-equality — it is:
#
#   1. flash-vs-math agreement on identical bf16 inputs (cosine + max-abs:
#      both are the same mathematical function; differences are bf16
#      reduction-order class).
#   2. ACCURACY REFEREE: both paths against the F32 math reference on the
#      same (bf16-rounded) values. Flash must not be meaningfully WORSE than
#      math-mode: rms_err(flash) <= 1.5 * rms_err(math) + 1e-7.
#   3. Same for the backward (d_q/d_k/d_v).
#   4. Speed: flash must beat math-mode (the entire point; the audit said
#      ~10-30x on these shapes).
#
# Shapes: the real trainer SDPAs —
#   zimage B1 [1,1248,30,128]  (S NOT 128-aligned -> exercises the padding
#                               + real_N mask path)
#   klein  B1 [1,1536,32,128]  (128-aligned -> no padding)
#   zimage B2 [2,1248,30,128]  (batch + padding)
#
# Build (note the extra cshim link args):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/ops/tests/sdpa_flash_parity.mojo -o /tmp/sdpa_flash_par
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     /tmp/sdpa_flash_par

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd,
    sdpa_flash_backward,
)


def _pattern(n: Int, seed: UInt64, amp: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u * Float32(2.0) - Float32(1.0)) * amp)
    return out^


struct Agree(Copyable, Movable):
    var cosine: Float64
    var max_abs: Float64
    var rms: Float64

    def __init__(out self, cosine: Float64, max_abs: Float64, rms: Float64):
        self.cosine = cosine
        self.max_abs = max_abs
        self.rms = rms


def _agree(a: List[Float32], b: List[Float32]) raises -> Agree:
    if len(a) != len(b):
        raise Error("_agree: length mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    var max_abs = 0.0
    var sse = 0.0
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        sse += (x - y) * (x - y)
    var denom = sqrt(na) * sqrt(nb)
    var cosine = 1.0
    if denom > 0.0:
        cosine = dot / denom
    return Agree(cosine, max_abs, sqrt(sse / Float64(len(a))))


def _gate_line(
    name: String, what: String, a: Agree, cos_bar: Float64
) raises -> Bool:
    var ok = a.cosine >= cos_bar
    print(
        "GATE sdpa_flash ", name, " ", what,
        " cos=", a.cosine, " max_abs=", a.max_abs, " rms=", a.rms,
        " ", "PASS" if ok else "FAIL (cos bar " + String(cos_bar) + ")",
    )
    return ok


def _referee_line(
    name: String, what: String, err_flash: Float64, err_math: Float64
) raises -> Bool:
    var ok = err_flash <= err_math * 1.5 + 1.0e-7
    print(
        "GATE sdpa_flash ", name, " ", what,
        " rms_vs_f32: flash=", err_flash, " math=", err_math,
        " ", "PASS" if ok else "FAIL (flash > 1.5x math)",
    )
    return ok


def _run_case[
    B: Int, S: Int, H: Int, Dh: Int
](name: String, ctx: DeviceContext) raises -> Bool:
    var n = B * S * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    print("=== case ", name, " [", B, ",", S, ",", H, ",", Dh, "] ===")

    var q_h = _pattern(n, 11, Float32(1.0))
    var k_h = _pattern(n, 22, Float32(1.0))
    var v_h = _pattern(n, 33, Float32(1.0))
    var do_h = _pattern(n, 44, Float32(1.0))

    var shape: List[Int] = [B, S, H, Dh]
    var q = Tensor.from_host(q_h.copy(), shape.copy(), STDtype.BF16, ctx)
    var k = Tensor.from_host(k_h.copy(), shape.copy(), STDtype.BF16, ctx)
    var v = Tensor.from_host(v_h.copy(), shape.copy(), STDtype.BF16, ctx)
    var d_o = Tensor.from_host(do_h.copy(), shape.copy(), STDtype.BF16, ctx)

    # F32 referee on the SAME bf16-rounded values.
    var q32 = Tensor.from_host(q.to_host(ctx), shape.copy(), STDtype.F32, ctx)
    var k32 = Tensor.from_host(k.to_host(ctx), shape.copy(), STDtype.F32, ctx)
    var v32 = Tensor.from_host(v.to_host(ctx), shape.copy(), STDtype.F32, ctx)
    var do32 = Tensor.from_host(d_o.to_host(ctx), shape.copy(), STDtype.F32, ctx)

    # math-mode bf16 (the incumbent), math-mode F32 (the referee), flash bf16.
    var o_math = sdpa_nomask[B, S, H, Dh](q, k, v, scale, ctx)
    var o_ref = sdpa_nomask[B, S, H, Dh](q32, k32, v32, scale, ctx)
    var fwd = sdpa_flash_train_fwd[B, S, H, Dh](q, k, v, scale, ctx)
    ctx.synchronize()

    var o_math_h = o_math.to_host(ctx)
    var o_ref_h = o_ref.to_host(ctx)
    var o_flash_h = fwd.o.to_host(ctx)

    var ok = True
    var fm = _agree(o_flash_h, o_math_h)
    ok = _gate_line(name, "fwd flash-vs-math", fm, 0.999) and ok
    var fr = _agree(o_flash_h, o_ref_h)
    var mr = _agree(o_math_h, o_ref_h)
    ok = _referee_line(name, "fwd", fr.rms, mr.rms) and ok

    # backward
    var g_math = sdpa_backward[B, S, H, Dh](q, k, v, d_o, scale, ctx)
    var g_ref = sdpa_backward[B, S, H, Dh](q32, k32, v32, do32, scale, ctx)
    var g_flash = sdpa_flash_backward[B, S, H, Dh](fwd, d_o, scale, ctx)
    ctx.synchronize()

    var names: List[String] = [String("d_q"), String("d_k"), String("d_v")]
    for which in range(3):
        var mh: List[Float32]
        var rh: List[Float32]
        var fh: List[Float32]
        if which == 0:
            mh = g_math.d_q.to_host(ctx); rh = g_ref.d_q.to_host(ctx); fh = g_flash.d_q.to_host(ctx)
        elif which == 1:
            mh = g_math.d_k.to_host(ctx); rh = g_ref.d_k.to_host(ctx); fh = g_flash.d_k.to_host(ctx)
        else:
            mh = g_math.d_v.to_host(ctx); rh = g_ref.d_v.to_host(ctx); fh = g_flash.d_v.to_host(ctx)
        var a_fm = _agree(fh, mh)
        ok = _gate_line(name, "bwd " + names[which] + " flash-vs-math", a_fm, 0.999) and ok
        var a_fr = _agree(fh, rh)
        var a_mr = _agree(mh, rh)
        ok = _referee_line(name, "bwd " + names[which], a_fr.rms, a_mr.rms) and ok

    # speed (10 iters each, synchronized)
    comptime ITERS = 10
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var om = sdpa_nomask[B, S, H, Dh](q, k, v, scale, ctx)
        var gm = sdpa_backward[B, S, H, Dh](q, k, v, d_o, scale, ctx)
        _ = om^
        _ = gm^
    ctx.synchronize()
    var t1 = perf_counter_ns()
    for _ in range(ITERS):
        var ff = sdpa_flash_train_fwd[B, S, H, Dh](q, k, v, scale, ctx)
        var gf = sdpa_flash_backward[B, S, H, Dh](ff, d_o, scale, ctx)
        _ = gf^
    ctx.synchronize()
    var t2 = perf_counter_ns()
    var math_ms = Float64(t1 - t0) / 1.0e6 / Float64(ITERS)
    var flash_ms = Float64(t2 - t1) / 1.0e6 / Float64(ITERS)
    var speedup = math_ms / flash_ms if flash_ms > 0.0 else 0.0
    var sp_ok = flash_ms < math_ms
    print(
        "GATE sdpa_flash ", name, " speed fwd+bwd: math=", math_ms,
        " ms  flash=", flash_ms, " ms  speedup=", speedup, "x ",
        "PASS" if sp_ok else "FAIL (flash not faster)",
    )
    ok = ok and sp_ok
    return ok


def main() raises:
    var ctx = DeviceContext()
    print("=== sdpa_flash_parity: cuDNN flash vs math-mode + F32 referee ===")
    var ok = True
    ok = _run_case[1, 1536, 32, 128](String("klein_b1_aligned"), ctx) and ok
    ok = _run_case[1, 1248, 30, 128](String("zimage_b1_pad"), ctx) and ok
    ok = _run_case[2, 1248, 30, 128](String("zimage_b2_pad"), ctx) and ok
    if ok:
        print("=== sdpa_flash_parity: ALL GATES PASS ===")
    else:
        print("=== sdpa_flash_parity: FAILURES (see GATE lines) ===")
        raise Error("sdpa_flash_parity failed")
