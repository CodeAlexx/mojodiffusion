# models/lingbotvideo/parity/dense_block_microbench.mojo — one dense block
# forward at full T2V sequence length (S = 31*12*20 + 420 = 7860), synthetic
# weights. Measures the per-block wall time (the AdaLN modulation path is the
# target: host-side before the GPU-modulation fix, device-side after).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/dense_block_microbench.mojo

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear, linear_bias
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_tiled
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.lingbotvideo.backbone import LingBotAttnConfig
from serenitymojo.models.lingbotvideo.dense import (
    lingbot_video_dense_block,
    lingbot_dense_ffn,
    _rms_norm_bf16_dev,
    _adaln_modulate,
)

comptime S = 7860        # 31*12*20 video tokens + 420 text tokens
comptime H = 2048
comptime I = 6144
comptime SIX = 6 * H


def _fill(n: Int, seed: Int, scale: Float32) raises -> List[Float32]:
    var out = List[Float32]()
    out.resize(n, Float32(0.0))
    var state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out[i] = (Float32(state % 2000) / Float32(1000.0) - Float32(1.0)) * scale
    return out^


def _t(n_rows: Int, n_cols: Int, seed: Int, dt: STDtype, scale: Float32,
       ctx: DeviceContext) raises -> Tensor:
    if n_rows == 0:
        var v1 = _fill(n_cols, seed, scale)
        return Tensor.from_host(v1, [n_cols], dt, ctx)
    var vals = _fill(n_rows * n_cols, seed, scale)
    return Tensor.from_host(vals, [n_rows, n_cols], dt, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[BENCH] building synthetic block weights (S =", S, ")")
    var x = Tensor.from_host(_fill(S * H, 1, 0.5), [1, S, H], STDtype.BF16, ctx)
    var temb_row = _t(1, SIX, 2, STDtype.F32, 0.1, ctx)
    var sst = _t(1, SIX, 3, STDtype.F32, 0.1, ctx)
    var w_q = _t(H, H, 4, STDtype.BF16, 0.02, ctx)
    var w_k = _t(H, H, 5, STDtype.BF16, 0.02, ctx)
    var w_v = _t(H, H, 6, STDtype.BF16, 0.02, ctx)
    var w_o = _t(H, H, 7, STDtype.BF16, 0.02, ctx)
    var b_o = _t(0, H, 8, STDtype.BF16, 0.02, ctx)
    var norm_q = _t(0, 128, 9, STDtype.F32, 1.0, ctx)
    var norm_k = _t(0, 128, 10, STDtype.F32, 1.0, ctx)
    var cos_h = _t(S * 16, 64, 11, STDtype.F32, 1.0, ctx)  # pre-tiled per-head
    var sin_h = _t(S * 16, 64, 12, STDtype.F32, 1.0, ctx)
    var n1w = _t(0, H, 13, STDtype.F32, 1.0, ctx)
    var n2w = _t(0, H, 14, STDtype.F32, 1.0, ctx)
    var npaw = _t(0, H, 15, STDtype.F32, 1.0, ctx)
    var npfw = _t(0, H, 16, STDtype.F32, 1.0, ctx)
    var gw = _t(I, H, 17, STDtype.BF16, 0.02, ctx)
    var uw = _t(I, H, 18, STDtype.BF16, 0.02, ctx)
    var dw = _t(H, I, 19, STDtype.BF16, 0.02, ctx)

    print("[BENCH] timing lingbot_video_dense_block[S=", S, "] (3 iters)")
    for it in range(3):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var out = lingbot_video_dense_block[S](
            x, temb_row, sst, w_q, w_k, w_v, w_o, b_o, norm_q, norm_k,
            cos_h, sin_h, n1w, n2w, npaw, npfw, gw, uw, dw, cfg, ctx,
        )
        _ = out.to_host(ctx)[0]  # force full sync
        var t1 = perf_counter_ns()
        print("[BENCH] iter", it, " block wall =",
              Float64(t1 - t0) / 1.0e9, "s")

    # ── phase breakdown ──────────────────────────────────────────────────────
    print("[BENCH] phase breakdown (synced per phase)")
    ctx.synchronize()
    var p0 = perf_counter_ns()
    var n1 = _rms_norm_bf16_dev(x, n1w, cfg.eps, ctx)
    ctx.synchronize()
    var p1 = perf_counter_ns()
    var attn_in = _adaln_modulate(
        n1, temb_row, sst, S, H, 0, H, [1, S, H], ctx)
    ctx.synchronize()
    var p2 = perf_counter_ns()
    var q = linear(attn_in, w_q, None, ctx)
    var k = linear(attn_in, w_k, None, ctx)
    var v = linear(attn_in, w_v, None, ctx)
    ctx.synchronize()
    var p3 = perf_counter_ns()
    var q_rows = reshape(q, [S * 16, 128], ctx)
    var k_rows = reshape(k, [S * 16, 128], ctx)
    var q_n = rms_norm(q_rows, norm_q, cfg.eps, ctx)
    var k_n = rms_norm(k_rows, norm_k, cfg.eps, ctx)
    var q_r = rope_interleaved(q_n, cos_h, sin_h, ctx)
    var k_r = rope_interleaved(k_n, cos_h, sin_h, ctx)
    ctx.synchronize()
    var p4 = perf_counter_ns()
    var q4 = reshape(q_r, [1, S, 16, 128], ctx)
    var k4 = reshape(k_r, [1, S, 16, 128], ctx)
    var v4 = reshape(v, [1, S, 16, 128], ctx)
    var attn = sdpa_nomask_tiled[1, S, 16, 128](
        q4, k4, v4, Float32(0.08838834764831845), ctx)
    ctx.synchronize()
    var p5 = perf_counter_ns()
    var attn_2d = reshape(attn, [S, H], ctx)
    var o = linear_bias(attn_2d, w_o, b_o, ctx)
    ctx.synchronize()
    var p6 = perf_counter_ns()
    var ffn_out = lingbot_dense_ffn(attn_2d, gw, uw, dw, ctx)
    ctx.synchronize()
    var p7 = perf_counter_ns()
    _ = o.to_host(ctx)[0]
    _ = ffn_out.to_host(ctx)[0]
    var attn_m = sdpa_nomask[1, S, 16, 128](
        q4, k4, v4, Float32(0.08838834764831845), ctx)
    ctx.synchronize()
    var p8 = perf_counter_ns()
    _ = attn_m.to_host(ctx)[0]
    print("[BENCH] sdpa math (cuBLAS)=", Float64(p8 - p7) / 1.0e9, "s")
    print("[BENCH] rms_norm_dev      =", Float64(p1 - p0) / 1.0e9, "s")
    print("[BENCH] adaln_modulate    =", Float64(p2 - p1) / 1.0e9, "s")
    print("[BENCH] qkv linears       =", Float64(p3 - p2) / 1.0e9, "s")
    print("[BENCH] qk norm+rope      =", Float64(p4 - p3) / 1.0e9, "s")
    print("[BENCH] sdpa tiled        =", Float64(p5 - p4) / 1.0e9, "s")
    print("[BENCH] out proj          =", Float64(p6 - p5) / 1.0e9, "s")
    print("[BENCH] dense ffn         =", Float64(p7 - p6) / 1.0e9, "s")
