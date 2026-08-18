# ops/tests/int8_linear_parity.mojo — W8A8 int8 fwd + bwd vs bf16 references.
#   fwd:  out = x @ Wᵀ        (cosine vs bf16 matmul)
#   bwd:  dX  = grad @ W      (cosine vs bf16 matmul, contract N)
# Weight is tensorwise-quantized (== reference trainer) and transposed for the backward.
#
# Build (rootless, bounded, cshim + libm):
#   MEM_MAX=8G scripts/mem_safe_runtime.sh pixi run mojo build \
#     --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
#     -Xlinker=-L/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     -Xlinker=-lm -Xlinker=-lcuda -Xlinker=-lcublas \
#     -Xlinker=-Lserenitymojo/ops/cshim/lib \
#     -Xlinker=-lserenity_cudnn_sdpa \
#     serenitymojo/ops/tests/int8_linear_parity.mojo \
#     -o output/checks/int8_linear_unaligned_gate/int8_linear_parity
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, pi
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.int8_quant import int8_tensorwise_scale, int8_encode_tensorwise, int8_transpose
from serenitymojo.ops.int8_linear import (
    int8_linear_fwd, int8_linear_bwd, int8_linear_bwd_nn,
    int8_linear_bwd_nn_f32, int8_linear_bwd_nt_f32,
    int8_linear_bwd_nn_f32_bands, int8_linear_bwd_nt_f32_bands,
)


def _gaussian(n: Int, seed: Int, sd: Float32) -> List[Float32]:
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 12345)
    for _i in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64(st >> 11) + 1.0) / Float64(1 << 53)
        st = st * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(st >> 11) / Float64(1 << 53)
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(2.0 * pi * u2)) * sd)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0; var na: Float64 = 0.0; var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i]); na += Float64(a[i]) * Float64(a[i]); nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    comptime M = 64
    comptime K = 256
    comptime N = 128

    var xh = _gaussian(M * K, 3, 1.0)
    var wh = _gaussian(N * K, 9, 0.5)
    var gh = _gaussian(M * N, 5, 1.0)
    var x = Tensor.from_host(xh.copy(), [M, K], STDtype.BF16, ctx)
    var w = Tensor.from_host(wh.copy(), [N, K], STDtype.BF16, ctx)
    var g = Tensor.from_host(gh.copy(), [M, N], STDtype.BF16, ctx)
    var x_h = x.to_host(ctx); var w_h = w.to_host(ctx); var g_h = g.to_host(ctx)

    # tensorwise quantize weight + transpose (both int8, one scalar scale)
    var w_scale = int8_tensorwise_scale(w, ctx)
    var w_8 = int8_encode_tensorwise(w, w_scale, ctx)       # [N,K]
    var w_8t = int8_transpose(w_8, ctx)                     # [K,N]

    # forward: out[M,N] = x @ Wᵀ
    var out = int8_linear_fwd(x, w_8, w_scale, ctx)
    var out_h = out.to_host(ctx)
    var ref_f = List[Float32]()
    for m in range(M):
        for n in range(N):
            var acc: Float32 = 0.0
            for k in range(K): acc += x_h[m * K + k] * w_h[n * K + k]
            ref_f.append(acc)
    var cos_f = _cos(out_h, ref_f)

    # backward: dX[M,K] = grad @ W  (sum_n grad[m,n]*W[n,k])
    var dx = int8_linear_bwd(g, w_8t, w_scale, ctx)
    var dx_h = dx.to_host(ctx)
    var ref_b = List[Float32]()
    for m in range(M):
        for k in range(K):
            var acc: Float32 = 0.0
            for n in range(N): acc += g_h[m * N + n] * w_h[n * K + k]
            ref_b.append(acc)
    var cos_b = _cos(dx_h, ref_b)

    print("W8A8 fwd cosine =", cos_f, "  bwd cosine =", cos_b)
    if cos_f < 0.99:
        print("FAIL: fwd cosine", cos_f); return
    if cos_b < 0.99:
        print("FAIL: bwd cosine", cos_b); return

    # backward NN variant: same dX from the STORED [N,K] orientation (no transpose).
    # Same int8 inputs + scales → the int32 accumulate is exact in both layouts →
    # must match the transpose path BIT-for-bit (gate max_abs == 0), and the bf16
    # reference to the same cosine.
    var dx_nn = int8_linear_bwd_nn(g, w_8, w_scale, ctx)
    var dx_nn_h = dx_nn.to_host(ctx)
    var cos_bn = _cos(dx_nn_h, ref_b)
    var max_diff: Float32 = 0.0
    for i in range(len(dx_h)):
        var d = dx_h[i] - dx_nn_h[i]
        if d < 0: d = -d
        if d > max_diff: max_diff = d
    print("W8A8 bwd_nn cosine =", cos_bn, "  max|bwd - bwd_nn| =", max_diff)
    if cos_bn < 0.99:
        print("FAIL: bwd_nn cosine", cos_bn); return
    if max_diff != 0.0:
        print("FAIL: bwd_nn != bwd (expected bit-identical)"); return

    # F32-boundary NT (pre-transposed W8T) vs NN (stored orientation): same int8
    # operands, exact int32 accumulate in either layout → BIT-IDENTICAL dX
    # (the Klein bwd NT lever's correctness gate, 2026-07-11).
    var g_f32 = Tensor.from_host(g.to_host(ctx), [M, N], STDtype.F32, ctx)
    var dx_nn_f = int8_linear_bwd_nn_f32(g_f32, w_8, w_scale, ctx)
    var dx_nt_f = int8_linear_bwd_nt_f32(g_f32, w_8t, w_scale, ctx)
    var dx_nn_f_h = dx_nn_f.to_host(ctx)
    var dx_nt_f_h = dx_nt_f.to_host(ctx)
    var max_diff_f: Float32 = 0.0
    for i in range(len(dx_nn_f_h)):
        var d = dx_nn_f_h[i] - dx_nt_f_h[i]
        if d < 0: d = -d
        if d > max_diff_f: max_diff_f = d
    print("W8A8 f32 max|bwd_nn_f32 - bwd_nt_f32| =", max_diff_f)
    if max_diff_f != 0.0:
        print("FAIL: bwd_nt_f32 != bwd_nn_f32 (expected bit-identical)"); return

    # bands variant: NT vs NN, band-for-band bit-identical (K = 256 = 128+128).
    var bn = int8_linear_bwd_nn_f32_bands(g_f32, w_8, w_scale, 128, 128, 0, 0, ctx)
    var bt = int8_linear_bwd_nt_f32_bands(g_f32, w_8t, w_scale, 128, 128, 0, 0, ctx)
    var max_diff_b: Float32 = 0.0
    for bi in range(len(bn)):
        var nh = bn[bi][].to_host(ctx)
        var th = bt[bi][].to_host(ctx)
        for i in range(len(nh)):
            var d = nh[i] - th[i]
            if d < 0: d = -d
            if d > max_diff_b: max_diff_b = d
    print("W8A8 f32 bands max|nn - nt| =", max_diff_b)
    if max_diff_b != 0.0:
        print("FAIL: bwd_nt_f32_bands != bwd_nn_f32_bands (expected bit-identical)"); return

    # H3 packed sequence lengths include variable caption rows, so M is not
    # generally GEMM-aligned. Exercise only the two shared wrappers used by the
    # resident H3 training path with a deliberately unaligned row count. The
    # other variants above keep their established aligned coverage.
    comptime MU = 65
    var xuh = _gaussian(MU * K, 13, 1.0)
    var guh = _gaussian(MU * N, 15, 1.0)
    var xu = Tensor.from_host(xuh.copy(), [MU, K], STDtype.BF16, ctx)
    var gu = Tensor.from_host(guh.copy(), [MU, N], STDtype.BF16, ctx)
    var xu_h = xu.to_host(ctx)
    var gu_h = gu.to_host(ctx)
    var out_u = int8_linear_fwd(xu, w_8, w_scale, ctx)
    var dx_u = int8_linear_bwd(gu, w_8t, w_scale, ctx)
    if out_u.shape()[0] != MU or out_u.shape()[1] != N:
        raise Error("W8A8 padded forward returned the wrong shape")
    if dx_u.shape()[0] != MU or dx_u.shape()[1] != K:
        raise Error("W8A8 padded backward returned the wrong shape")
    var out_u_h = out_u.to_host(ctx)
    var dx_u_h = dx_u.to_host(ctx)
    var ref_fu = List[Float32]()
    for m in range(MU):
        for n in range(N):
            var acc: Float32 = 0.0
            for k in range(K):
                acc += xu_h[m * K + k] * w_h[n * K + k]
            ref_fu.append(acc)
    var ref_bu = List[Float32]()
    for m in range(MU):
        for k in range(K):
            var acc: Float32 = 0.0
            for n in range(N):
                acc += gu_h[m * N + n] * w_h[n * K + k]
            ref_bu.append(acc)
    var cos_fu = _cos(out_u_h, ref_fu)
    var cos_bu = _cos(dx_u_h, ref_bu)
    print("W8A8 unaligned M=65 fwd cosine =", cos_fu, "  bwd cosine =", cos_bu)
    if cos_fu < 0.99 or cos_bu < 0.99:
        print("FAIL: unaligned W8A8 padding cosine"); return
    print("PASS: int8 W8A8 fwd+bwd+bwd_nn+nt_f32(+bands) (tensorwise weight) match bf16 (cos>=0.99); nt==nn bit-identical")
