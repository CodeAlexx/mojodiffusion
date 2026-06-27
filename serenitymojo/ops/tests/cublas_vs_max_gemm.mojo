# ops/tests/cublas_vs_max_gemm.mojo — DECISIVE microbenchmark: real cuBLAS
# gemmEx (ops/cublas_gemm.mojo, via the C shim) vs MAX's
# `linalg.matmul.vendor.blas.matmul` (the current ops/linear.mojo path) on
# krea2's two mlp GEMM shapes, on consumer hardware (RTX 3090 Ti, sm_86).
#
# WHY: krea2 LoRA trainer is compute-bound; GEMMs are ~63% of the step. nsys
# shows the big mlp GEMM running as MAX's bundled cutlass_80 tensorop kernel at
# ~49% of the 3090 Ti's peak, where real cuBLAS (ampere_s16816gemm) hits ~70%
# on the SAME card. This benchmark MEASURES the cuBLAS-vs-MAX speedup + achieved
# TFLOP/s + parity, the decisive measurement before any trainer wiring.
#
# Both paths: A row-major [M,K] bf16, B row-major [N,K] bf16, C row-major [M,N]
# f32, computing C = A @ Bᵀ (matmul transpose_b=True, c_row_major=True — the
# EXACT linear.mojo convention). Inputs are NON-DEGENERATE (deterministic LCG
# fill in [-1,1], bf16-rounded). Kernel-only timing: warmup, then N iters with
# a single synchronize at the end (the GEMM is the only enqueued work per iter).
#
# Shapes (krea2 mlp, M = 4864 = packed seq×batch):
#   mlp_up:   M=4864 K=6144  N=16384
#   mlp_down: M=4864 K=16384 N=6144
#
# Build (link the shim into the test binary):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/ops/tests/cublas_vs_max_gemm.mojo -o /tmp/cublas_vs_max
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     /tmp/cublas_vs_max

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.time import perf_counter_ns
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.ops.cublas_gemm import cublas_gemm_bf16_nt

comptime _DYN2 = Layout.row_major(-1, -1)


# Deterministic non-degenerate host fill: LCG → [-1, 1], staged as bf16 bytes.
def _fill_bf16(buf: DeviceBuffer[DType.uint8], n: Int, seed: UInt64, ctx: DeviceContext) raises:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n * 2)
    var bp = host.unsafe_ptr().bitcast[BFloat16]()
    var state = seed
    for i in range(n):
        # 64-bit LCG (Numerical Recipes constants).
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 40) & 0xFFFFFF) / Float32(0x1000000)  # [0,1)
        var v = u * 2.0 - 1.0                                          # [-1,1)
        bp[i] = v.cast[DType.bfloat16]()
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _max_matmul(
    c_buf: DeviceBuffer[DType.uint8],
    a_buf: DeviceBuffer[DType.uint8],
    b_buf: DeviceBuffer[DType.uint8],
    m: Int, n: Int, k: Int, ctx: DeviceContext,
) raises:
    """The exact ops/linear.mojo bf16 path: matmul transpose_b, c_row_major."""
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, k))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n, k))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](m, n))
    var A = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        a_buf.unsafe_ptr().bitcast[BFloat16](), a_rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        b_buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        c_buf.unsafe_ptr().bitcast[Float32](), c_rl
    )
    matmul(ctx, C, A, B, transpose_b=True, c_row_major=True)


def _cos(a: DeviceBuffer[DType.uint8], b: DeviceBuffer[DType.uint8], n: Int, ctx: DeviceContext) raises -> Float64:
    """Cosine similarity of two f32 device buffers (length n)."""
    var ha = ctx.enqueue_create_host_buffer[DType.uint8](n * 4)
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n * 4)
    ctx.enqueue_copy(dst_buf=ha, src_buf=a)
    ctx.enqueue_copy(dst_buf=hb, src_buf=b)
    ctx.synchronize()
    var pa = ha.unsafe_ptr().bitcast[Float32]()
    var pb = hb.unsafe_ptr().bitcast[Float32]()
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(n):
        var x = Float64(pa[i])
        var y = Float64(pb[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _bench_shape(label: String, m: Int, n: Int, k: Int, ctx: DeviceContext) raises:
    print("")
    print("=== ", label, "  M=", m, " K=", k, " N=", n, " ===")
    var a_buf = ctx.enqueue_create_buffer[DType.uint8](m * k * 2)  # bf16
    var b_buf = ctx.enqueue_create_buffer[DType.uint8](n * k * 2)  # bf16
    var c_max = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)  # f32
    var c_cub = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)  # f32
    _fill_bf16(a_buf, m * k, 0x1234567 + UInt64(k), ctx)
    _fill_bf16(b_buf, n * k, 0x89ABCDE + UInt64(n), ctx)

    var flop = 2.0 * Float64(m) * Float64(n) * Float64(k)  # 2*M*N*K
    comptime WARMUP = 5
    comptime ITERS = 30

    # --- MAX vendor.blas.matmul ---
    for _ in range(WARMUP):
        _max_matmul(c_max, a_buf, b_buf, m, n, k, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        _max_matmul(c_max, a_buf, b_buf, m, n, k, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var max_us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    var max_tflops = flop / (max_us * 1e-6) / 1e12

    # --- cuBLAS gemmEx ---
    for _ in range(WARMUP):
        cublas_gemm_bf16_nt(a_buf, b_buf, c_cub, m, n, k, ctx)
    ctx.synchronize()
    var t2 = perf_counter_ns()
    for _ in range(ITERS):
        cublas_gemm_bf16_nt(a_buf, b_buf, c_cub, m, n, k, ctx)
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var cub_us = Float64(t3 - t2) / Float64(ITERS) / 1000.0
    var cub_tflops = flop / (cub_us * 1e-6) / 1e12

    var cos = _cos(c_max, c_cub, m * n, ctx)
    var speedup = max_us / cub_us

    print("  MAX  vendor.blas : ", max_us, " us/iter   ", max_tflops, " TFLOP/s")
    print("  cuBLAS gemmEx    : ", cub_us, " us/iter   ", cub_tflops, " TFLOP/s")
    print("  speedup (MAX/cuBLAS) = ", speedup, "x")
    print("  parity cos(cuBLAS, MAX) = ", cos, "  (bar >= 0.999)")
    if cos < 0.999:
        print("  *** PARITY FAIL: cos < 0.999 — mapping or dtype bug ***")


def main() raises:
    var ctx = DeviceContext()
    print("cuBLAS gemmEx vs MAX vendor.blas.matmul — krea2 mlp shapes")
    print("RTX 3090 Ti (sm_86), bf16 in / F32 accumulate, C = A @ Bᵀ")
    # peak bf16 tensor-core (no sparsity) for 3090 Ti ≈ 160 TFLOP/s.
    _bench_shape("mlp_up  ", 4864, 16384, 6144, ctx)   # M, N, K
    _bench_shape("mlp_down", 4864, 6144, 16384, ctx)   # M, N, K
    print("")
    print("done.")
