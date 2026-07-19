# ops/tests/bench_bf16_klein_max.mojo — Klein base-GEMM bf16 microbench:
# MAX `linalg.matmul.vendor.blas.matmul` (the ops/linear.mojo path) vs real
# cuBLAS gemmEx (ops/cublas_gemm.mojo shim) on Klein's real GEMM shapes,
# RTX 5080 (sm_120, CUDA 13.1). Same convention as cublas_vs_max_gemm.mojo:
# A row-major [M,K] bf16, B row-major [N,K] bf16, C row-major [M,N] f32,
# C = A @ Bᵀ (transpose_b=True, c_row_major=True). Kernel-only timing.
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build --optimization-level 2 \
#     -I . -I /home/alex/MOJO-libs \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/ops/tests/bench_bf16_klein_max.mojo -o /tmp/bench_bf16_klein
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib:/home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     /tmp/bench_bf16_klein

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.time import perf_counter_ns
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.ops.cublas_gemm import cublas_gemm_bf16_nt

comptime _DYN2 = Layout.row_major(-1, -1)


def _fill_bf16(buf: DeviceBuffer[DType.uint8], n: Int, seed: UInt64, ctx: DeviceContext) raises:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n * 2)
    var bp = host.unsafe_ptr().bitcast[BFloat16]()
    var state = seed
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32((state >> 40) & 0xFFFFFF) / Float32(0x1000000)
        var v = u * 2.0 - 1.0
        bp[i] = v.cast[DType.bfloat16]()
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _max_matmul(
    c_buf: DeviceBuffer[DType.uint8],
    a_buf: DeviceBuffer[DType.uint8],
    b_buf: DeviceBuffer[DType.uint8],
    m: Int, n: Int, k: Int, ctx: DeviceContext,
) raises:
    """Exact ops/linear.mojo bf16 path: matmul transpose_b, c_row_major."""
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
    print("=== ", label, "  M=", m, " N=", n, " K=", k, " ===")
    var a_buf = ctx.enqueue_create_buffer[DType.uint8](m * k * 2)
    var b_buf = ctx.enqueue_create_buffer[DType.uint8](n * k * 2)
    var c_max = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)
    var c_cub = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)
    _fill_bf16(a_buf, m * k, 0x1234567 + UInt64(k), ctx)
    _fill_bf16(b_buf, n * k, 0x89ABCDE + UInt64(n), ctx)

    var flop = 2.0 * Float64(m) * Float64(n) * Float64(k)
    comptime WARMUP = 15
    comptime ITERS = 60

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
        print("  *** PARITY FAIL: cos < 0.999 ***")


def main() raises:
    var ctx = DeviceContext()
    print("Klein base bf16 GEMM: MAX vendor.blas.matmul vs cuBLAS gemmEx")
    print("RTX 5080 (sm_120), bf16 in / F32 accumulate, C = A @ Bᵀ")
    print("5080 bf16 tensor-core peak (fp32-acc, no sparsity) ~ 225 TFLOP/s")
    _bench_shape("qkv/ffB single ", 1536, 12288, 4096, ctx)
    _bench_shape("proj single    ", 1536,  4096, 4096, ctx)
    _bench_shape("ff-in single   ", 1536, 24576, 4096, ctx)
    _bench_shape("qkv batch2     ", 3072, 12288, 4096, ctx)
    _bench_shape("square         ", 4096,  4096, 4096, ctx)
    print("")
    print("done.")
