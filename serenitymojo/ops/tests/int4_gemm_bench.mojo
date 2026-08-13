# ops/tests/int4_gemm_bench.mojo — THE W4A4 speed gate (MJ-1099).
#
# CUTLASS int4×int4→int32 tensor-core GEMM (ops/int4_gemm.mojo, via int4_gemm.cu)
# vs cuBLAS bf16 gemmEx (ops/cublas_gemm.mojo) on the DOMINANT LTX-2.3 ff shapes.
# The decisive measurement before building the rest of W4A4 (activation quant +
# rotation): does int4 actually deliver ~2x bf16 on THIS card (3090 Ti sm_86)?
# If it doesn't, W4A4's whole point (halve the GEMM) evaporates → stop.
#
# Values are irrelevant to timing → uninitialized buffers. FLOP = 2*M*N*K.
#
# Build (link BOTH shims):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -O2 -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -lserenity_int4_gemm \
#     serenitymojo/ops/tests/int4_gemm_bench.mojo -o /tmp/int4_gemm_bench
# Run:
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:.pixi/envs/default/lib /tmp/int4_gemm_bench

from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from serenitymojo.ops.cublas_gemm import cublas_gemm_bf16_nt
from serenitymojo.ops.int4_gemm import int4_gemm_s4_nt


def _fill_byte(buf: DeviceBuffer[DType.uint8], nbytes: Int, val: UInt8, ctx: DeviceContext) raises:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var p = host.unsafe_ptr()
    for i in range(nbytes):
        p[i] = val
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _read_i32(buf: DeviceBuffer[DType.uint8], count: Int, ctx: DeviceContext) raises -> List[Int32]:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](count * 4)
    ctx.enqueue_copy(dst_buf=host, src_buf=buf)
    ctx.synchronize()
    var p = host.unsafe_ptr().bitcast[Int32]()
    var out = List[Int32]()
    for i in range(count):
        out.append(p[i])
    return out^


def _verify(ctx: DeviceContext) raises:
    """Correctness gate: int4×int4→int32 with KNOWN packed patterns. Confirms the
    lda/ldb/ldc + nibble packing convention matches CUTLASS (all-ones→K, ones×-1→
    -K, signed handling). Bit-exact expected."""
    var m = 32; var n = 32; var k = 64
    var a4 = ctx.enqueue_create_buffer[DType.uint8](m * k // 2)
    var b4 = ctx.enqueue_create_buffer[DType.uint8](n * k // 2)
    var c32 = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)

    # (1) A=+1 (nibble 0x1, byte 0x11), B=+1 → every C[m,n] = sum_k 1*1 = K.
    _fill_byte(a4, m * k // 2, 0x11, ctx)
    _fill_byte(b4, n * k // 2, 0x11, ctx)
    int4_gemm_s4_nt(a4, b4, c32, m, n, k, ctx)
    var c1 = _read_i32(c32, m * n, ctx)
    var ok1 = c1[0] == Int32(k) and c1[m * n - 1] == Int32(k)

    # (2) A=+1, B=-1 (two's-complement int4 0xF, byte 0xFF) → every C = -K.
    _fill_byte(b4, n * k // 2, 0xFF, ctx)
    int4_gemm_s4_nt(a4, b4, c32, m, n, k, ctx)
    var c2 = _read_i32(c32, m * n, ctx)
    var ok2 = c2[0] == Int32(-k) and c2[m * n - 1] == Int32(-k)

    print("[verify] all-ones  C[0]=", c1[0], " C[last]=", c1[m*n-1], " (expect ", k, ")  ", "OK" if ok1 else "FAIL")
    print("[verify] ones×(-1) C[0]=", c2[0], " C[last]=", c2[m*n-1], " (expect ", -k, ")  ", "OK" if ok2 else "FAIL")
    if not (ok1 and ok2):
        raise Error("int4 GEMM correctness FAILED — packing/layout mismatch")
    print("[verify] int4 GEMM correctness PASS (packing + signed accumulate)")


def _bench(label: String, m: Int, n: Int, k: Int, ctx: DeviceContext) raises:
    # int4 operands: A [M,K] and B [N,K] packed 2 nibbles/byte; C int32 [M,N].
    var a4 = ctx.enqueue_create_buffer[DType.uint8](m * k // 2)
    var b4 = ctx.enqueue_create_buffer[DType.uint8](n * k // 2)
    var c32 = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)
    # bf16 operands for the cuBLAS baseline.
    var abf = ctx.enqueue_create_buffer[DType.uint8](m * k * 2)
    var bbf = ctx.enqueue_create_buffer[DType.uint8](n * k * 2)
    var cf = ctx.enqueue_create_buffer[DType.uint8](m * n * 4)

    var flop = 2.0 * Float64(m) * Float64(n) * Float64(k)
    comptime WARMUP = 5
    comptime ITERS = 40

    # --- cuBLAS bf16 ---
    for _ in range(WARMUP):
        cublas_gemm_bf16_nt(abf, bbf, cf, m, n, k, ctx)
    ctx.synchronize()
    var b0 = perf_counter_ns()
    for _ in range(ITERS):
        cublas_gemm_bf16_nt(abf, bbf, cf, m, n, k, ctx)
    ctx.synchronize()
    var b1 = perf_counter_ns()
    var bf_us = Float64(b1 - b0) / Float64(ITERS) / 1000.0
    var bf_tflops = flop / (bf_us * 1e-6) / 1e12

    # --- CUTLASS int4 ---
    for _ in range(WARMUP):
        int4_gemm_s4_nt(a4, b4, c32, m, n, k, ctx)
    ctx.synchronize()
    var i0 = perf_counter_ns()
    for _ in range(ITERS):
        int4_gemm_s4_nt(a4, b4, c32, m, n, k, ctx)
    ctx.synchronize()
    var i1 = perf_counter_ns()
    var i4_us = Float64(i1 - i0) / Float64(ITERS) / 1000.0
    var i4_tflops = flop / (i4_us * 1e-6) / 1e12

    print("")
    print("=== ", label, "  M=", m, " K=", k, " N=", n, " ===")
    print("  cuBLAS bf16 : ", bf_us, " us   ", bf_tflops, " TOP/s")
    print("  CUTLASS int4: ", i4_us, " us   ", i4_tflops, " TOP/s")
    print("  int4 speedup (bf16/int4) = ", bf_us / i4_us, "x")


def main() raises:
    var ctx = DeviceContext()
    print("W4A4 SPEED GATE — CUTLASS int4 vs cuBLAS bf16 (RTX 3090 Ti sm_86)")
    print("bar: int4 must be ~2x bf16 to justify the rest of W4A4")
    _verify(ctx)   # correctness FIRST — a fast wrong kernel is worthless
    _bench("ff.net.0.proj S1", 1536, 16384, 4096, ctx)   # M,N,K  (up-proj)
    _bench("ff.net.0.proj S2", 6144, 16384, 4096, ctx)
    _bench("ff.net.2      S1", 1536, 4096, 16384, ctx)    # down-proj
    print("")
    print("done.")
