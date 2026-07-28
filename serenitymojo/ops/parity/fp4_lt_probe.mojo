# ops/parity/fp4_lt_probe.mojo — does the installed cuBLASLt run a native
# block-scaled NVFP4 GEMM on THIS GPU (5080 / sm_120)? (SquareQ chunk 0b.)
#
# Uniform block scales (ue4m3 byte 0x38 == 1.0) make correctness independent of
# cuBLASLt's tiled scale-factor layout, so the probe answers two questions
# cleanly separated:
#   1. does the heuristic return a native algo at all (rc == -4 -> NO)?
#   2. if it runs, does D match a host e2m1 dequant reference (cos >= 0.999)?
# Non-uniform scale LAYOUT verification is deliberately chunk-7 work.
#
# Build:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
#     -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
#     serenitymojo/ops/parity/fp4_lt_probe.mojo -o output/checks/fp4_lt_probe

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.ops.fp4_gemm import fp4_gemm_nt_rc
from serenitymojo.ops.arch import query_gpu_arch

comptime M = 128
comptime N = 128
comptime K = 256
comptime SCALE_ONE: UInt8 = 0x38  # ue4m3: exp bias 7 -> 2^0 * 1.0


def _e2m1_value(code: Int) -> Float64:
    """Decode one e2m1 nibble (sign bit 3, {0,.5,1,1.5,2,3,4,6} magnitudes)."""
    var mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    var v = mags[code & 0x7]
    if (code & 0x8) != 0:
        return -v
    return v


def main() raises:
    var ctx = DeviceContext()
    var arch = query_gpu_arch()
    print("fp4_lt_probe: sm_", arch.sm(), " fp4_tc=", arch.has_fp4_tensorcores())

    # Deterministic pseudo-random e2m1 codes (LCG), packed 2/byte lo=even.
    var a_host = ctx.enqueue_create_host_buffer[DType.uint8](M * K // 2)
    var b_host = ctx.enqueue_create_host_buffer[DType.uint8](N * K // 2)
    var seed: Int = 12345
    var ap = a_host.unsafe_ptr()
    for i in range(M * K // 2):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        ap[i] = UInt8(seed & 0xFF)
    var bp = b_host.unsafe_ptr()
    for i in range(N * K // 2):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        bp[i] = UInt8(seed & 0xFF)

    var as_host = ctx.enqueue_create_host_buffer[DType.uint8](M * K // 16)
    var bs_host = ctx.enqueue_create_host_buffer[DType.uint8](N * K // 16)
    for i in range(M * K // 16):
        as_host.unsafe_ptr()[i] = SCALE_ONE
    for i in range(N * K // 16):
        bs_host.unsafe_ptr()[i] = SCALE_ONE

    var a_dev = ctx.enqueue_create_buffer[DType.uint8](M * K // 2)
    var b_dev = ctx.enqueue_create_buffer[DType.uint8](N * K // 2)
    var as_dev = ctx.enqueue_create_buffer[DType.uint8](M * K // 16)
    var bs_dev = ctx.enqueue_create_buffer[DType.uint8](N * K // 16)
    var d_dev = ctx.enqueue_create_buffer[DType.uint8](M * N * 4)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.enqueue_copy(dst_buf=as_dev, src_buf=as_host)
    ctx.enqueue_copy(dst_buf=bs_dev, src_buf=bs_host)
    ctx.synchronize()

    var rc = fp4_gemm_nt_rc(a_dev, as_dev, b_dev, bs_dev, d_dev, M, N, K, ctx)
    if rc != 0:
        var why: String
        if rc == -4:
            why = "heuristic returned no native sm_120 NVFP4 algo"
        elif rc == -5:
            why = "VEC16_UE4M3 scale mode rejected by MatmulDesc"
        elif rc == -3:
            why = "desc/layout create failed (dtype unsupported?)"
        elif rc == -2:
            why = "cublasLt handle creation failed"
        else:
            why = String("cublasLtMatmul runtime status ") + String(rc)
        print("fp4_lt_probe: UNSUPPORTED rc=", rc, " — ", why)
        return

    ctx.synchronize()
    var d_host = ctx.enqueue_create_host_buffer[DType.uint8](M * N * 4)
    ctx.enqueue_copy(dst_buf=d_host, src_buf=d_dev)
    ctx.synchronize()

    # Host reference: decode nibbles, f64 matmul.
    var dot: Float64 = 0.0
    var nrm_g: Float64 = 0.0
    var nrm_r: Float64 = 0.0
    var max_abs_err: Float64 = 0.0
    var dp = d_host.unsafe_ptr().bitcast[Float32]()
    for mi in range(M):
        for ni in range(N):
            var acc: Float64 = 0.0
            for ki in range(K):
                var abyte = Int(ap[mi * (K // 2) + (ki >> 1)])
                var bbyte = Int(bp[ni * (K // 2) + (ki >> 1)])
                var acode: Int
                var bcode: Int
                if (ki & 1) == 0:
                    acode = abyte & 0xF
                    bcode = bbyte & 0xF
                else:
                    acode = (abyte >> 4) & 0xF
                    bcode = (bbyte >> 4) & 0xF
                acc += _e2m1_value(acode) * _e2m1_value(bcode)
            var got = Float64(dp[mi * N + ni])
            dot += got * acc
            nrm_g += got * got
            nrm_r += acc * acc
            var err = got - acc
            if err < 0:
                err = -err
            if err > max_abs_err:
                max_abs_err = err
    var cos = dot / (sqrt(nrm_g) * sqrt(nrm_r) + 1e-30)
    print(
        "fp4_lt_probe: ran. cos=", cos, " max_abs_err=", max_abs_err,
        " (M=", M, " N=", N, " K=", K, ")"
    )
    if cos >= 0.999:
        print("fp4_lt_probe: SUPPORTED — native NVFP4 GEMM verified on sm_120")
    else:
        print("fp4_lt_probe: RAN-BUT-WRONG — algo exists, layout/scale mismatch")
