# ops/fp4_gemm.mojo — Mojo wrapper for the cuBLASLt NVFP4 block-scaled GEMM.
#
# Calls serenity_lt_fp4_gemm_nt (ops/cshim/cublas_gemm.cpp) via external_call,
# mirroring ops/int4_gemm.mojo. Semantics (row-major NT, like the fp8 entry):
#   A packed fp4 e2m1 [M, K] (K/2 bytes/row) + Ascale ue4m3 [M, K/16]
#   B packed fp4 e2m1 [N, K] (K/2 bytes/row) + Bscale ue4m3 [N, K/16]
#   D f32 [M, N] = dequant(A) @ dequant(B)ᵀ.  K must be a multiple of 32.
#
# Distinct shim rcs (see the cpp): -4 = heuristic miss (no native sm_120
# algo), -5 = scale-mode attr rejected, -3 = desc/layout create failed. The
# probe binary translates these into a SUPPORTED/UNSUPPORTED verdict; the
# production path (chunk 7) fail-louds through fp4_gemm_nt below.

from std.ffi import external_call
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host._nvidia_cuda import CUDA
from serenitymojo.io.ffi import BytePtr


def fp4_gemm_nt_rc(
    a_buf: DeviceBuffer[DType.uint8],   # packed e2m1 [M, K]  (M*K/2 bytes)
    a_scale: DeviceBuffer[DType.uint8],  # ue4m3 [M, K/16]
    b_buf: DeviceBuffer[DType.uint8],   # packed e2m1 [N, K]  (N*K/2 bytes)
    b_scale: DeviceBuffer[DType.uint8],  # ue4m3 [N, K/16]
    d_buf: DeviceBuffer[DType.uint8],   # f32 [M, N]  (M*N*4 bytes, written)
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
    alpha: Float32 = 1.0,   # combined per-tensor global scale (a_g * b_g)
) raises -> Int:
    """Raw shim call; returns the shim rc (0 = OK). Probe entry point."""
    var a_ptr = BytePtr(unsafe_from_address=Int(a_buf.unsafe_ptr()))
    var as_ptr = BytePtr(unsafe_from_address=Int(a_scale.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=Int(b_buf.unsafe_ptr()))
    var bs_ptr = BytePtr(unsafe_from_address=Int(b_scale.unsafe_ptr()))
    var d_ptr = BytePtr(unsafe_from_address=Int(d_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    return Int(external_call["serenity_lt_fp4_gemm_nt", Int32](
        a_ptr, as_ptr, b_ptr, bs_ptr, d_ptr,
        Int32(m), Int32(n), Int32(k),
        alpha,
        stream,
    ))


def fp4_gemm_nt(
    a_buf: DeviceBuffer[DType.uint8],
    a_scale: DeviceBuffer[DType.uint8],
    b_buf: DeviceBuffer[DType.uint8],
    b_scale: DeviceBuffer[DType.uint8],
    d_buf: DeviceBuffer[DType.uint8],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    """D[M,N] f32 = alpha * A @ Bᵀ, NVFP4 block-16 ue4m3 scales. FAIL-LOUD."""
    var rc = fp4_gemm_nt_rc(a_buf, a_scale, b_buf, b_scale, d_buf, m, n, k, ctx, alpha)
    if rc != 0:
        raise Error(
            String("fp4_gemm_nt: shim rc=") + String(rc)
            + " (M=" + String(m) + " N=" + String(n) + " K=" + String(k) + ")"
        )
