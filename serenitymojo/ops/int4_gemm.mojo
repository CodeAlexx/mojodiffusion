# ops/int4_gemm.mojo — Mojo wrapper for the CUTLASS int4×int4→int32 GEMM shim.
#
# Calls serenity_int4_gemm_s4_nt (ops/cshim/int4_gemm.cu) via external_call,
# mirroring the cuBLAS shim wrapper (ops/cublas_gemm.mojo). Semantics:
#   A packed int4 [M, K]  (K/2 bytes/row)   row-major   (per-token quantized x)
#   B packed int4 [N, K]  (weight [out,in], transpose_b) row-major
#   C int32       [M, N]                    row-major   (caller rescales → bf16)
#   C = A @ Bᵀ   (int4·int4 accumulated in int32).  K must be a multiple of 32.
#
# The compute core of W4A4 (MJ-1099). Build: link the shim —
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_int4_gemm

from std.ffi import external_call
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from serenitymojo.io.ffi import BytePtr


def int4_gemm_s4_nt(
    a_buf: DeviceBuffer[DType.uint8],   # packed int4 [M, K]  (M*K/2 bytes)
    b_buf: DeviceBuffer[DType.uint8],   # packed int4 [N, K]  (N*K/2 bytes)
    c_buf: DeviceBuffer[DType.uint8],   # int32 [M, N]  (M*N*4 bytes, written)
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """C[M,N] = A[M,K] @ B[N,K]ᵀ via CUTLASS int4 tensor-core GEMM (int32 acc).
    FAIL-LOUD on any nonzero shim rc."""
    var a_ptr = BytePtr(unsafe_from_address=Int(a_buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=Int(b_buf.unsafe_ptr()))
    var c_ptr = BytePtr(unsafe_from_address=Int(c_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["serenity_int4_gemm_s4_nt", Int32](
        a_ptr, b_ptr, c_ptr,
        Int32(m), Int32(n), Int32(k),
        stream,
    ))
    if rc != 0:
        raise Error(
            String("int4_gemm_s4_nt: shim rc=") + String(rc)
            + " (M=" + String(m) + " N=" + String(n) + " K=" + String(k) + ")"
        )
