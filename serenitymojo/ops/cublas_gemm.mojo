# ops/cublas_gemm.mojo — Mojo wrapper for the cuBLAS gemmEx shim.
#
# Calls serenity_cublas_gemm_bf16_rowmajor_nt (ops/cshim/cublas_gemm.cpp) via
# external_call, mirroring the cuDNN SDPA FFI pattern (ops/attention_flash.mojo).
#
# Semantics match ops/linear.mojo's GEMM exactly:
#   A row-major [M, K] bf16  (activation x)
#   B row-major [N, K] bf16  (weight [out, in])
#   C row-major [M, N] f32   (output, F32 accumulate)
#   C = A @ Bᵀ   (the transpose_b=True, c_row_major=True GEMM)
#
# This is the cuBLAS alternative to `linalg.matmul.vendor.blas.matmul` for the
# bf16-A/B → f32-C path. Measurement-only for now; NOT wired into linear.mojo.
#
# Build: link the shim — pass to `mojo build`:
#   -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
# Mojo 1.0.0b1, NVIDIA sm_86+, cuBLAS 12.

from std.ffi import external_call
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host._nvidia_cuda import CUDA
from serenitymojo.io.ffi import BytePtr


def cublas_gemm_bf16_nt(
    a_buf: DeviceBuffer[DType.uint8],   # row-major [M, K] bf16
    b_buf: DeviceBuffer[DType.uint8],   # row-major [N, K] bf16
    c_buf: DeviceBuffer[DType.uint8],   # row-major [M, N] f32 (written)
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """C[M,N] = A[M,K] @ B[N,K]ᵀ via cuBLAS gemmEx (bf16 in, F32 accumulate).
    FAIL-LOUD on any nonzero shim rc."""
    var a_ptr = BytePtr(unsafe_from_address=Int(a_buf.unsafe_ptr()))
    var b_ptr = BytePtr(unsafe_from_address=Int(b_buf.unsafe_ptr()))
    var c_ptr = BytePtr(unsafe_from_address=Int(c_buf.unsafe_ptr()))
    var stream = CUDA(ctx.stream())
    var rc = Int(external_call["serenity_cublas_gemm_bf16_rowmajor_nt", Int32](
        a_ptr, b_ptr, c_ptr,
        Int32(m), Int32(n), Int32(k),
        stream,
    ))
    if rc != 0:
        raise Error(
            String("cublas_gemm_bf16_nt: shim rc=") + String(rc)
            + " (M=" + String(m) + " N=" + String(n) + " K=" + String(k) + ")"
        )
