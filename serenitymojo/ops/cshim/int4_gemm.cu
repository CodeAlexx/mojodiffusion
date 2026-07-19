// int4_gemm.cu — CUTLASS int4×int4 → int32 tensor-core GEMM shim (Ampere sm_80/86).
//
// The compute foundation of the W4A4 path (MJ-1099). Mirrors the cublas_gemm.cpp
// FFI convention: A row-major [M,K], B row-major [N,K] read as ColumnMajor →
// C = A @ Bᵀ (transpose_b=True), the exact ops/linear.mojo GEMM shape. Operands
// are packed int4 (2 nibbles/byte, signed [-8,7]); accumulate + output are int32
// (the caller rescales int32 → bf16 by xscale⊗wscale in Mojo).
//
// Kernel: cutlass::gemm::device::Gemm<int4b_t,RowMajor, int4b_t,ColumnMajor,
//   int32,RowMajor, int32 acc, OpClassTensorOp, Sm80, TB<128,128,256>,
//   Warp<64,64,256>, Inst<16,8,64>, ...> — the mma.sync.m16n8k64.s4 path. Ref:
//   pytorch/third_party/cutlass/test/unit/gemm/device/
//   gemm_s4t_s4n_s4t_tensor_op_s32_sm80.cu (value-verified in CUTLASS's own tests).
//
// Entry (extern "C", mirrors serenity_cublas_gemm_bf16_rowmajor_nt):
//   int serenity_int4_gemm_s4_nt(const void* A, const void* B, void* C,
//                                int M, int N, int K, void* stream);
//   A: packed int4 [M, K]   (K nibbles/row, K/2 bytes/row)   row-major
//   B: packed int4 [N, K]   (weight [out,in], transpose_b)   row-major
//   C: int32       [M, N]                                    row-major
//   Computes C = A @ Bᵀ (int4·int4 accumulated in int32). K must be a multiple
//   of 32 (int4 tensor-core crosswise). Returns 0 on success, cutlass::Status
//   (or -1 bad-args) otherwise.
//
// Built by build.sh via nvcc -arch=sm_86 into libserenity_int4_gemm.so.

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime_api.h>

using ElementA = cutlass::int4b_t;
using ElementB = cutlass::int4b_t;
using ElementOut = int32_t;
using ElementAcc = int32_t;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementOut, cutlass::layout::RowMajor,
    ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 256>,
    cutlass::gemm::GemmShape<64, 64, 256>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOut, 128 / cutlass::sizeof_bits<ElementOut>::value,
        ElementAcc, ElementAcc>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;

extern "C" int serenity_int4_gemm_s4_nt(
    const void* A, const void* B, void* C,
    int M, int N, int K, void* stream) {
    if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0) return -1;

    cutlass::gemm::GemmCoord problem(M, N, K);
    // A [M,K] RowMajor (lda=K), B [N,K] read ColumnMajor (ldb=K) → Bᵀ = [K,N],
    // C [M,N] RowMajor (ldc=N). Leading dims are in ELEMENTS.
    typename Gemm::Arguments args(
        problem,
        {reinterpret_cast<ElementA const*>(A), K},
        {reinterpret_cast<ElementB const*>(B), K},
        {reinterpret_cast<ElementOut*>(C), N},
        {reinterpret_cast<ElementOut*>(C), N},
        {1, 0});  // alpha=1, beta=0 (pure product; rescale happens in Mojo)

    Gemm gemm_op;
    cutlass::Status s = gemm_op.can_implement(args);
    if (s != cutlass::Status::kSuccess) return 100 + static_cast<int>(s);
    s = gemm_op.initialize(args, nullptr, static_cast<cudaStream_t>(stream));
    if (s != cutlass::Status::kSuccess) return 200 + static_cast<int>(s);
    s = gemm_op(static_cast<cudaStream_t>(stream));
    if (s != cutlass::Status::kSuccess) return 300 + static_cast<int>(s);
    return 0;
}
