// cublas_gemm.cpp — real cuBLAS gemmEx shim for serenitymojo's linear op.
//
// WHY: serenitymojo's linear op (ops/linear.mojo) dispatches its GEMM through
// `linalg.matmul.vendor.blas.matmul` (MAX's bundled CUTLASS). On consumer
// GPUs (RTX 3090 Ti, sm_86) nsys shows MAX picking a cutlass_80 tensorop
// kernel that lands ~49% of peak on krea2's big mlp shapes, where real cuBLAS
// (ampere_s16816gemm) hits ~70% on the SAME card. This shim exposes
// cublasGemmEx so a microbenchmark can MEASURE cuBLAS-vs-MAX on the exact mlp
// shapes before any trainer wiring (the user's directive: consumer-appropriate
// cuBLAS via FFI, no MAX-backend work).
//
// Entry point exposed to Mojo (mirrors the cudnn_sdpa shim convention):
//
//   int serenity_cublas_gemm_bf16_rowmajor_nt(
//       const void* A, const void* B, void* C,
//       int M, int N, int K,
//       void* stream);
//
// SEMANTICS (matches ops/linear.mojo exactly):
//   A is row-major [M, K] bf16   (the activation x, M = prod(leading), K = in)
//   B is row-major [N, K] bf16   (the weight  [out, in], N = out, K = in)
//   C is row-major [M, N] f32    (output, F32 accumulate — linear.mojo's C buf)
//   Computes  C = A @ Bᵀ         (the "transpose_b=True, c_row_major=True" GEMM)
//   bf16 inputs, F32 accumulate (CUBLAS_COMPUTE_32F), tensor-op math.
//
// ROW-MAJOR ↔ COL-MAJOR MAPPING (the fiddly part):
//   cuBLAS is column-major. A row-major buffer M[r,c] with leading dim c is, in
//   column-major eyes, the transpose Mᵀ. So our row-major buffers, read as
//   col-major, are: A_buf = Aᵀ (K×M col-major), B_buf = Bᵀ (K×N col-major),
//   C_buf = Cᵀ (N×M col-major). We want to produce Cᵀ (the row-major C buffer).
//     Cᵀ = (A·Bᵀ)ᵀ = B·Aᵀ
//   In col-major gemm  Cout = op(X)·op(Y)  with Cout = N×M:
//     X = B_buf (col-major Bᵀ, K×N); op_X = T  -> op(X) = B  (N×K)
//     Y = A_buf (col-major Aᵀ, K×M); op_Y = N  -> op(Y) = Aᵀ (K×M)
//     product (N×K)·(K×M) = N×M = Cᵀ  ✓   (col-major N×M == row-major C[M,N])
//   => cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
//                   m=N, n=M, k=K,
//                   alpha, B, lda=K, A, ldb=K, beta, C, ldc=N)
//   Leading dims are the row-major inner dims: A is [M,K]→ld=K, B is [N,K]→ld=K,
//   C is [M,N]→ld=N (== the col-major Cᵀ leading dim of N rows).
//
// Returns 0 on success, non-zero (cublasStatus_t, or -1 for bad args) on error.
//
// Linux x86-64, CUDA 12.4, NVIDIA sm_86+, cuBLAS 12. Built by build.sh into
// ops/cshim/lib/libserenity_cudnn_sdpa.so (same .so as the SDPA shim).

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <mutex>

// One process-wide cuBLAS handle, lazily created. cuBLAS handles are not
// thread-safe to share across concurrent streams without serialization, but
// the trainer (and this benchmark) drives one stream at a time; we guard
// creation with a mutex and set the stream per call, matching the cuDNN shim.
static cublasHandle_t g_cublas = nullptr;
static std::mutex     g_cublas_mutex;

static int ensure_handle() {
    if (g_cublas) return 0;
    std::lock_guard<std::mutex> lock(g_cublas_mutex);
    if (g_cublas) return 0;
    cublasStatus_t s = cublasCreate(&g_cublas);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] cublasCreate failed: %d\n", (int)s);
        return (int)s;
    }
    // Allow tensor-core math for bf16 GEMMs (the whole point on Ampere).
    cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    return 0;
}

extern "C" int serenity_cublas_gemm_bf16_rowmajor_nt(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    void* stream
) {
    if (!A || !B || !C) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -1;

    int rc = ensure_handle();
    if (rc != 0) return rc;

    cublasStatus_t s = cublasSetStream(g_cublas, (cudaStream_t)stream);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] cublasSetStream failed: %d\n", (int)s);
        return (int)s;
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Produce Cᵀ (col-major N×M, == row-major C[M,N]) = B · Aᵀ.
    //   op_A = CUBLAS_OP_T applied to B_buf (col-major Bᵀ, K×N) -> B (N×K)
    //   op_B = CUBLAS_OP_N applied to A_buf (col-major Aᵀ, K×M) -> Aᵀ (K×M)
    s = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T,            // transa: op on first matrix (B)
        CUBLAS_OP_N,            // transb: op on second matrix (A)
        N,                      // m (rows of output Cᵀ, col-major)
        M,                      // n (cols of output Cᵀ, col-major)
        K,                      // k (shared contraction dim)
        &alpha,
        B, CUDA_R_16BF, K,      // A_gemm = B buffer, type bf16, lda = K
        A, CUDA_R_16BF, K,      // B_gemm = A buffer, type bf16, ldb = K
        &beta,
        C, CUDA_R_32F, N,       // C_gemm = C buffer, type f32, ldc = N
        CUBLAS_COMPUTE_32F,     // F32 accumulate
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] cublasGemmEx failed: %d "
                "(M=%d N=%d K=%d)\n", (int)s, M, N, K);
        return (int)s;
    }
    return 0;
}
