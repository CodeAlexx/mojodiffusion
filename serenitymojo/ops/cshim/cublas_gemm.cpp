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
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// cuBLASLt native-int8 path (Blackwell sm_120).
//
// WHY: the classic cublasGemmEx int8 IMMA path on CUDA 13.x falls back to an
// Ampere-generation forward-compat kernel on the 5080 (~275 TOPS). cuBLASLt's
// heuristic (cublasLtMatmulAlgoGetHeuristic) lets CUDA 13.1+ pick a *native*
// Blackwell int8 kernel when one exists. We keep the GemmEx path as a fallback
// for heuristic-miss / Lt-unavailable so nothing regresses.
//
// The int8×int8→int32 accumulate is EXACT integer arithmetic (alpha=1,beta=0),
// so the Lt kernel is bit-for-bit equivalent to GemmEx regardless of tiling.
static cublasLtHandle_t g_lt          = nullptr;
static void*            g_lt_ws       = nullptr;   // one shared workspace
static size_t           g_lt_ws_bytes = 32ull * 1024 * 1024;   // 32 MB
static std::mutex       g_lt_mutex;    // guards creation
static std::mutex       g_lt_call_mutex;   // guards the shared workspace / matmul

// Lazily create the LtHandle + workspace (same static/mutex pattern as g_cublas).
static int ensure_lt() {
    if (g_lt) return 0;
    std::lock_guard<std::mutex> lock(g_lt_mutex);
    if (g_lt) return 0;
    cublasStatus_t s = cublasLtCreate(&g_lt);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] cublasLtCreate failed: %d\n", (int)s);
        g_lt = nullptr;
        return (int)s;
    }
    cudaError_t e = cudaMalloc(&g_lt_ws, g_lt_ws_bytes);
    if (e != cudaSuccess) {
        fprintf(stderr, "[serenity_cublas] cudaMalloc(Lt workspace %zu) failed: %d\n",
                g_lt_ws_bytes, (int)e);
        g_lt_ws = nullptr;
        g_lt_ws_bytes = 0;   // heuristic will pick a no-workspace algo
    }
    return 0;
}

// Generic int8×int8→int32 cuBLASLt GEMM in the SAME col-major op/ld convention
// as the cublasGemmEx calls below (m,n,k / lda,ldb,ldc are identical to the
// GemmEx args). alpha=1,beta=0 int32. Returns 0 on success; non-zero means the
// caller should FALL BACK to the cublasGemmEx path (heuristic miss, layout
// rejected, or Lt error). Never fatal — the fallback is always available.
static int lt_gemm_s8s8s32(
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const void* A, int lda,
    const void* B, int ldb,
    void* C, int ldc,
    cudaStream_t stream
) {
    if (ensure_lt() != 0 || g_lt == nullptr) return 1;   // -> fallback

    // Serialize on the shared workspace (single-stream trainer; cheap).
    std::lock_guard<std::mutex> lock(g_lt_call_mutex);

    cublasLtMatmulDesc_t       op   = nullptr;
    cublasLtMatrixLayout_t     Ad   = nullptr, Bd = nullptr, Cd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    int rc = 1;   // default: fall back

    do {
        if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I)
                != CUBLAS_STATUS_SUCCESS) break;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &opA, sizeof(opA));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &opB, sizeof(opB));

        // Layout rows/cols describe the matrix AS STORED (col-major, ld=rows).
        // op(A) is m×k, so the stored A is m×k when OP_N, k×m when OP_T; ditto B.
        uint64_t Ar = (opA == CUBLAS_OP_N) ? (uint64_t)m : (uint64_t)k;
        uint64_t Ac = (opA == CUBLAS_OP_N) ? (uint64_t)k : (uint64_t)m;
        uint64_t Br = (opB == CUBLAS_OP_N) ? (uint64_t)k : (uint64_t)n;
        uint64_t Bc = (opB == CUBLAS_OP_N) ? (uint64_t)n : (uint64_t)k;
        if (cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8I,  Ar, Ac, lda)
                != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8I,  Br, Bc, ldb)
                != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32I, (uint64_t)m, (uint64_t)n, ldc)
                != CUBLAS_STATUS_SUCCESS) break;

        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) break;
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &g_lt_ws_bytes, sizeof(g_lt_ws_bytes));

        cublasLtMatmulHeuristicResult_t heur;
        int returned = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            g_lt, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);
        if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) break;   // -> fallback

        const int32_t alpha = 1, beta = 0;
        cublasStatus_t ms = cublasLtMatmul(
            g_lt, op, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd,
            &heur.algo, g_lt_ws, g_lt_ws_bytes, stream);
        if (ms != CUBLAS_STATUS_SUCCESS) break;                     // -> fallback
        rc = 0;
    } while (0);

    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (Cd)   cublasLtMatrixLayoutDestroy(Cd);
    if (Bd)   cublasLtMatrixLayoutDestroy(Bd);
    if (Ad)   cublasLtMatrixLayoutDestroy(Ad);
    if (op)   cublasLtMatmulDescDestroy(op);
    return rc;
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

// ---------------------------------------------------------------------------
// Native FP8 E4M3 row-major NT GEMM for Blackwell:
//   D[M,N] f32 = A[M,K] fp8 @ B[N,K] fp8^T
//
// Per-row activation and weight scales are applied to this F32 accumulator by
// the caller's fused scale+bias+BF16 output kernel. The installed Blackwell
// cuBLASLt rejects OUTER_VEC_32F for this FP8 layout, while the unscaled native
// GEMM is supported and preserves the identical algebra.
extern "C" int serenity_cublas_gemm_fp8e4m3_f32_rowmajor_nt(
    const void* A, const void* B, void* D,
    int M, int N, int K,
    void* stream
) {
    if (!A || !B || !D) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -1;
    if (ensure_lt() != 0 || g_lt == nullptr) return -2;

    std::lock_guard<std::mutex> lock(g_lt_call_mutex);

    cublasLtMatmulDesc_t       op   = nullptr;
    cublasLtMatrixLayout_t     Ad   = nullptr, Bd = nullptr;
    cublasLtMatrixLayout_t     Cd   = nullptr, Dd = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    int rc = -3;

    do {
        if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)
                != CUBLAS_STATUS_SUCCESS) break;
        cublasOperation_t opA = CUBLAS_OP_T;
        cublasOperation_t opB = CUBLAS_OP_N;
        if (cublasLtMatmulDescSetAttribute(
                op, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA))
                != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatmulDescSetAttribute(
                op, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB))
                != CUBLAS_STATUS_SUCCESS) break;

        // Stored col-major views: weight B[N,K] row-major -> [K,N], and
        // activation A[M,K] row-major -> [K,M].
        if (cublasLtMatrixLayoutCreate(
                &Ad, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)N, K)
                != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(
                &Bd, CUDA_R_8F_E4M3, (uint64_t)K, (uint64_t)M, K)
                != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(
                &Cd, CUDA_R_32F, (uint64_t)N, (uint64_t)M, N)
                != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatrixLayoutCreate(
                &Dd, CUDA_R_32F, (uint64_t)N, (uint64_t)M, N)
                != CUBLAS_STATUS_SUCCESS) break;

        int8_t fast_accum = 0;
        if (cublasLtMatmulDescSetAttribute(
                op, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                &fast_accum, sizeof(fast_accum)) != CUBLAS_STATUS_SUCCESS) break;

        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) break;
        if (cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &g_lt_ws_bytes, sizeof(g_lt_ws_bytes))
                != CUBLAS_STATUS_SUCCESS) break;

        cublasLtMatmulHeuristicResult_t heur;
        int returned = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            g_lt, op, Ad, Bd, Cd, Dd, pref, 1, &heur, &returned);
        if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
            fprintf(stderr,
                    "[serenity_cublas] FP8 heuristic miss "
                    "status=%d returned=%d (M=%d N=%d K=%d)\n",
                    (int)hs, returned, M, N, K);
            rc = -4;
            break;
        }

        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t ms = cublasLtMatmul(
            g_lt, op, &alpha, B, Ad, A, Bd, &beta, D, Cd, D, Dd,
            &heur.algo, g_lt_ws, g_lt_ws_bytes, (cudaStream_t)stream);
        if (ms != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[serenity_cublas] FP8 matmul failed "
                    "status=%d (M=%d N=%d K=%d)\n",
                    (int)ms, M, N, K);
            rc = (int)ms;
            break;
        }
        rc = 0;
    } while (0);

    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (Dd)   cublasLtMatrixLayoutDestroy(Dd);
    if (Cd)   cublasLtMatrixLayoutDestroy(Cd);
    if (Bd)   cublasLtMatrixLayoutDestroy(Bd);
    if (Ad)   cublasLtMatrixLayoutDestroy(Ad);
    if (op)   cublasLtMatmulDescDestroy(op);
    return rc;
}

// ---------------------------------------------------------------------------
// int8 W8A8 GEMM (mirrors SerenityTrainer LinearW8A8 int8_forward_tokenwise:
// res = torch._int_mm(x_8, weight.T)  →  int8×int8 accumulate int32).
//
//   A is row-major [M, K] int8   (x_8, per-token-quantized activation)
//   B is row-major [N, K] int8   (weight [out, in], per-row-quantized)
//   C is row-major [M, N] int32  (raw int32 accumulate; caller scales by
//                                 weight_scale * x_scale → bf16, exactly as the reference trainer)
//   Computes  C = A @ Bᵀ  (== the "transpose_b=True" GEMM).
//
// Same row/col-major mapping as the bf16 path above; only the element types
// (CUDA_R_8I), accumulate/compute (CUDA_R_32I / CUBLAS_COMPUTE_32I), and the
// int32 alpha/beta differ. cuBLAS IMMA int8 kernels require A transposed +
// m/n/k/ld multiples of 4 — satisfied by krea2 dims (features 6144, mlp 16384,
// seq buckets ×4) and asserted by the caller.
extern "C" int serenity_cublas_gemm_s8s8s32_rowmajor_nt(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    void* stream
) {
    if (!A || !B || !C) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -1;
    if ((M & 3) || (N & 3) || (K & 3)) {
        fprintf(stderr, "[serenity_cublas] int8 gemm needs M,N,K %%4==0 "
                "(M=%d N=%d K=%d)\n", M, N, K);
        return -2;
    }

    int rc = ensure_handle();
    if (rc != 0) return rc;

    cublasStatus_t s = cublasSetStream(g_cublas, (cudaStream_t)stream);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] cublasSetStream failed: %d\n", (int)s);
        return (int)s;
    }

    const int32_t alpha = 1;
    const int32_t beta  = 0;

    // Native Blackwell int8 via cuBLASLt heuristic; fall back to GemmEx on miss.
    if (lt_gemm_s8s8s32(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                        B, K, A, K, C, N, (cudaStream_t)stream) == 0) {
        return 0;
    }

    s = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T,            // op on first matrix (B) — IMMA requires the T operand first
        CUBLAS_OP_N,            // op on second matrix (A)
        N,                      // m (rows of output Cᵀ, col-major)
        M,                      // n (cols of output Cᵀ, col-major)
        K,                      // k (shared contraction dim)
        &alpha,
        B, CUDA_R_8I, K,        // A_gemm = B buffer, int8, lda = K
        A, CUDA_R_8I, K,        // B_gemm = A buffer, int8, ldb = K
        &beta,
        C, CUDA_R_32I, N,       // C_gemm = C buffer, int32, ldc = N
        CUBLAS_COMPUTE_32I,     // int32 accumulate (IMMA)
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] int8 cublasGemmEx failed: %d "
                "(M=%d N=%d K=%d)\n", (int)s, M, N, K);
        return (int)s;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// int8 NN GEMM: C[M,K] = A[M,N] @ B[N,K], all row-major, int32 accumulate.
//
// The backward's dX = grad[M,N] @ w8[N,K] contraction — using the STORED weight
// orientation directly, with NO transposed copy. Removes the per-step
// int8_transpose (224 calls/step) AND its transient [K,N] buffer.
//
// Row/col-major mapping: row-major buffers read col-major are transposes:
//   A_buf = Aᵀ (N×M), B_buf = Bᵀ (K×N), C_buf = Cᵀ (K×M).
//   Cᵀ = Bᵀ · Aᵀ  →  gemm_cm(op_X=N on B_buf (K×N), op_Y=N on A_buf (N×M))
//   => cublasGemmEx(OP_N, OP_N, m=K, n=M, k=N, B, lda=K, A, ldb=N, C, ldc=K)
//
// NOTE: classic IMMA tensor-core kernels want the (T,N) col-major layout; on
// newer cuBLAS (12/13) the cutlass-backed int8 path accepts (N,N) too. If this
// arch/toolkit combination rejects it, cuBLAS returns NOT_SUPPORTED (15) and
// the Mojo caller falls back to int8_transpose + the NT GEMM above.
extern "C" int serenity_cublas_gemm_s8s8s32_rowmajor_nn(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    void* stream
) {
    if (!A || !B || !C) return -1;
    if (M <= 0 || N <= 0 || K <= 0) return -1;
    if ((M & 3) || (N & 3) || (K & 3)) {
        fprintf(stderr, "[serenity_cublas] int8 nn gemm needs M,N,K %%4==0 "
                "(M=%d N=%d K=%d)\n", M, N, K);
        return -2;
    }

    int rc = ensure_handle();
    if (rc != 0) return rc;

    cublasStatus_t s = cublasSetStream(g_cublas, (cudaStream_t)stream);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] cublasSetStream failed: %d\n", (int)s);
        return (int)s;
    }

    const int32_t alpha = 1;
    const int32_t beta  = 0;

    // Native Blackwell int8 via cuBLASLt heuristic; fall back to GemmEx on miss.
    // The (N,N) col-major layout is often unsupported by IMMA-style int8 kernels;
    // heuristic will simply return 0 results and we fall through to GemmEx.
    if (lt_gemm_s8s8s32(CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,
                        B, K, A, N, C, K, (cudaStream_t)stream) == 0) {
        return 0;
    }

    s = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_N,            // op on first matrix (B buffer = Bᵀ col-major K×N)
        CUBLAS_OP_N,            // op on second matrix (A buffer = Aᵀ col-major N×M)
        K,                      // m (rows of output Cᵀ, col-major)
        M,                      // n (cols of output Cᵀ, col-major)
        N,                      // k (shared contraction dim)
        &alpha,
        B, CUDA_R_8I, K,        // A_gemm = B buffer, int8, lda = K
        A, CUDA_R_8I, N,        // B_gemm = A buffer, int8, ldb = N
        &beta,
        C, CUDA_R_32I, K,       // C_gemm = C buffer, int32, ldc = K
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[serenity_cublas] int8 nn cublasGemmEx failed: %d "
                "(M=%d N=%d K=%d)\n", (int)s, M, N, K);
        return (int)s;
    }
    return 0;
}
