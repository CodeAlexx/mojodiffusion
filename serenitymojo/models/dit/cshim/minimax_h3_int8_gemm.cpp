// MiniMax-H3 row-major INT8 GEMM shim.
//
// H3's product sequence has M=9145 rows. The shared W8A8 shim intentionally
// rejects M%4!=0 because its original Krea callers use aligned sequence
// buckets, but cuBLAS only needs the contraction/output widths aligned for
// this (T,N) mapping. This model-scoped entry preserves the same arithmetic
// while accepting arbitrary positive M.

#include <cstdint>
#include <cstdio>
#include <mutex>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace {
cublasHandle_t handle = nullptr;
std::once_flag handle_once;
int handle_rc = 0;

int ensure_handle() {
  std::call_once(handle_once, [] {
    auto status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      handle_rc = static_cast<int>(status);
    }
  });
  return handle_rc;
}
}  // namespace

extern "C" int serenity_minimax_h3_gemm_s8s8s32_rowmajor_nt(
    const void* a, const void* b, void* c,
    int m, int n, int k, void* stream) {
  if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return -1;
  if ((n & 3) || (k & 3)) return -2;
  int rc = ensure_handle();
  if (rc != 0) return rc;

  auto status = cublasSetStream(handle, static_cast<cudaStream_t>(stream));
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  const int32_t alpha = 1;
  const int32_t beta = 0;
  status = cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      b,
      CUDA_R_8I,
      k,
      a,
      CUDA_R_8I,
      k,
      &beta,
      c,
      CUDA_R_32I,
      n,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(
        stderr,
        "[minimax-h3] int8 GEMM failed status=%d M=%d N=%d K=%d\n",
        static_cast<int>(status), m, n, k);
  }
  return static_cast<int>(status);
}
