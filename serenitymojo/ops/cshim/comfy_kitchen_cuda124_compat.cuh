// CUDA 12.4 compatibility for Comfy Kitchen v0.2.31's attention-only build.
// Upstream's generic 128-bit half/BF16 load is guarded by CUDA >=12.8 only
// because the same header also defines FP4 helpers. The Sage V quantizer uses
// no FP4 operation, so provide the identical vector load on the older toolkit.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace comfy {

#pragma nv_diag_suppress 1056
template <typename IType>
__forceinline__ __device__ const IType *load_f16x8(const IType *val) {
  float4 vals = *reinterpret_cast<const float4 *>(val);
  return reinterpret_cast<const IType *>(&vals);
}
#pragma nv_diag_default 1056

}  // namespace comfy
