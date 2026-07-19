#!/usr/bin/env bash
# Build libserenity_int4_gemm.so — CUTLASS int4×int4→int32 tensor-core GEMM shim
# (int4_gemm.cu). Separate .so (nvcc, not the g++ cudnn/cublas shim) because it
# needs CUTLASS headers + device code compiled for Ampere sm_86.
set -euo pipefail
cd "$(dirname "$0")"

CUTLASS=/home/alex/pytorch/third_party/cutlass/include
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p lib

nvcc -shared -Xcompiler -fPIC -std=c++17 -arch=sm_86 -O3 \
  --expt-relaxed-constexpr \
  -I "$CUTLASS" -I "$CUDA_HOME/include" \
  int4_gemm.cu \
  -L "$CUDA_HOME/lib64" -lcudart \
  -o lib/libserenity_int4_gemm.so

echo "built: $(ls -la lib/libserenity_int4_gemm.so)"
nm -D lib/libserenity_int4_gemm.so | grep serenity_int4_gemm
