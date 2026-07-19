#!/usr/bin/env bash
# Build the bf16 GEMM microbench (cublasGemmEx compat vs cuBLASLt heuristic).
# Run: bash build_bench_bf16.sh && ./bin/bench_bf16_gemm
set -euo pipefail
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p bin
g++ -std=c++17 -O2 -I "$CUDA_HOME/include" \
  bench_bf16_gemm.cpp \
  -L "$CUDA_HOME/lib64" -lcudart -lcublas -lcublasLt \
  -Wl,-rpath,"$CUDA_HOME/lib64" \
  -o bin/bench_bf16_gemm
echo "built: bin/bench_bf16_gemm"
