#!/usr/bin/env bash
# Build the int8 GEMM microbench (cublasGemmEx compat vs cuBLASLt heuristic).
# Standalone — links CUDA cublas/cublasLt directly (reproduces the shim's exact
# op/ld convention). Run: bash build_bench.sh && ./bin/bench_int8_gemm
set -euo pipefail
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p bin
g++ -std=c++17 -O2 -I "$CUDA_HOME/include" \
  bench_int8_gemm.cpp \
  -L "$CUDA_HOME/lib64" -lcudart -lcublas -lcublasLt \
  -Wl,-rpath,"$CUDA_HOME/lib64" \
  -o bin/bench_int8_gemm
echo "built: bin/bench_int8_gemm"
