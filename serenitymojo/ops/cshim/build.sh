#!/usr/bin/env bash
# Build libserenity_cudnn_sdpa.so — cuDNN v9 flash SDPA shim (flame-core port).
# Sources cudnn_sdpa{,_bwd}.cpp are byte-copies of
# /home/alex/EriDiffusion/flame-core/src/cuda/cudnn_sdpa{,_bwd}.cpp.
# Headers: flame's vendored cudnn_frontend + pip-wheel cuDNN + /usr/local/cuda.
# pip cuDNN wheels ship only versioned .so.9 — the stub symlink dir gives the
# linker an unversioned libcudnn.so (the flame build.rs trick).
set -euo pipefail
cd "$(dirname "$0")"

CUDNN=/home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn
FRONTEND=/home/alex/EriDiffusion/flame-core/third_party/cudnn_frontend/include
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

mkdir -p lib/cudnn_stubs
ln -sf "$CUDNN/lib/libcudnn.so.9" lib/cudnn_stubs/libcudnn.so

g++ -shared -fPIC -std=c++17 -O2 \
  -I "$FRONTEND" -I "$CUDNN/include" -I "$CUDA_HOME/include" \
  cudnn_sdpa.cpp cudnn_sdpa_bwd.cpp cublas_gemm.cpp \
  -L lib/cudnn_stubs -lcudnn -L "$CUDA_HOME/lib64" -lcudart -lnvrtc -lcublas \
  -Wl,-rpath,"$CUDNN/lib" -Wl,-rpath,"$CUDA_HOME/lib64" \
  -Wno-deprecated-declarations -Wno-unused-parameter -Wno-unused-variable \
  -Wno-sign-compare -Wno-reorder \
  -o lib/libserenity_cudnn_sdpa.so

echo "built: $(ls -la lib/libserenity_cudnn_sdpa.so)"
nm -D lib/libserenity_cudnn_sdpa.so | grep -E 'flame_cudnn|serenity_cublas'
