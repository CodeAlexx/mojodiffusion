#!/usr/bin/env bash
# Build libserenity_cudnn_sdpa.so — cuDNN v9 flash SDPA shim (flame-core port).
# The Pixi environment supplies cuDNN and CUDA development libraries. The
# header-only cuDNN frontend is vendored under vendor/cudnn-frontend.
set -euo pipefail
repo_root=${PIXI_PROJECT_ROOT:?run this build through Pixi}
env_prefix=${CONDA_PREFIX:?Pixi environment prefix is missing}
cd "$repo_root/serenitymojo/ops/cshim"

cudnn_root=${SERENITY_CUDNN_ROOT:-$env_prefix}
frontend=${SERENITY_CUDNN_FRONTEND:-../../../vendor/cudnn-frontend/include}
cuda_root=${SERENITY_CUDA_ROOT:-$env_prefix}
cuda_include=${SERENITY_CUDA_INCLUDE:-$cuda_root/targets/x86_64-linux/include}
pixi_runtime_rpath='$ORIGIN/../../../../.pixi/envs/default/lib'
cudnn_rpath=${SERENITY_CUDNN_ROOT:+$cudnn_root/lib}
cudnn_rpath=${cudnn_rpath:-$pixi_runtime_rpath}
cuda_rpath=${SERENITY_CUDA_ROOT:+$cuda_root/lib}
cuda_rpath=${cuda_rpath:-$pixi_runtime_rpath}

mkdir -p lib/cudnn_stubs
if [[ -z ${SERENITY_CUDNN_ROOT:-} ]]; then
  ln -sfn ../../../../../.pixi/envs/default/lib/libcudnn.so.9 lib/cudnn_stubs/libcudnn.so
else
  ln -sfn "$cudnn_root/lib/libcudnn.so.9" lib/cudnn_stubs/libcudnn.so
fi

"${CXX:-g++}" -shared -fPIC -std=c++17 -O2 \
  -I "$frontend" -I "$cudnn_root/include" -I "$cuda_include" \
  cudnn_sdpa.cpp cudnn_sdpa_bwd.cpp cudnn_conv2d.cpp cudnn_conv3d.cpp cublas_gemm.cpp \
  -L lib/cudnn_stubs -lcudnn -L "$cuda_root/lib" -lcudart -lnvrtc -lcublas -lcublasLt \
  -Wl,-rpath,"$cudnn_rpath" -Wl,-rpath,"$cuda_rpath" \
  -Wno-deprecated-declarations -Wno-unused-parameter -Wno-unused-variable \
  -Wno-sign-compare -Wno-reorder \
  -o lib/libserenity_cudnn_sdpa.so

echo "built: $(ls -la lib/libserenity_cudnn_sdpa.so)"
nm -D lib/libserenity_cudnn_sdpa.so | grep -E 'flame_cudnn|serenity_cudnn|serenity_cublas'
