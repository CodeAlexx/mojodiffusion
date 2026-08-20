#!/usr/bin/env bash
# Build only the three Comfy Kitchen v0.2.31 Sage attention launchers for the
# local RTX 3090 Ti (sm_86). This avoids the Python/nanobind dependency and the
# startup cost of registering the full 169-MiB multi-op wheel extension.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tag=v0.2.31
commit=7c6ca3a5b63857d42c2d49777d6afb69de23f13f
vendor="$repo_root/output/vendor/comfy-kitchen-$tag"
build="$repo_root/output/lib/ck_minimal_build"
output="$repo_root/output/lib/libserenity_ck_attention.so"
nvcc=${SERENITY_CK_NVCC:-/usr/local/cuda-12.4/bin/nvcc}

if [[ ! -d "$vendor/.git" ]]; then
  mkdir -p "$repo_root/output/vendor"
  git clone --quiet --depth 1 --branch "$tag" \
    https://github.com/Comfy-Org/comfy-kitchen.git "$vendor"
fi
actual=$(git -C "$vendor" rev-parse HEAD)
if [[ "$actual" != "$commit" ]]; then
  echo "Comfy Kitchen checkout mismatch: expected $commit, got $actual" >&2
  exit 2
fi
if [[ ! -x "$nvcc" ]]; then
  echo "nvcc not found: $nvcc" >&2
  exit 2
fi

cuda_src="$vendor/comfy_kitchen/backends/cuda"
sage_src="$cuda_src/sage_attention"
mkdir -p "$build" "$(dirname "$output")"
common=(
  -I "$cuda_src" -I "$sage_src"
  -std=c++17 -O3 --use_fast_math
  --expt-relaxed-constexpr --expt-extended-lambda
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__
  -U__CUDA_NO_BFLOAT16_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
  -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT162_CONVERSIONS__
  -Xcompiler=-fPIC -gencode arch=compute_86,code=sm_86
)

"$nvcc" -c "$sage_src/quant_qk_int8.cu" \
  -o "$build/quant_qk_int8.o" "${common[@]}"
"$nvcc" -c "$sage_src/quant_v_int8.cu" \
  -include "$repo_root/serenitymojo/ops/cshim/comfy_kitchen_cuda124_compat.cuh" \
  -o "$build/quant_v_int8.o" "${common[@]}"
"$nvcc" -c "$sage_src/sage_attn_launcher.cu" \
  -o "$build/sage_attn_launcher.o" "${common[@]}"
"$nvcc" -shared -Xcompiler=-fPIC \
  "$build/quant_qk_int8.o" "$build/quant_v_int8.o" \
  "$build/sage_attn_launcher.o" -o "$output"

echo "built: $output"
ls -lh "$output"
nm -D "$output" | grep -E 'launch_(quant_qk|quant_v|sage_attn)'
