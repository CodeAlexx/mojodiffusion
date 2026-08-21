#!/usr/bin/env bash
# Build only the three Comfy Kitchen v0.2.31 Sage attention launchers for one
# explicit CUDA architecture. This avoids the Python/nanobind dependency and
# the startup cost of registering the full 169-MiB multi-op wheel extension.
#
# The output is architecture-tagged both in its path and through exported ABI
# metadata. The runtime will not load it on a different GPU. Set
# SERENITY_CK_CUDA_ARCH=sm_89 (or 8.9/89) for a non-local target; otherwise a
# single locally visible GPU compute capability is detected with nvidia-smi.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tag=v0.2.31
commit=7c6ca3a5b63857d42c2d49777d6afb69de23f13f
vendor="$repo_root/output/vendor/comfy-kitchen-$tag"
nvcc=${SERENITY_CK_NVCC:-/usr/local/cuda-12.4/bin/nvcc}

raw_arch=${SERENITY_CK_CUDA_ARCH:-}
if [[ -z "$raw_arch" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; set SERENITY_CK_CUDA_ARCH explicitly" >&2
    exit 2
  fi
  mapfile -t detected_arches < <(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits \
      | tr -d ' ' | sort -u
  )
  if [[ ${#detected_arches[@]} -ne 1 || -z ${detected_arches[0]:-} ]]; then
    echo "could not select one CUDA architecture; set SERENITY_CK_CUDA_ARCH explicitly" >&2
    printf 'detected compute capabilities: %s\n' "${detected_arches[*]:-none}" >&2
    exit 2
  fi
  raw_arch=${detected_arches[0]}
fi

arch=${raw_arch#sm_}
arch=${arch#compute_}
arch=${arch//./}
if [[ ! "$arch" =~ ^[0-9]+$ ]] || (( 10#$arch < 80 )); then
  echo "CK INT8 attention requires an NVIDIA SM80-or-newer target; got '$raw_arch'" >&2
  exit 2
fi

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
if ! "$nvcc" --list-gpu-code | grep -qx "sm_$arch"; then
  echo "$nvcc cannot compile sm_$arch; choose a CUDA toolkit that lists this target" >&2
  exit 2
fi

build="$repo_root/output/lib/ck_minimal_build/sm$arch"
output=${SERENITY_CK_OUTPUT:-"$repo_root/output/lib/ck/sm$arch/libserenity_ck_attention.so"}

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
  -Xcompiler=-fPIC -gencode arch=compute_$arch,code=sm_$arch
)

"$nvcc" -c "$sage_src/quant_qk_int8.cu" \
  -o "$build/quant_qk_int8.o" "${common[@]}"
"$nvcc" -c "$sage_src/quant_v_int8.cu" \
  -include "$repo_root/serenitymojo/ops/cshim/comfy_kitchen_cuda124_compat.cuh" \
  -o "$build/quant_v_int8.o" "${common[@]}"
"$nvcc" -c "$sage_src/sage_attn_launcher.cu" \
  -o "$build/sage_attn_launcher.o" "${common[@]}"
"$nvcc" -c "$repo_root/serenitymojo/ops/cshim/comfy_kitchen_arch.cpp" \
  -DSERENITY_CK_TARGET_SM="$arch" \
  -o "$build/comfy_kitchen_arch.o" -std=c++17 -O2 -Xcompiler=-fPIC
"$nvcc" -shared -Xcompiler=-fPIC \
  "$build/quant_qk_int8.o" "$build/quant_v_int8.o" \
  "$build/sage_attn_launcher.o" "$build/comfy_kitchen_arch.o" -o "$output"

echo "built CK attention for sm_$arch: $output"
ls -lh "$output"
nm -D "$output" | grep -E 'launch_(quant_qk|quant_v|sage_attn)|serenity_ck_attention_(abi_version|target_sm)'
