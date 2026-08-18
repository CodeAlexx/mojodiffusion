#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

native_width=${H3_NATIVE_WIDTH:-512}
native_height=${H3_NATIVE_HEIGHT:-320}
dit_precision=${H3_DIT_PRECISION:-int8}
case "$dit_precision" in
  int8)
    fp8_resident=1
    resident_blocks=${H3_RESIDENT_BLOCKS:-2}
    default_output="output/bin/minimax_h3_mk2va_${native_width}x${native_height}x1450_int8_ultravico"
    ;;
  bf16)
    fp8_resident=0
    resident_blocks=0
    default_output="output/bin/minimax_h3_mk2va_${native_width}x${native_height}x1450_bf16"
    ;;
  *)
    echo "H3_DIT_PRECISION must be int8 or bf16" >&2
    exit 64
    ;;
esac
output=${H3_RUNNER_OUTPUT:-$default_output}

echo "building six-anchor 60-second MiniMax-H3 runner: $output"
echo "native_geometry=${native_width}x${native_height}x1450"
echo "dit_precision=$dit_precision resident_blocks=$resident_blocks"
MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  scripts/mem_safe_runtime.sh pixi run mojo build \
    --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
    -D H3_HEIGHT="$native_height" -D H3_WIDTH="$native_width" -D H3_FRAMES=1450 \
    -D H3_TEXT_TOKENS=2048 -D H3_KEYFRAMES=6 \
    -D H3_ENCODER_INT8_ALLOW_CACHE_BUILD=0 \
    -D H3_FP8_RESIDENT="$fp8_resident" -D H3_RESIDENT_BLOCKS="$resident_blocks" \
    serenitymojo/pipeline/minimax_h3_i2va.mojo -o "$output" \
    -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
