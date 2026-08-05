#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# One executable owns every admitted T2VA geometry and precision. The three
# sequence-length-specific attention kernels remain AOT specializations inside
# that executable; the request selects width/height/frames/FPS/quant at runtime.
output="output/bin/minimax_h3_serenity_runtime"
if [[ -x "$output" && "${H3_REBUILD_PROFILES:-0}" != 1 ]]; then
  echo "already built: $output"
  exit 0
fi

echo "building unified MiniMax-H3 request runner: $output"
H3_BUILD_MEM_MAX=${H3_BUILD_MEM_MAX:-12G} \
H3_BUILD_MEM_HIGH=${H3_BUILD_MEM_HIGH:-10G} \
H3_BUILD_SWAP_MAX=${H3_BUILD_SWAP_MAX:-2G} \
MEM_MAX="$H3_BUILD_MEM_MAX" MEM_HIGH="$H3_BUILD_MEM_HIGH" \
SWAP_MAX="$H3_BUILD_SWAP_MAX" \
  pixi run scripts/mem_safe.sh mojo build \
    --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
    -D H3_TEXT_TOKENS=241 \
    -D H3_VAE_STREAM_DECODE=1 \
    -D H3_FP8_RESIDENT=0 \
    serenitymojo/pipeline/minimax_h3_t2va.mojo -o "$output" \
    -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
