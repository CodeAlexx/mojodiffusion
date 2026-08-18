#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

output=output/bin/minimax_h3_mk2va_512x320x2895_int8_fast

echo "building six-anchor 120-second MiniMax-H3 runner: $output"
MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  scripts/mem_safe_runtime.sh pixi run mojo build \
    --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
    -D H3_HEIGHT=320 -D H3_WIDTH=512 -D H3_FRAMES=2895 \
    -D H3_TEXT_TOKENS=2048 -D H3_KEYFRAMES=6 \
    -D H3_ENCODER_INT8_ALLOW_CACHE_BUILD=0 \
    -D H3_FP8_RESIDENT=1 -D H3_RESIDENT_BLOCKS=2 \
    serenitymojo/pipeline/minimax_h3_i2va.mojo -o "$output" \
    -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
