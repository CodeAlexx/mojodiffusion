#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p output/bin

pixi run mojo build --optimization-level 2 -j 1 \
    -I . -I vendor/mojo-libs \
    serenitymojo/pipeline/wan22_encode_prompt.mojo \
    -o output/bin/wan22_encode_prompt

pixi run mojo build --optimization-level 2 -j 1 \
    -I . -I vendor/mojo-libs \
    serenitymojo/pipeline/wan22_prepare_fp8_cache.mojo \
    -o output/bin/wan22_prepare_fp8_cache

pixi run mojo build --optimization-level 2 -j 1 \
    -I . -I vendor/mojo-libs \
    -Xlinker -lm \
    -Xlinker -lcuda \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn \
    -Xlinker -rpath \
    -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
    -Xlinker -rpath \
    -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
    serenitymojo/pipeline/wan22_t2v.mojo \
    -o output/bin/wan22_t2v

pixi run mojo build --optimization-level 2 -j 1 \
    -D WAN22_WIDTH=480 \
    -D WAN22_HEIGHT=832 \
    -D WAN22_FRAMES=121 \
    -I . -I vendor/mojo-libs \
    -Xlinker -lm \
    -Xlinker -lcuda \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn \
    -Xlinker -rpath \
    -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
    -Xlinker -rpath \
    -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
    serenitymojo/pipeline/wan22_t2v.mojo \
    -o output/bin/wan22_t2v_480x832

printf '%s\n' \
    "Built output/bin/wan22_encode_prompt" \
    "Built output/bin/wan22_prepare_fp8_cache" \
    "Built output/bin/wan22_t2v" \
    "Built output/bin/wan22_t2v_480x832"
