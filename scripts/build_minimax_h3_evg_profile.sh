#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# EVG is deliberately a separate binary. The large sparse executor is behind
# `H3_EVG=1`, so the standard H3 product build does not pay its compile-time or
# code-size cost and cannot select an unvalidated experimental backend.
output="output/bin/minimax_h3_serenity_runtime_evg"
if [[ -x "$output" && "${H3_REBUILD_EVG_PROFILE:-0}" != 1 ]]; then
  echo "already built: $output"
  exit 0
fi

echo "building MiniMax-H3 EVG SM86 experimental runner: $output"
H3_EVG_BUILD_MEM_MAX=${H3_EVG_BUILD_MEM_MAX:-24G} \
H3_EVG_BUILD_MEM_HIGH=${H3_EVG_BUILD_MEM_HIGH:-infinity} \
H3_EVG_BUILD_SWAP_MAX=${H3_EVG_BUILD_SWAP_MAX:-2G} \
MEM_MAX="$H3_EVG_BUILD_MEM_MAX" MEM_HIGH="$H3_EVG_BUILD_MEM_HIGH" \
SWAP_MAX="$H3_EVG_BUILD_SWAP_MAX" \
  scripts/mem_safe_runtime.sh pixi run mojo build \
    --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
    -D H3_TEXT_TOKENS=241 \
    -D H3_VAE_STREAM_DECODE=1 \
    -D H3_FP8_RESIDENT=0 \
    -D H3_EVG=1 \
    serenitymojo/pipeline/minimax_h3_t2va.mojo -o "$output" \
    -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
    -Xlinker -rpath -Xlinker "$repo_root/output/lib" \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -rpath -Xlinker "$repo_root/serenitymojo/ops/cshim/lib" \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
