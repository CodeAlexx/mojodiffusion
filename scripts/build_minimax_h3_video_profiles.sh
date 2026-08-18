#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# One executable owns the full H3 runtime geometry envelope and every
# precision mode. Width, height, internal/output frames, FPS, quantization,
# and attention backend are runtime values; registered profiles are retained
# only as measured quality/performance anchors.
output="output/bin/minimax_h3_serenity_runtime"
if [[ -x "$output" && "${H3_REBUILD_PROFILES:-0}" != 1 ]]; then
  echo "already built: $output"
  exit 0
fi

echo "building unified MiniMax-H3 request runner: $output"
# H3's whole-program Mojo compile can exceed 10 GiB. Keep the accepted O2/-j1
# shape inside the rootless large-job service: hard cap, OOM group, desktop
# reserve admission, and no sudo/password handoff.
H3_BUILD_MEM_MAX=${H3_BUILD_MEM_MAX:-24G} \
H3_BUILD_MEM_HIGH=${H3_BUILD_MEM_HIGH:-infinity} \
H3_BUILD_SWAP_MAX=${H3_BUILD_SWAP_MAX:-2G} \
MEM_MAX="$H3_BUILD_MEM_MAX" MEM_HIGH="$H3_BUILD_MEM_HIGH" \
SWAP_MAX="$H3_BUILD_SWAP_MAX" \
  scripts/mem_safe_runtime.sh pixi run mojo build \
    --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
    -D H3_TEXT_TOKENS=241 \
    -D H3_VAE_STREAM_DECODE=1 \
    -D H3_FP8_RESIDENT=0 \
    serenitymojo/pipeline/minimax_h3_t2va.mojo -o "$output" \
    -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
