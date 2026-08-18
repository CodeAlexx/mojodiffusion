#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

output="output/bin/train_minimax_h3"
if [[ -x "$output" && "${H3_REBUILD_TRAINER:-0}" != 1 ]]; then
  echo "already built: $output"
  exit 0
fi

mkdir -p output/bin
echo "building MiniMax-H3 trainer: $output"

# H3's whole-program Mojo compile can exceed 10 GiB. Keep it inside the
# rootless hard-capped build service. The accepted optimized build is O2 with
# one compiler job; O3 is intentionally excluded.
H3_BUILD_MEM_MAX=${H3_BUILD_MEM_MAX:-24G} \
H3_BUILD_MEM_HIGH=${H3_BUILD_MEM_HIGH:-infinity} \
H3_BUILD_SWAP_MAX=${H3_BUILD_SWAP_MAX:-2G} \
MEM_MAX="$H3_BUILD_MEM_MAX" MEM_HIGH="$H3_BUILD_MEM_HIGH" \
SWAP_MAX="$H3_BUILD_SWAP_MAX" \
  scripts/mem_safe_runtime.sh pixi run mojo build \
    --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
    serenitymojo/training/train_minimax_h3.mojo -o "$output" \
    -Xlinker="-L$repo_root/.pixi/envs/default/lib" \
    -Xlinker=-Loutput/lib -Xlinker=-lserenity_minimax_h3_int8 \
    -Xlinker=-Lserenitymojo/ops/cshim/lib \
    -Xlinker=-lserenity_cudnn_sdpa \
    -Xlinker=-Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker=-lcudnn -Xlinker=-lcuda -Xlinker=-lm \
    -Xlinker=-lpng16 -Xlinker=-lturbojpeg \
    -Xlinker=-lsqlite3 -Xlinker=-ldl
