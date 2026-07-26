#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner="$repo_root/output/checks/ltx2_conditioning_mask_parity"

mkdir -p "$repo_root/output/checks"
cd "$repo_root"
pixi run mojo build -O0 -j 1 \
    -I . -I vendor/mojo-libs \
    -Xlinker -lm -Xlinker -lcuda \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
    -Xlinker -lcudnn \
    -Xlinker -rpath -Xlinker '$ORIGIN/../../serenitymojo/ops/cshim/lib' \
    -Xlinker -rpath -Xlinker '$ORIGIN/../../.pixi/envs/default/lib' \
    serenitymojo/sampling/parity/ltx2_conditioning_mask_parity.mojo \
    -o "$runner"

LD_LIBRARY_PATH="$repo_root/serenitymojo/ops/cshim/lib:$repo_root/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$runner"
