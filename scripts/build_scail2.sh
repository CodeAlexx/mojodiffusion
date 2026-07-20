#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mkdir -p output/bin

sources=(
  serenitymojo/pipeline/scail2_stage_inputs.mojo
  serenitymojo/pipeline/scail2_encode_prompt.mojo
  serenitymojo/pipeline/scail2_encode_clip.mojo
  serenitymojo/pipeline/scail2_encode_vae.mojo
  serenitymojo/pipeline/scail2_prepare_fp8_cache.mojo
  serenitymojo/pipeline/scail2_animation.mojo
  serenitymojo/pipeline/scail2_decode.mojo
)

for source_path in "${sources[@]}"; do
  binary_name=$(basename "$source_path" .mojo)
  mojo build --optimization-level 2 \
    -I . -I vendor/mojo-libs \
    -Xlinker -lm -Xlinker -ldl \
    -Xlinker -L"$CONDA_PREFIX/targets/x86_64-linux/lib/stubs" \
    -Xlinker -lcuda \
    -Xlinker -L"$CONDA_PREFIX/lib" \
    -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib" \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -rpath \
    -Xlinker '$ORIGIN/../../.pixi/envs/default/lib:$ORIGIN/../../serenitymojo/ops/cshim/lib' \
    "$source_path" -o "output/bin/$binary_name"
done

bash scripts/fix_mojo_runpath.sh \
  output/bin/scail2_stage_inputs \
  output/bin/scail2_encode_prompt \
  output/bin/scail2_encode_clip \
  output/bin/scail2_encode_vae \
  output/bin/scail2_prepare_fp8_cache \
  output/bin/scail2_animation \
  output/bin/scail2_decode
