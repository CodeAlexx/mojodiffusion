#!/usr/bin/env bash
# Build the GPU-only MiniMax-H3 reference-image precision runners. Target
# width, height, and duration are selected at runtime; reference conditioning
# retains its own fixed presentation budget.
set -euo pipefail

repo=/home/alex/mojodiffusion
ref_root=/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA
cd "$repo"

common=(
  --optimization-level 2 -j 1 -I . -I vendor/mojo-libs
  -D H3_HEIGHT=768 -D H3_WIDTH=768 -D H3_FRAMES=124
  -D H3_REF_IMAGE_SHORT_EDGE=768
  -D H3_TEXT_TOKENS=937 -D H3_REF_SEQ_LEN=23239
  -D H3_REF2VA_REAL_BLOCKS=1
  -D H3_ENCODER_INT8_ALLOW_CACHE_BUILD=0
)
link=(
  -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
  -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs
  -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
)

build_one() {
  local quant=$1
  local resident=()
  case "$quant" in
    bf16) ;;
    int8|int8_fast) resident=(-D H3_FP8_RESIDENT=1 -D H3_RESIDENT_BLOCKS=4) ;;
    *) echo "unknown quant: $quant" >&2; exit 2 ;;
  esac
  local output="output/bin/minimax_h3_ref2va_768x768x124_${quant}"
  echo "building $output"
  pixi run scripts/mem_safe.sh mojo build \
    "${common[@]}" "${resident[@]}" \
    serenitymojo/pipeline/minimax_h3_ref2va.mojo -o "$output" \
    "${link[@]}"
}

[[ -f "$ref_root/transformer/model.safetensors.index.json" ]] || {
  echo "missing Ref2VA transformer" >&2; exit 1;
}

build_one bf16
build_one int8
build_one int8_fast
