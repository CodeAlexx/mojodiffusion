#!/usr/bin/env bash
# Build the exact GPU-only MiniMax-H3 keyframe product profiles. All INT8
# runners reuse the existing FL2VA resident caches; this script writes only
# small executables.
set -euo pipefail

repo=/home/alex/mojodiffusion
cd "$repo"

common=(
  --optimization-level 2 -j 1 -I . -I vendor/mojo-libs
  -D H3_HEIGHT=768 -D H3_WIDTH=768 -D H3_FRAMES=124
  -D H3_ENCODER_INT8_ALLOW_CACHE_BUILD=0
)
link=(
  -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
  -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs
  -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
)

build_mode() {
  local task=$1
  local text_tokens=$2
  local keyframes=$3
  local quant=$4
  local output="output/bin/minimax_h3_${task}_768x768x124_${quant}"
  local resident=()
  case "$quant" in
    bf16) resident=() ;;
    int8)
      # Keyframe sequences are about 2.7x larger than the admitted T2VA
      # profile. Keep the measured H3 law's ~5 GiB of allocator/workspace
      # headroom above the store. A reusable tail removed device allocation
      # churn, but the long-sequence activation pool still expanded by about
      # 2.55 GiB over eight steps; four/two blocks keep that growth bounded.
      if [[ "$task" == fl2va ]]; then
        resident=(-D H3_FP8_RESIDENT=1 -D H3_RESIDENT_BLOCKS=2)
      else
        resident=(-D H3_FP8_RESIDENT=1 -D H3_RESIDENT_BLOCKS=4)
      fi
      ;;
    int8_fast)
      if [[ "$task" == fl2va ]]; then
        resident=(-D H3_FP8_RESIDENT=1 -D H3_RESIDENT_BLOCKS=2)
      else
        resident=(-D H3_FP8_RESIDENT=1 -D H3_RESIDENT_BLOCKS=4)
      fi
      ;;
    *) echo "unknown quant mode: $quant" >&2; exit 2 ;;
  esac
  echo "building $output"
  pixi run scripts/mem_safe.sh mojo build \
    "${common[@]}" \
    -D "H3_TEXT_TOKENS=$text_tokens" -D "H3_KEYFRAMES=$keyframes" \
    "${resident[@]}" \
    serenitymojo/pipeline/minimax_h3_i2va.mojo -o "$output" \
    "${link[@]}"
}

for quant in bf16 int8 int8_fast; do
  build_mode i2va 970 1 "$quant"
  build_mode l2va 975 1 "$quant"
  build_mode fl2va 1581 2 "$quant"
done

echo "building output/bin/minimax_h3_decode_768x768x124"
pixi run scripts/mem_safe.sh mojo build \
  "${common[@]}" -D H3_TEXT_TOKENS=241 -D H3_VAE_STREAM_DECODE=1 \
  serenitymojo/pipeline/minimax_h3_t2va.mojo \
  -o output/bin/minimax_h3_decode_768x768x124 \
  "${link[@]}"
