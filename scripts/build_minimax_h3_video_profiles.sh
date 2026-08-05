#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# These profiles keep both DiT sequence length and decoded pixel-frames near
# the measured 24-GB product profile. Higher spatial resolution therefore
# comes with a shorter clip instead of an unsafe free-form VRAM increase.
profiles=(
  "512 320 175"
  "832 480 73"
  "960 544 56"
)

modes=("int8_fast" "int8" "bf16")
requested_profile=${1:-all}
requested_mode=${2:-all}

for profile in "${profiles[@]}"; do
  read -r width height frames <<<"$profile"
  profile_id="${width}x${height}x${frames}"
  if [[ "$requested_profile" != all && "$requested_profile" != "$profile_id" ]]; then
    continue
  fi

  for mode in "${modes[@]}"; do
    if [[ "$requested_mode" != all && "$requested_mode" != "$mode" ]]; then
      continue
    fi

    output="output/bin/minimax_h3_t2va_${profile_id}_${mode}"
    if [[ -x "$output" && "${H3_REBUILD_PROFILES:-0}" != 1 ]]; then
      echo "already built: $output"
      continue
    fi

    resident_flags=()
    case "$mode" in
      int8_fast)
        resident_blocks=48
        # The higher-resolution cuDNN shapes need two compact W8A8 tail
        # blocks of extra headroom. 832x480 OOMs before its first evaluation
        # at 48 resident blocks; 960x544 OOMs at evaluation four.
        if [[ "$profile_id" != "512x320x175" ]]; then
          resident_blocks=46
        fi
        resident_flags=(-D H3_FP8_RESIDENT=1 -D "H3_RESIDENT_BLOCKS=$resident_blocks")
        ;;
      int8)
        resident_blocks=43
        if [[ "$profile_id" != "512x320x175" ]]; then
          resident_blocks=41
        fi
        resident_flags=(-D H3_FP8_RESIDENT=1 -D "H3_RESIDENT_BLOCKS=$resident_blocks")
        ;;
      bf16)
        resident_flags=(-D H3_FP8_RESIDENT=0)
        ;;
      *)
        echo "unknown MiniMax-H3 mode: $mode" >&2
        exit 2
        ;;
    esac

    echo "building: $profile_id $mode"
    H3_BUILD_MEM_MAX=${H3_BUILD_MEM_MAX:-12G} \
    H3_BUILD_MEM_HIGH=${H3_BUILD_MEM_HIGH:-10G} \
    H3_BUILD_SWAP_MAX=${H3_BUILD_SWAP_MAX:-2G} \
    MEM_MAX="$H3_BUILD_MEM_MAX" MEM_HIGH="$H3_BUILD_MEM_HIGH" \
    SWAP_MAX="$H3_BUILD_SWAP_MAX" \
      pixi run scripts/mem_safe.sh mojo build \
        --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
        -D H3_TEXT_TOKENS=241 \
        -D "H3_FRAMES=$frames" -D "H3_HEIGHT=$height" -D "H3_WIDTH=$width" \
        -D H3_VAE_STREAM_DECODE=1 \
        "${resident_flags[@]}" \
        serenitymojo/pipeline/minimax_h3_t2va.mojo -o "$output" \
        -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8 \
        -Xlinker -Lserenitymojo/ops/cshim/lib \
        -Xlinker -lserenity_cudnn_sdpa \
        -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs \
        -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
  done
done
