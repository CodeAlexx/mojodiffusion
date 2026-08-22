#!/usr/bin/env bash
set -euo pipefail

# Build the Serenity Studio video/H3 runtimes that are not image workers, one
# compiler job at a time under scripts/mem_safe_runtime.sh (MemoryHigh=infinity,
# finite MemoryMax, desktop reserve; NOT the low-MemoryHigh mem_safe.sh scope
# that caused the 2026-08-13 oomd session kills, MJ-1140).
#
# Passing names builds a subset, for example:
#   bash scripts/build_serenity_video_runtimes.sh ltx2_runtime qwen3vl_caption
#
# Targets (binary under output/bin/):
#   h3_ref2va_cache  minimax_h3_ref2va_runtime_cache   (H3 Ref2VA cache runtime)
#   qwen3vl_caption  qwen3vl_caption                   (H3 studio captioner)
#   ideogram4_magic  ideogram4_magic                   (magic prompt, Qwen3-8B)
#   ltx2_runtime     ltx2_serenity_runtime             (pixi build-ltx2-request)
#   ltx2_prompt      ltx2_encode_prompt                (pixi build-ltx2-conditioning)
#   ltx25_prompt     ltx25_encode_prompt               (pixi build-ltx25-conditioning)
#   realesrgan       serenity_realesrgan_x4            (pixi build-realesrgan-x4)
# The main H3 runtime (minimax_h3_serenity_runtime) has its own builder:
#   bash scripts/build_minimax_h3_video_profiles.sh

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if (( $# )); then
  targets=("$@")
else
  targets=(h3_ref2va_cache qwen3vl_caption ideogram4_magic ltx2_runtime ltx2_prompt ltx25_prompt realesrgan)
fi

mkdir -p output/bin
export MEM_MAX="${MEM_MAX:-24G}"

h3_link=(
  -Xlinker -Loutput/lib -Xlinker -lserenity_minimax_h3_int8
  -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa
  -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs
  -Xlinker -lcudnn -Xlinker -lcuda -Xlinker -lm
)

build_mojo() {
  # build_mojo <final_binary> <source.mojo> [extra mojo build args...]
  local final_path=$1 source_path=$2
  shift 2
  local scratch_path="${final_path}.optimizing"
  echo "[video-runtime-build] $(basename "$final_path"): pinned -O2, one compiler job"
  scripts/mem_safe_runtime.sh pixi run mojo build \
    --optimization-level 2 --disable-warnings -j 1 \
    -I . -I vendor/mojo-libs \
    "$source_path" -o "$scratch_path" "$@"
  mv "$scratch_path" "$final_path"
  stat -c '[video-runtime-build] installed %n (%s bytes)' "$final_path"
}

build_pixi_task() {
  # build_pixi_task <task> <final_binary>
  local task=$1 final_path=$2
  echo "[video-runtime-build] $(basename "$final_path"): pixi run $task (pinned -O2 in pixi.toml)"
  scripts/mem_safe_runtime.sh pixi run "$task"
  stat -c '[video-runtime-build] installed %n (%s bytes)' "$final_path"
}

for target in "${targets[@]}"; do
  case "$target" in
    h3_ref2va_cache)
      build_mojo output/bin/minimax_h3_ref2va_runtime_cache \
        serenitymojo/pipeline/minimax_h3_ref2va_runtime_cache.mojo "${h3_link[@]}"
      ;;
    qwen3vl_caption)
      # Build line from the source header (pipeline/qwen3vl_caption.mojo).
      build_mojo output/bin/qwen3vl_caption serenitymojo/pipeline/qwen3vl_caption.mojo \
        -Xlinker -lm -Xlinker -lcuda
      ;;
    ideogram4_magic)
      build_mojo output/bin/ideogram4_magic serenitymojo/pipeline/ideogram4_magic.mojo \
        -Xlinker -lm -Xlinker -lcuda
      ;;
    ltx2_runtime) build_pixi_task build-ltx2-request output/bin/ltx2_serenity_runtime ;;
    ltx2_prompt) build_pixi_task build-ltx2-conditioning output/bin/ltx2_encode_prompt ;;
    ltx25_prompt) build_pixi_task build-ltx25-conditioning output/bin/ltx25_encode_prompt ;;
    realesrgan) build_pixi_task build-realesrgan-x4 output/bin/serenity_realesrgan_x4 ;;
    *)
      echo "unsupported video runtime target: $target" >&2
      exit 2
      ;;
  esac
done
