#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mode=${1:-gate}
requested_dir=${2:-}
runner="$repo_root/output/bin/minimax_h3_serenity_runtime"
prompt_path="$repo_root/output/minimax_h3_prompts/waterbearer_king_180s_t2va.txt"
steps=20
seed=314159
blocks=50

if [[ ! -x "$runner" ]]; then
  echo "missing H3 runner: $runner" >&2
  echo "run H3_REBUILD_PROFILES=1 scripts/build_minimax_h3_video_profiles.sh" >&2
  exit 66
fi
if [[ ! -s "$prompt_path" ]]; then
  echo "missing H3 prompt: $prompt_path" >&2
  exit 66
fi

prompt=$(<"$prompt_path")
stamp=$(date +%Y%m%d-%H%M%S)
case "$mode" in
  validate)
    run_dir=${requested_dir:-"$repo_root/output/checks/h3_monolithic_180s_validate_$stamp"}
    ;;
  gate)
    run_dir=${requested_dir:-"$repo_root/output/checks/h3_monolithic_180s_gate_$stamp"}
    ;;
  full)
    run_dir=${requested_dir:-"$repo_root/output/checks/h3_monolithic_180s_full_$stamp"}
    ;;
  decode|decode-audio|decode-video)
    if [[ -z "$requested_dir" ]]; then
      echo "usage: $0 decode RUN_DIR" >&2
      exit 64
    fi
    run_dir=$requested_dir
    ;;
  *)
    echo "usage: $0 validate|gate|full|decode|decode-audio|decode-video [RUN_DIR]" >&2
    exit 64
    ;;
esac

common=(
  --width=320 --height=192
  --frames=4323 --output-frames=4320
  --fps=24 --output-fps=24
  --quant=int8-fast --resident-blocks=0
  --encoder-storage=int8 --attention-backend=cudnn
  --step-cache=exact
)
export LD_LIBRARY_PATH="$repo_root/output/lib:$repo_root/serenitymojo/ops/cshim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "$mode" == validate ]]; then
  "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks" \
    "${common[@]}" --validate-request
  exit 0
fi

mkdir -p "$run_dir"
if [[ "$mode" == decode || "$mode" == decode-audio || "$mode" == decode-video ]]; then
  decode_entry=decode_only
  if [[ "$mode" == decode-audio ]]; then
    decode_entry=decode_audio_only
  elif [[ "$mode" == decode-video ]]; then
    decode_entry=decode_video_only
  fi
  command=(
    "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks" "$decode_entry"
    "${common[@]}"
  )
  log_path="$run_dir/${mode}.log"
else
  command=(
    "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks"
    "${common[@]}" --defer-video-decode
  )
  if [[ "$mode" == gate ]]; then
    command+=(--eval-stop=1)
  fi
  log_path="$run_dir/run.log"
fi

echo "mode=$mode"
echo "run_dir=$run_dir"
echo "geometry=320x192 internal_frames=4323 output_frames=4320 duration=180s"
echo "trajectory=one global packed sequence, no continuation windows"

MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  scripts/mem_safe_runtime.sh /usr/bin/time -v "${command[@]}" \
  2>&1 | tee "$log_path"
