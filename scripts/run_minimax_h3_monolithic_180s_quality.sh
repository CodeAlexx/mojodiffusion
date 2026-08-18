#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mode=${1:-quality-gate}
requested_dir=${2:-}
runner="$repo_root/output/bin/minimax_h3_serenity_runtime"
gate_prompt_path="$repo_root/output/minimax_h3_prompts/waterbearer_king_15s_quality_gate_t2va.txt"
full_prompt_path="$repo_root/output/minimax_h3_prompts/waterbearer_king_180s_quality_t2va.txt"
steps=20
seed=271828
blocks=50

if [[ ! -x "$runner" ]]; then
  echo "missing H3 runner: $runner" >&2
  echo "run H3_REBUILD_PROFILES=1 scripts/build_minimax_h3_video_profiles.sh" >&2
  exit 66
fi

case "$mode" in
  quality-gate)
    prompt_path=$gate_prompt_path
    internal_frames=362
    output_frames=360
    duration=15
    run_dir=${requested_dir:-"$repo_root/output/checks/h3_monolithic_quality_gate_$(date +%Y%m%d-%H%M%S)"}
    ;;
  memory-gate|full|decode|decode-audio|decode-video|mux)
    prompt_path=$full_prompt_path
    internal_frames=4323
    output_frames=4320
    duration=180
    run_dir=${requested_dir:-"$repo_root/output/checks/h3_monolithic_180s_quality_$(date +%Y%m%d-%H%M%S)"}
    ;;
  *)
    echo "usage: $0 quality-gate|memory-gate|full|decode|decode-audio|decode-video|mux [RUN_DIR]" >&2
    exit 64
    ;;
esac

if [[ ! -s "$prompt_path" ]]; then
  echo "missing H3 prompt: $prompt_path" >&2
  exit 66
fi
prompt=$(<"$prompt_path")

common=(
  --width=256 --height=256
  --frames="$internal_frames" --output-frames="$output_frames"
  --fps=24 --output-fps=24
  --quant=bf16 --resident-blocks=0
  --encoder-storage=int8 --attention-backend=cudnn
  --step-cache=exact
)
export LD_LIBRARY_PATH="$repo_root/output/lib:$repo_root/serenitymojo/ops/cshim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$run_dir"
if [[ "$mode" == mux ]]; then
  # Keep the final conditioned endpoint by uniformly retiming every legal
  # internal frame into the exact 180-second delivery instead of truncating
  # the tail. Apply the same tiny speed-up to audio to preserve A/V sync.
  audio_speed=$(awk -v input="$internal_frames" -v output="$output_frames" \
    'BEGIN { printf "%.12f", input / output }')
  ffmpeg -v error -y \
    -f rawvideo -pixel_format rgb24 -video_size 256x256 -framerate 24 \
    -i "$run_dir/frames.rgb" -i "$run_dir/audio.wav" \
    -vf "setpts=PTS*${output_frames}/${internal_frames},fps=24:round=near:eof_action=pass" \
    -af "atempo=${audio_speed},atrim=duration=${duration},asetpts=PTS-STARTPTS" \
    -frames:v 4320 -c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 \
    -b:v 0 -pix_fmt yuv420p -c:a aac -shortest \
    -movflags +faststart "$run_dir/video.mp4"
  ffprobe -v error -show_entries stream=index,codec_type,width,height,r_frame_rate,nb_frames,duration \
    -of default=noprint_wrappers=1 "$run_dir/video.mp4"
  exit 0
fi

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
    "${common[@]}"
  )
  if [[ "$mode" == memory-gate ]]; then
    command+=(--eval-stop=1 --defer-video-decode)
  elif [[ "$mode" == full ]]; then
    command+=(--defer-video-decode)
  fi
  log_path="$run_dir/run.log"
fi

echo "mode=$mode"
echo "run_dir=$run_dir"
echo "geometry=256x256 internal_frames=$internal_frames output_frames=$output_frames duration=${duration}s"
echo "quality=streamed-bf16 attention=cudnn step_cache=exact"
echo "trajectory=one global packed sequence, no continuation windows"

MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  scripts/mem_safe_runtime.sh /usr/bin/time -v "${command[@]}" \
  2>&1 | tee "$log_path"
