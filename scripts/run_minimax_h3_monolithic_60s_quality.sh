#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mode=${1:-gate}
requested_dir=${2:-}
runner="$repo_root/output/bin/minimax_h3_serenity_runtime"
prompt_path="$repo_root/output/minimax_h3_prompts/waterbearer_king_60s_quality_t2va.txt"
steps=20
seed=271828
blocks=50
internal_frames=1450
output_frames=1440
run_dir=${requested_dir:-"$repo_root/output/checks/h3_monolithic_60s_quality_$(date +%Y%m%d-%H%M%S)"}

if [[ ! -x "$runner" ]]; then
  echo "missing H3 runner: $runner" >&2
  exit 66
fi
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
  --temporal-rope-scale=1.0
)
export LD_LIBRARY_PATH="$repo_root/output/lib:$repo_root/serenitymojo/ops/cshim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$run_dir"

case "$mode" in
  validate)
    command=(
      "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks"
      "${common[@]}" --validate-request
    )
    log_path="$run_dir/validate.log"
    ;;
  gate)
    command=(
      "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks"
      "${common[@]}" --eval-stop=1 --defer-video-decode
    )
    log_path="$run_dir/gate.log"
    ;;
  resume)
    if [[ ! -s "$run_dir/resume_latents.safetensors" ]]; then
      echo "missing one-evaluation checkpoint: $run_dir/resume_latents.safetensors" >&2
      exit 66
    fi
    command=(
      "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks"
      "${common[@]}" --eval-start=1 --defer-video-decode
    )
    log_path="$run_dir/resume.log"
    ;;
  full)
    command=(
      "$runner" "$prompt" "$run_dir" "$steps" "$seed" "$blocks"
      "${common[@]}" --defer-video-decode
    )
    log_path="$run_dir/run.log"
    ;;
  decode|decode-audio|decode-video)
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
    ;;
  mux)
    # H3's legal internal frame count can exceed the exact delivery count.
    # Retiming the complete internal trajectory preserves both conditioned
    # endpoints; taking only the first output_frames silently discards the
    # final-frame anchor.
    audio_speed=$(awk -v input="$internal_frames" -v output="$output_frames" \
      'BEGIN { printf "%.12f", input / output }')
    ffmpeg -v error -y \
      -f rawvideo -pixel_format rgb24 -video_size 256x256 -framerate 24 \
      -i "$run_dir/frames.rgb" -i "$run_dir/audio.wav" \
      -vf "setpts=PTS*${output_frames}/${internal_frames},fps=24:round=near:eof_action=pass" \
      -af "atempo=${audio_speed},atrim=duration=60,asetpts=PTS-STARTPTS" \
      -frames:v "$output_frames" \
      -c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 \
      -b:v 0 -pix_fmt yuv420p -c:a aac -shortest \
      -movflags +faststart "$run_dir/video.mp4"
    ffprobe -v error \
      -show_entries stream=index,codec_type,width,height,r_frame_rate,nb_frames,duration \
      -of default=noprint_wrappers=1 "$run_dir/video.mp4"
    exit 0
    ;;
  *)
    echo "usage: $0 validate|gate|resume|full|decode|decode-audio|decode-video|mux [RUN_DIR]" >&2
    exit 64
    ;;
esac

echo "mode=$mode"
echo "run_dir=$run_dir"
echo "geometry=256x256 internal_frames=1450 output_frames=1440 duration=60s"
echo "quality=streamed-bf16 attention=cudnn step_cache=exact temporal_rope_scale=1.0"
echo "trajectory=one global packed sequence, no continuation windows"

MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  scripts/mem_safe_runtime.sh /usr/bin/time -v "${command[@]}" \
  2>&1 | tee "$log_path"
