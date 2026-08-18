#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

mode=${1:-finish}
run_dir=${2:-}
if [[ -z "$run_dir" ]]; then
  echo "usage: $0 finish|decode-video|decode-audio|mux RUN_DIR" >&2
  exit 64
fi

runner="$repo_root/output/bin/minimax_h3_serenity_runtime"
prompt_path="$repo_root/output/minimax_h3_prompts/waterbearer_king_120s_mk2va.txt"
internal_frames=2895
output_frames=2880
fps=24

if [[ ! -x "$runner" ]]; then
  echo "missing decode runner: $runner" >&2
  exit 66
fi
if [[ ! -s "$prompt_path" ]]; then
  echo "missing prompt: $prompt_path" >&2
  exit 66
fi
if [[ ! -s "$run_dir/latents.safetensors" ]]; then
  echo "missing completed latent artifact: $run_dir/latents.safetensors" >&2
  exit 66
fi

prompt=$(<"$prompt_path")
export LD_LIBRARY_PATH="$repo_root/output/lib:$repo_root/serenitymojo/ops/cshim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

decode() {
  local entry=$1
  local log_path=$2
  MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
    scripts/mem_safe_runtime.sh /usr/bin/time -v \
      "$runner" "$prompt" "$run_dir" 20 271828 50 "$entry" \
      --width=512 --height=320 --frames="$internal_frames" \
      --output-frames="$output_frames" --fps="$fps" --output-fps="$fps" \
      --quant=bf16 --resident-blocks=0 --encoder-storage=int8 \
      --attention-backend=cudnn --step-cache=exact --temporal-rope-scale=1.0 \
      2>&1 | tee "$log_path"
}

mux() {
  if [[ ! -s "$run_dir/frames.rgb" || ! -s "$run_dir/audio.wav" ]]; then
    echo "mux needs frames.rgb and audio.wav in $run_dir" >&2
    exit 66
  fi
  local audio_speed
  audio_speed=$(awk -v input="$internal_frames" -v output="$output_frames" \
    'BEGIN { printf "%.12f", input / output }')
  /usr/bin/time -v ffmpeg -v error -y \
    -f rawvideo -pixel_format rgb24 -video_size 512x320 -framerate "$fps" \
    -i "$run_dir/frames.rgb" -i "$run_dir/audio.wav" \
    -vf "setpts=PTS*${output_frames}/${internal_frames},fps=${fps}:round=near:eof_action=pass" \
    -af "atempo=${audio_speed},atrim=duration=120,asetpts=PTS-STARTPTS" \
    -frames:v "$output_frames" -c:v h264_nvenc -preset p7 -tune hq \
    -rc vbr -cq 18 -b:v 0 -pix_fmt yuv420p -c:a aac -shortest \
    -movflags +faststart "$run_dir/video.mp4" \
    2>&1 | tee "$run_dir/mux.log"
  ffprobe -v error \
    -show_entries stream=index,codec_type,width,height,r_frame_rate,nb_frames,duration \
    -of default=noprint_wrappers=1 "$run_dir/video.mp4" \
    | tee "$run_dir/ffprobe.txt"
}

case "$mode" in
  decode-video)
    decode decode_video_only "$run_dir/decode_video.log"
    ;;
  decode-audio)
    decode decode_audio_only "$run_dir/decode_audio.log"
    ;;
  mux)
    mux
    ;;
  finish)
    "$0" decode-video "$run_dir"
    "$0" decode-audio "$run_dir"
    "$0" mux "$run_dir"
    ;;
  *)
    echo "usage: $0 finish|decode-video|decode-audio|mux RUN_DIR" >&2
    exit 64
    ;;
esac
