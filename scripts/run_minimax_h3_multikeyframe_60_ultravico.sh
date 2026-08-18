#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

runner=${H3_RUNNER:-$repo_root/output/bin/minimax_h3_mk2va_512x320x1450_int8_ultravico}
prompt_path=${H3_PROMPT_PATH:-$repo_root/output/minimax_h3_prompts/waterbearer_king_60s_mk2va.txt}
anchor_dir=${H3_ANCHOR_DIR:-$repo_root/output/minimax_h3_keyframes/waterbearer_60s_high_quality_v2}
anchor_prefix=${H3_ANCHOR_PREFIX:-prepared}
native_width=${H3_NATIVE_WIDTH:-512}
native_height=${H3_NATIVE_HEIGHT:-320}
run_dir=${1:-$repo_root/output/checks/h3_mk2va_waterbearer_60s_${native_width}x${native_height}_$(date +%Y%m%d-%H%M%S)}
anchors=(
  "$anchor_dir/${anchor_prefix}_00.png"
  "$anchor_dir/${anchor_prefix}_01.png"
  "$anchor_dir/${anchor_prefix}_02.png"
  "$anchor_dir/${anchor_prefix}_03.png"
  "$anchor_dir/${anchor_prefix}_04.png"
  "$anchor_dir/${anchor_prefix}_05.png"
)

if [[ ! -x "$runner" ]]; then
  echo "missing runner: $runner (run scripts/build_minimax_h3_multikeyframe_60_ultravico.sh)" >&2
  exit 66
fi
if [[ ! -s "$prompt_path" ]]; then
  echo "missing prompt: $prompt_path" >&2
  exit 66
fi
for anchor in "${anchors[@]}"; do
  if [[ ! -s "$anchor" ]]; then
    echo "missing anchor: $anchor" >&2
    exit 66
  fi
done

mkdir -p "$run_dir"
prompt=$(<"$prompt_path")
export LD_LIBRARY_PATH="$repo_root/output/lib:$repo_root/serenitymojo/ops/cshim/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
eval_args=()
profile_prefix=()
riflex_args=()
semantic_anchor_args=()
attention_backend=${H3_ATTENTION_BACKEND:-sage-int8-ultravico}
resident_backend=${H3_RESIDENT_BACKEND:-w8a8}
steps=${H3_STEPS:-20}
case "$attention_backend" in
  cudnn|sage-int8|sage-int8-ultravico) ;;
  *)
    echo "unsupported H3_ATTENTION_BACKEND: $attention_backend" >&2
    exit 64
    ;;
esac
case "$resident_backend" in
  groupwise|w8a8) ;;
  *)
    echo "unsupported H3_RESIDENT_BACKEND: $resident_backend" >&2
    exit 64
    ;;
esac
if [[ ! "$steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "H3_STEPS must be a positive integer" >&2
  exit 64
fi
if [[ -n "${H3_EVAL_START:-}" ]]; then
  eval_args+=("--eval-start=$H3_EVAL_START")
fi
if [[ -n "${H3_EVAL_STOP:-}" ]]; then
  eval_args+=("--eval-stop=$H3_EVAL_STOP")
fi
if [[ -n "${H3_RIFLEX_K:-}" ]]; then
  if [[ "$attention_backend" == "sage-int8-ultravico" ]]; then
    echo "H3_RIFLEX_K cannot be combined with sage-int8-ultravico" >&2
    exit 64
  fi
  if [[ ! "$H3_RIFLEX_K" =~ ^[1-9][0-9]*$ ]]; then
    echo "H3_RIFLEX_K must be a positive integer" >&2
    exit 64
  fi
  riflex_args+=("--riflex-k=$H3_RIFLEX_K")
fi
if [[ "${H3_SEMANTIC_ROPE_ANCHORS:-0}" == "1" ]]; then
  semantic_anchor_args+=(--semantic-rope-anchors)
elif [[ "${H3_SEMANTIC_ROPE_ANCHORS:-0}" != "0" ]]; then
  echo "H3_SEMANTIC_ROPE_ANCHORS must be 0 or 1" >&2
  exit 64
fi
if [[ -n "${H3_NSYS_OUTPUT:-}" ]]; then
  mkdir -p "$(dirname "$H3_NSYS_OUTPUT")"
  profile_prefix=(
    "${H3_NSYS_BINARY:-nsys}" profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --force-overwrite=true
    --output="$H3_NSYS_OUTPUT"
  )
  if [[ -n "${H3_NSYS_DELAY:-}" ]]; then
    profile_prefix+=("--delay=$H3_NSYS_DELAY")
  fi
  if [[ -n "${H3_NSYS_DURATION:-}" ]]; then
    profile_prefix+=("--duration=$H3_NSYS_DURATION" --kill=sigterm)
  fi
fi

echo "run_dir=$run_dir"
denoise_log="$run_dir/denoise.log"
if [[ -n "${H3_EVAL_START:-}" ]]; then
  denoise_log="$run_dir/denoise_resume_from_${H3_EVAL_START}.log"
elif [[ -n "${H3_EVAL_STOP:-}" ]]; then
  denoise_log="$run_dir/denoise_gate_to_${H3_EVAL_STOP}.log"
fi
MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  scripts/mem_safe_runtime.sh /usr/bin/time -v \
    "${profile_prefix[@]}" \
    "$runner" mk2va "$prompt" "${anchors[@]}" "$run_dir" "$steps" 271828 50 \
    --width="$native_width" --height="$native_height" --frames=1450 --fps=24 \
    "--attention-backend=$attention_backend" --step-cache=exact \
    "--resident-backend=$resident_backend" --defer-video-decode \
    "${riflex_args[@]}" "${semantic_anchor_args[@]}" "${eval_args[@]}" \
    2>&1 | tee "$denoise_log"
