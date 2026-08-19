#!/usr/bin/env bash
# Config-driven, process-separated pure-Mojo MiniMax-H3 cadence supervisor.
# Musubi supplies oracle semantics only; no Python/Musubi trainer is launched.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

config=${1:-"$repo_root/serenitymojo/configs/minimax_h3_eri2_4000.json"}
trainer="$repo_root/output/bin/train_minimax_h3"
sampler="$repo_root/output/bin/minimax_h3_serenity_runtime"
runtime_ld="$repo_root/.pixi/envs/default/lib:$repo_root/serenitymojo/ops/cshim/lib:$repo_root/output/lib"

for required in "$config" "$trainer" "$sampler"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required path: $required" >&2
    exit 66
  fi
done
if ! command -v jq >/dev/null; then
  echo "jq is required to read the trainer and sample configs" >&2
  exit 69
fi

workspace=$(jq -er '.workspace_dir' "$config")
cache_dir=$(jq -er '.dataset_cache_dir // .cache_dir' "$config")
dataset_path=$(jq -er '.dataset_path' "$config")
name=$(jq -er '.save_filename_prefix' "$config")
max_steps=$(jq -er '.max_steps' "$config")
sample_every=$(jq -er '.sample_every' "$config")
sample_config=$(jq -er '.validation_prompts_file' "$config")
model_blocks=$(jq -er '(.num_double // 0) + (.num_single // 0)' "$config")

if [[ ! -d "$cache_dir" || ! -d "$dataset_path" || ! -s "$sample_config" ]]; then
  echo "config points at a missing dataset, cache, or sample config" >&2
  exit 66
fi
if (( max_steps <= 0 || sample_every <= 0 )); then
  echo "config max_steps and sample_every must be positive" >&2
  exit 64
fi

mkdir -p "$repo_root/output/training_runs" "$workspace/samples"
printf '%s\n' "$workspace" > "$repo_root/output/training_runs/minimax_h3.latest"

echo "MiniMax-H3 PURE-MOJO config-driven training"
echo "config=$config"
echo "sample_config=$sample_config"
echo "workspace=$workspace"
echo "dataset=$dataset_path cache=$cache_dir"
echo "max_steps=$max_steps save_every=$(jq -r '.save_every' "$config") sample_every=$sample_every"
echo "recipe=$(jq -c '{rank:.lora_rank,alpha:.lora_alpha,lr:.learning_rate,optimizer:.optimizer.optimizer,warmup:.learning_rate_warmup_steps,grad_clip:.max_grad_norm,guidance:.guidance_scale,timestep_buckets:.h3_num_timestep_buckets,density_jitter:.h3_spatial_density_jitter,preservation:.h3_base_preservation_loss_weight,preservation_probability:.h3_base_preservation_probability,resident_blocks:.resident_blocks}' "$config")"

latest_checkpoint_step() {
  local latest=0 path step
  shopt -s nullglob
  for path in "$workspace/${name}_step"*.safetensors; do
    step=${path##*_step}
    step=${step%.safetensors}
    if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest )); then
      latest=$step
    fi
  done
  shopt -u nullglob
  printf '%d\n' "$latest"
}

sample_all() {
  local step=$1 lora=$2 count index prompt_id prompt width height frames fps denoise_steps seed
  local quant resident attention sample_dir
  local -a lora_arg
  count=$(jq -er '[.prompts[] | select(.enabled != false)] | length' "$sample_config")
  for ((index=0; index<count; index++)); do
    prompt_id=$(jq -er "[.prompts[] | select(.enabled != false)][$index].id" "$sample_config")
    prompt=$(jq -er "[.prompts[] | select(.enabled != false)][$index].prompt" "$sample_config")
    width=$(jq -er "[.prompts[] | select(.enabled != false)][$index].width // .defaults.width" "$sample_config")
    height=$(jq -er "[.prompts[] | select(.enabled != false)][$index].height // .defaults.height" "$sample_config")
    frames=$(jq -er "[.prompts[] | select(.enabled != false)][$index].frames // .defaults.frames" "$sample_config")
    fps=$(jq -er "[.prompts[] | select(.enabled != false)][$index].fps // .defaults.fps" "$sample_config")
    denoise_steps=$(jq -er "[.prompts[] | select(.enabled != false)][$index].steps // .defaults.steps" "$sample_config")
    seed=$(jq -er "[.prompts[] | select(.enabled != false)][$index].seed // .defaults.seed" "$sample_config")
    quant=$(jq -er '.defaults.h3_quant // "int8"' "$sample_config")
    resident=$(jq -er '.defaults.h3_resident_blocks // 8' "$sample_config")
    attention=$(jq -er '.defaults.h3_attention_backend // "cudnn"' "$sample_config")
    sample_dir="$workspace/samples/step$(printf '%04d' "$step")_${prompt_id}"
    if [[ -s "$sample_dir/video.mp4" ]]; then
      echo "[H3-cadence] sample already complete: $sample_dir/video.mp4"
      continue
    fi
    mkdir -p "$sample_dir"
    echo "[H3-cadence] sampling step=$step prompt=$prompt_id in a fresh pure-Mojo process"
    lora_arg=()
    if [[ -n "$lora" ]]; then
      lora_arg+=("--lora=$lora")
    fi
    LD_LIBRARY_PATH="$runtime_ld" \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=85 \
    MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 \
    DESKTOP_RESERVE=24G MEM_MAX=18G MEM_HIGH=infinity SWAP_MAX=2G \
      scripts/mem_safe_runtime.sh "$sampler" \
        "$prompt" "$sample_dir" "$denoise_steps" "$seed" "$model_blocks" \
        --width="$width" --height="$height" \
        --frames="$frames" --output-frames="$frames" \
        --fps="$fps" --output-fps="$fps" \
        --quant="$quant" --resident-blocks="$resident" \
        --encoder-storage=int8 --attention-backend="$attention" \
        --step-cache=exact --temporal-rope-scale=1.0 "${lora_arg[@]}" \
        2>&1 | tee "$sample_dir/sample.log"
    if [[ ! -s "$sample_dir/video.mp4" ]]; then
      echo "step-$step validation did not produce video.mp4" >&2
      exit 1
    fi
    ffprobe -v error \
      -show_entries stream=index,codec_type,width,height,r_frame_rate,nb_frames,duration \
      -of default=noprint_wrappers=1 "$sample_dir/video.mp4" \
      | tee "$sample_dir/ffprobe.txt"
  done
}

current=$(latest_checkpoint_step)
if (( current == 0 )) && [[ $(jq -r '.defaults.sample_at_start // false' "$sample_config") == true ]]; then
  sample_all 0 ""
elif (( current > 0 )); then
  sample_all "$current" "$workspace/${name}_step${current}.safetensors"
fi

while (( current < max_steps )); do
  echo "[H3-cadence] training next config-defined segment after step $current"
  LD_LIBRARY_PATH="$runtime_ld" \
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=85 \
  MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT=100 \
  H3_ALLOW_USER_SLICE=1 DESKTOP_RESERVE=24G \
  MEM_MAX=18G MEM_HIGH=infinity SWAP_MAX=2G \
    scripts/mem_safe_runtime.sh "$trainer" "$config" --cadence_segment 1
  next=$(latest_checkpoint_step)
  if (( next <= current )); then
    echo "trainer exited without advancing beyond step $current" >&2
    exit 1
  fi
  current=$next
  lora="$workspace/${name}_step${current}.safetensors"
  sample_all "$current" "$lora"
done

echo "[H3-cadence] COMPLETE: $workspace"
