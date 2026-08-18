#!/usr/bin/env bash
# Build and gate the corrected H3 training path, then measure a deterministic
# untrained-loss baseline. Rootless; no sudo credential or system slice.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

steps="${H3_BASELINE_STEPS:-200}"
base_config="${1:-$repo_root/serenitymojo/configs/minimax_h3_eri2_4000.json}"
if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "H3_BASELINE_STEPS must be a positive integer" >&2
  exit 64
fi
if [[ ! -s "$base_config" ]]; then
  echo "H3 baseline config not found: $base_config" >&2
  exit 66
fi
gpu_busy_limit_mib="${H3_GPU_BUSY_LIMIT_MIB:-1024}"
gpu_used_mib=$(nvidia-smi \
  --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk '{sum += $1} END {print sum + 0}')
if (( gpu_used_mib > gpu_busy_limit_mib )); then
  echo "H3 baseline: GPU compute users hold ${gpu_used_mib} MiB" >&2
  echo "H3 baseline: limit is ${gpu_busy_limit_mib} MiB" >&2
  exit 75
fi

tag=$(date +%Y%m%d-%H%M%S)
run_dir="$repo_root/output/checks/h3_loss_baseline_$tag"
mkdir -p "$run_dir"
log="$run_dir/baseline.log"
run_config="$run_dir/train_config.json"
exec > >(tee -a "$log") 2>&1

jq --arg workspace "$run_dir" \
   --arg output "$run_dir/h3_corrected_base_seed42.safetensors" \
   --arg name "h3_corrected_base_seed42" \
   --argjson steps "$steps" \
   '.workspace_dir = $workspace
    | .output_model_destination = $output
    | .save_filename_prefix = $name
    | .max_steps = $steps
    | .save_every = $steps
    | .sample_every = $steps' \
   "$base_config" > "$run_config"

echo "H3 config-driven corrected-base loss baseline"
echo "config=$run_config"
echo "recipe=$(jq -c '{steps:.max_steps,seed:.seed,rank:.lora_rank,alpha:.lora_alpha,targets:.layer_filter_preset,lr:.learning_rate,optimizer:.optimizer.optimizer,guidance:.guidance_scale,preservation:.h3_base_preservation_loss_weight,preservation_probability:.h3_base_preservation_probability,resident_blocks:.resident_blocks}' "$run_config")"
echo "run_dir=$run_dir"

# O2/-j1 is the accepted H3 compiler configuration. All builds and gates use
# the rootless hard-capped user service.
scripts/build_minimax_h3_training_gates.sh

runtime_ld="$repo_root/.pixi/envs/default/lib:$repo_root/serenitymojo/ops/cshim/lib:$repo_root/output/lib"
for gate in \
  output/checks/minimax_h3_block_train_parity \
  output/checks/minimax_h3_stack_train_parity \
  output/checks/minimax_h3_adamw8bit_device_parity
do
  echo "running H3 training gate: $gate"
  LD_LIBRARY_PATH="$runtime_ld" \
  MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G \
    scripts/mem_safe_runtime.sh "$gate"
done

H3_REBUILD_TRAINER=1 scripts/build_minimax_h3_trainer.sh

echo "starting $steps-step corrected frozen-base loss baseline"
LD_LIBRARY_PATH="$runtime_ld" \
H3_ALLOW_USER_SLICE=1 DESKTOP_RESERVE=24G \
MEM_MAX=18G MEM_HIGH=infinity SWAP_MAX=2G \
  scripts/mem_safe_runtime.sh output/bin/train_minimax_h3 \
    "$run_config" --baseline_only 1

echo "H3 baseline complete: $log"
