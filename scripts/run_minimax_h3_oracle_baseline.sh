#!/usr/bin/env bash
# Run the pinned Musubi MiniMax-H3 implementation as the external 200-step
# training-loss oracle on the same native 768x768 image cache used by Mojo.
# Rootless; this is reference measurement only, never the production trainer.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
oracle_root=/home/alex/musubi-h3
oracle_python="$oracle_root/.venv/bin/python"
oracle_accelerate="$oracle_root/.venv/bin/accelerate"
dataset=/home/alex/datasets/h3_eri_cache/dataset.toml
dit=/home/alex/.serenity/models/checkpoints/MiniMax-H3/single_file/diffusion_models/minimax_h3_fl2va_bf16.safetensors
steps="${H3_ORACLE_STEPS:-200}"

if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "H3_ORACLE_STEPS must be a positive integer" >&2
  exit 64
fi
for required in "$oracle_python" "$oracle_accelerate" "$dataset" "$dit"; do
  if [[ ! -e "$required" ]]; then
    echo "H3 oracle baseline: missing $required" >&2
    exit 66
  fi
done
gpu_busy_limit_mib="${H3_GPU_BUSY_LIMIT_MIB:-1024}"
gpu_used_mib=$(nvidia-smi \
  --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk '{sum += $1} END {print sum + 0}')
if (( gpu_used_mib > gpu_busy_limit_mib )); then
  echo "H3 oracle baseline: GPU compute users hold ${gpu_used_mib} MiB" >&2
  echo "H3 oracle baseline: limit is ${gpu_busy_limit_mib} MiB" >&2
  exit 75
fi

tag=$(date +%Y%m%d-%H%M%S)
run_dir="$repo_root/output/checks/h3_musubi_oracle_$tag"
mkdir -p "$run_dir"
log="$run_dir/oracle.log"
exec > >(tee -a "$log") 2>&1

oracle_commit=$(git -C "$oracle_root" rev-parse HEAD)
echo "MiniMax-H3 external training-loss oracle"
echo "oracle_commit=$oracle_commit"
echo "steps=$steps seed=42 geometry=native buckets at 768x768 target area guidance=off"
echo "weights=FP8-channel blocks_to_swap=48 ring=1 pinned=no"
echo "dataset=$dataset"
echo "run_dir=$run_dir"

# The official Comfy single-file export already stores QKV as contiguous
# [q_all;k_all;v_all], matching the pinned Musubi model's chunk(3) contract.
# FP8 channel scaling is the closest upstream low-memory analogue to Mojo's
# per-output-row FP8 frozen base. The rootless user service contains any host
# OOM to this run without requiring a sudo credential.
cd "$oracle_root"
MEM_MAX=40G MEM_HIGH=infinity SWAP_MAX=2G \
  "$repo_root/scripts/mem_safe_runtime.sh" /usr/bin/env \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$repo_root/scripts/h3_oracle_pydeps:$oracle_root/src" \
    MUSUBI_DASHBOARD_METRICS=1 \
    "$oracle_accelerate" launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    "$oracle_root/minimax_h3_train_network.py" \
    --dit "$dit" \
    --dataset_config "$dataset" \
    --network_module networks.lora_minimax_h3 \
    --network_dim 16 --network_alpha 16 \
    --sdpa --mixed_precision bf16 --gradient_checkpointing \
    --fp8_base --h3_fp8_quantization_mode channel \
    --blocks_to_swap 48 --block_swap_h2d_only --block_swap_ring_size 1 \
    --optimizer_type AdamW8bit --learning_rate 1e-4 \
    --max_train_steps "$steps" --save_every_n_steps "$steps" \
    --log_cuda_memory_every_n_steps 1 \
    --max_data_loader_n_workers 0 \
    --seed 42 \
    --output_dir "$run_dir" --output_name h3_musubi_oracle_seed42 \
    --save_precision bf16

python3 "$repo_root/scripts/summarize_minimax_h3_oracle.py" "$run_dir"
echo "H3 Musubi oracle complete: $log"
