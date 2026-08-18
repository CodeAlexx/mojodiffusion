#!/usr/bin/env bash
# Gate, rebuild, and run a fresh corrected 200-step Mojo H3 LoRA probe.
# Rootless: the trainer is isolated in a tightly capped transient user service.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

steps="${H3_PROBE_STEPS:-200}"
if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "H3_PROBE_STEPS must be a positive integer" >&2
  exit 64
fi
gpu_busy_limit_mib="${H3_GPU_BUSY_LIMIT_MIB:-1024}"
gpu_used_mib=$(nvidia-smi \
  --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
  | awk '{sum += $1} END {print sum + 0}')
if (( gpu_used_mib > gpu_busy_limit_mib )); then
  echo "H3 training probe: GPU compute users hold ${gpu_used_mib} MiB" >&2
  echo "H3 training probe: limit is ${gpu_busy_limit_mib} MiB" >&2
  exit 75
fi

tag=$(date +%Y%m%d-%H%M%S)
run_dir="$repo_root/output/checks/h3_mojo_probe_$tag"
mkdir -p "$run_dir"
log="$run_dir/mojo_probe.log"
exec > >(tee -a "$log") 2>&1

echo "MiniMax-H3 corrected Mojo training probe"
echo "steps=$steps seed=42 geometry=native-bucket guidance=off"
echo "weights=direct-W8A8 resident_blocks=42 compute=BF16"
echo "adapters=qkv/out/fc1/fc2 rank=16 alpha=16"
echo "run_dir=$run_dir"

# O2/-j1 is the accepted H3 compiler configuration; O3 is intentionally
# unsupported here. Each build/gate is rootless and hard-capped.
scripts/build_minimax_h3_training_gates.sh

runtime_ld="$repo_root/.pixi/envs/default/lib:$repo_root/serenitymojo/ops/cshim/lib:$repo_root/output/lib"
for gate in \
  output/checks/minimax_h3_int8_linear_parity \
  output/checks/minimax_h3_block_train_parity \
  output/checks/minimax_h3_stack_train_parity \
  output/checks/minimax_h3_lora_peft_roundtrip
do
  echo "running H3 training gate: $gate"
  LD_LIBRARY_PATH="$runtime_ld" \
  MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G \
    scripts/mem_safe_runtime.sh "$gate"
done

H3_REBUILD_TRAINER=1 scripts/build_minimax_h3_trainer.sh

echo "starting $steps-step corrected Mojo optimizer probe"
LD_LIBRARY_PATH="$runtime_ld" \
H3_ALLOW_USER_SLICE=1 DESKTOP_RESERVE=24G \
MEM_MAX=18G MEM_HIGH=infinity SWAP_MAX=2G \
  scripts/mem_safe_runtime.sh output/bin/train_minimax_h3 \
    --cache_dir /home/alex/datasets/h3_eri_cache/cache \
    --out_dir "$run_dir" \
    --name h3_mojo_probe_seed42 \
    --steps "$steps" \
    --dim 16 --alpha 16 --lr 0.0001 \
    --seed 42 --resident_blocks 42 \
    --save_every "$steps" --sample_every 1000000 \
    --guidance_scale 0

python3 "$repo_root/scripts/summarize_minimax_h3_training.py" "$log"
echo "H3 corrected Mojo probe complete: $log"
