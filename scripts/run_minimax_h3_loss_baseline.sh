#!/usr/bin/env bash
# Build and gate the corrected H3 training path, then measure a deterministic
# untrained-loss baseline. Run only after `sudo -v` in the same local terminal.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

steps="${H3_BASELINE_STEPS:-200}"
if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "H3_BASELINE_STEPS must be a positive integer" >&2
  exit 64
fi

if ! sudo -n true 2>/dev/null; then
  echo "H3 baseline: sudo is not cached; run 'sudo -v' in this terminal" >&2
  exit 77
fi

tag=$(date +%Y%m%d-%H%M%S)
run_dir="$repo_root/output/checks/h3_loss_baseline_$tag"
mkdir -p "$run_dir"
log="$run_dir/baseline.log"
exec > >(tee -a "$log") 2>&1

echo "H3 corrected-base loss baseline"
echo "steps=$steps seed=42 guidance=off"
echo "run_dir=$run_dir"

# All whole-program builds and large real-weight gates run outside
# user@UID.service. O2/-j1 is the accepted H3 compiler configuration.
scripts/build_minimax_h3_training_gates.sh

runtime_ld="$repo_root/.pixi/envs/default/lib:$repo_root/serenitymojo/ops/cshim/lib:$repo_root/output/lib"
for gate in \
  output/checks/minimax_h3_block_train_parity \
  output/checks/minimax_h3_stack_train_parity
do
  echo "running H3 training gate: $gate"
  LD_LIBRARY_PATH="$runtime_ld" \
  MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G \
    scripts/mem_safe_system.sh "$gate"
done

H3_REBUILD_TRAINER=1 scripts/build_minimax_h3_trainer.sh

echo "starting $steps-step corrected frozen-base loss baseline"
LD_LIBRARY_PATH="$runtime_ld" \
MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G \
  scripts/mem_safe_system.sh output/bin/train_minimax_h3 \
    --cache_dir /home/alex/datasets/h3_eri_cache/cache \
    --out_dir "$run_dir" \
    --name h3_corrected_base_seed42 \
    --steps "$steps" \
    --dim 16 --alpha 16 --lr 0 \
    --seed 42 --resident_blocks 42 \
    --baseline_only 1

echo "H3 baseline complete: $log"
