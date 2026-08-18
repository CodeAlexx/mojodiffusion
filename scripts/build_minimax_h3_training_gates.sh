#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
mkdir -p output/checks

build_gate() {
  local source=$1
  local output=$2
  echo "building H3 training gate: $output"
  MEM_MAX="${H3_BUILD_MEM_MAX:-24G}" \
  MEM_HIGH="${H3_BUILD_MEM_HIGH:-infinity}" \
  SWAP_MAX="${H3_BUILD_SWAP_MAX:-2G}" \
    scripts/mem_safe_runtime.sh pixi run mojo build \
      --optimization-level 2 --disable-warnings -j 1 -I . -I vendor/mojo-libs \
      -Xlinker="-L$repo_root/.pixi/envs/default/lib" \
      -Xlinker=-rpath-link \
      -Xlinker="$repo_root/.pixi/envs/default/lib" \
      -Xlinker=-lm -Xlinker=-lcuda -Xlinker=-lcublas \
      -Xlinker=-Lserenitymojo/ops/cshim/lib \
      -Xlinker=-lserenity_cudnn_sdpa \
      -Xlinker=-Lserenitymojo/ops/cshim/lib/cudnn_stubs \
      -Xlinker=-lcudnn \
      "$source" -o "$output"
}

# Each whole-program build gets its own isolated unit. O2/-j1 is the accepted
# H3 compiler setting; never raise this script to O3 on the desktop.
build_gate \
  serenitymojo/ops/tests/int8_linear_parity.mojo \
  output/checks/minimax_h3_int8_linear_parity
build_gate \
  serenitymojo/models/minimax_h3/parity/minimax_h3_block_train_parity.mojo \
  output/checks/minimax_h3_block_train_parity
build_gate \
  serenitymojo/models/minimax_h3/parity/minimax_h3_stack_train_parity.mojo \
  output/checks/minimax_h3_stack_train_parity
build_gate \
  serenitymojo/models/minimax_h3/parity/minimax_h3_lora_peft_roundtrip.mojo \
  output/checks/minimax_h3_lora_peft_roundtrip
