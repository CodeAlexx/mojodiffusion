#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fixture="$repo/serenitymojo/training/parity/minimax_h3_released_base_sparse_fixture.py"
smoke="$repo/serenitymojo/training/parity/minimax_h3_product_preflight_smoke.mojo"
entry="$repo/serenitymojo/models/minimax_h3/train.mojo"
binary=/tmp/minimax_h3_product_preflight_smoke
entry_binary=/tmp/minimax_h3_train_preflight_build
tmp=/tmp/serenity_h3_product_preflight_v1

target_accelerator=${H3_TARGET_ACCELERATOR:-}
if [[ -z "$target_accelerator" ]]; then
  compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
  if [[ ! "$compute_cap" =~ ^[0-9]+$ ]]; then
    echo "Unable to detect the NVIDIA compute capability; set H3_TARGET_ACCELERATOR (for example sm_120)." >&2
    exit 1
  fi
  target_accelerator="sm_$compute_cap"
fi

cleanup() {
  rm -f \
    "$binary" "$entry_binary" \
    "$tmp/released_base.safetensors" \
    "$tmp/released_base_bad_shape.safetensors" \
    "$tmp/cache/sample.latent.safetensors" \
    "$tmp/cache/sample.text.safetensors" \
    "$tmp/cache/cache_manifest.json" \
    "$tmp/eri_with_trigger/sample.png" \
    "$tmp/eri_with_trigger/sample.txt" \
    "$tmp/config.json"
}
trap cleanup EXIT

cd "$repo"
python3 "$fixture"
python3 "$fixture" --check
pixi run mojo build --target-accelerator "$target_accelerator" \
  --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$entry" -o "$entry_binary"
pixi run mojo build --target-accelerator "$target_accelerator" \
  --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$smoke" -o "$binary"
"$binary"
