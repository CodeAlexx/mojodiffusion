#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
musubi_repo="${MUSUBI_REPO:-$(dirname "$repo")/musubi-tuner}"
python_bin="${H3_ORACLE_PYTHON:-${H3_TORCH_ORACLE_PYTHON:-python3}}"
fixture="$repo/serenitymojo/training/parity/fixtures/minimax_h3_one_frame_target_cache_prepare_v1.json"
gate="$repo/serenitymojo/training/parity/minimax_h3_one_frame_target_cache_prepare_parity.mojo"
binary=/tmp/minimax_h3_one_frame_target_cache_prepare_parity

cleanup() {
  rm -f "$binary"
}
trap cleanup EXIT

test -x "$python_bin"
git -C "$musubi_repo" cat-file -e \
  b8717864713c9e4e7ef3d56eba1fc695a9b626a5^{commit}

cd "$repo"
ATEN_CPU_CAPABILITY=avx2 CUDA_VISIBLE_DEVICES='' \
  "$python_bin" scripts/minimax_h3_one_frame_target_cache_prepare_oracle.py \
  --musubi-repo "$musubi_repo" \
  --check "$fixture"
sha256sum -c \
  serenitymojo/training/parity/fixtures/minimax_h3_one_frame_target_cache_prepare_v1.sha256
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$gate" -o "$binary" -Xlinker -lm
"$binary"
