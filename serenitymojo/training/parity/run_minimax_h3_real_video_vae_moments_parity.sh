#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
musubi_repo="${MUSUBI_REPO:-$(dirname "$repo")/musubi-tuner}"
python_bin="${H3_ORACLE_PYTHON:-${H3_TORCH_ORACLE_PYTHON:-python3}}"
dataset="${H3_REAL_DATASET:-$(dirname "$repo")/eri2_with_trigger}"
vae_file="${H3_VIDEO_VAE_FILE:-$(dirname "$repo")/SwarmUI/Models/VAE/MiniMaxH3/minimax_h3_video_vae_fp16.safetensors}"
vae_dir="${H3_VIDEO_VAE_DIR:-$(dirname "$repo")/SerenityFlow/serenityflow/models/minimax_h3/vae/video/source}"
fixture="$repo/serenitymojo/training/parity/fixtures/minimax_h3_real_video_vae_moments_f32_v1.json"
gate="$repo/serenitymojo/training/parity/minimax_h3_real_video_vae_moments_parity.mojo"
binary=/tmp/minimax_h3_real_video_vae_moments_parity

cleanup() {
  rm -f "$binary"
}
trap cleanup EXIT

test -x "$python_bin"
test -f "$dataset/1.jpg"
test -f "$vae_file"
test -f "$vae_dir/model.safetensors"
test -f "$fixture"
git -C "$musubi_repo" cat-file -e \
  b8717864713c9e4e7ef3d56eba1fc695a9b626a5^{commit}

test "$(stat -c %s "$dataset/1.jpg")" = 499926
test "$(sha256sum "$dataset/1.jpg" | awk '{print $1}')" = \
  fc41782cac93cafc92e83ddb57e93243c9f4f97c70f25f4b4fec5d64f875a996
test "$(stat -c %s "$vae_file")" = 5207808496
test "$(sha256sum "$vae_file" | awk '{print $1}')" = \
  7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522
test "$(sha256sum "$vae_dir/model.safetensors" | awk '{print $1}')" = \
  7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522

cd "$repo"
"$python_bin" scripts/minimax_h3_real_video_vae_moments_oracle.py \
  --musubi-repo "$musubi_repo" \
  --dataset "$dataset" \
  --vae "$vae_file" \
  --check "$fixture"
sha256sum -c \
  serenitymojo/training/parity/fixtures/minimax_h3_real_video_vae_moments_f32_v1.sha256

pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$gate" -o "$binary" -Xlinker -lm
"$binary" "$dataset" "$vae_dir" "$fixture"
