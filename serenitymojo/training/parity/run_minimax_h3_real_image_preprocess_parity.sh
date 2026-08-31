#!/usr/bin/env bash
set -euo pipefail

repo=/home/alex/mojodiffusion
dataset_path="${MINIMAX_H3_ERI_DATASET_PATH:?set MINIMAX_H3_ERI_DATASET_PATH to the physical eri_with_trigger source directory}"
fixture="$repo/serenitymojo/training/parity/fixtures/minimax_h3_real_image_preprocess_v1.json"
fixture_sha="$repo/serenitymojo/training/parity/fixtures/minimax_h3_real_image_preprocess_v1.sha256"
fresh="$(mktemp)"
binary=/tmp/minimax_h3_real_image_preprocess_parity

cleanup() {
  rm -f "$fresh" "$binary"
}
trap cleanup EXIT

cd "$repo"
uv run --isolated --no-project \
  --with numpy==2.5.2 \
  --with opencv-python==4.10.0.84 \
  --with pillow==11.3.0 \
  python scripts/minimax_h3_real_image_preprocess_oracle.py "$dataset_path" --output "$fresh"
cmp "$fixture" "$fresh"
(cd "$(dirname "$fixture")" && sha256sum -c "$(basename "$fixture_sha")")
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  serenitymojo/training/parity/minimax_h3_real_image_preprocess_parity.mojo -o "$binary"
"$binary" "$dataset_path" "$fixture"
