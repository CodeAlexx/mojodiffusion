#!/usr/bin/env bash
set -euo pipefail

repo=/home/alex/mojodiffusion
source_file="$repo/serenitymojo/training/parity/minimax_h3_source_dataset_intake_smoke.mojo"
binary=/tmp/minimax_h3_source_dataset_intake_smoke
dataset_path="${MINIMAX_H3_ERI_DATASET_PATH:?set MINIMAX_H3_ERI_DATASET_PATH to the physical eri_with_trigger source directory}"

cleanup() {
  rm -f "$binary"
}
trap cleanup EXIT

cd "$repo"
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$source_file" -o "$binary"
"$binary" "$dataset_path"
