#!/usr/bin/env bash
set -euo pipefail

repo=/home/alex/mojodiffusion
gate="$repo/serenitymojo/training/parity/minimax_h3_image_preprocess_parity.mojo"
binary=/tmp/minimax_h3_image_preprocess_parity

cleanup() {
  rm -f "$binary"
}
trap cleanup EXIT

cd "$repo"
(cd serenitymojo/training/parity/fixtures && \
  sha256sum -c minimax_h3_image_preprocess_v1.sha256)
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$gate" -o "$binary" -Xlinker -lm
"$binary"
