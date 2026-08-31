#!/usr/bin/env bash
set -euo pipefail

repo=/home/alex/mojodiffusion
source_file="$repo/serenitymojo/training/parity/minimax_h3_bucket_geometry_parity.mojo"
binary=/tmp/minimax_h3_bucket_geometry_parity

cleanup() {
  rm -f "$binary"
}
trap cleanup EXIT

cd "$repo"
(cd serenitymojo/training/parity/fixtures && \
  sha256sum -c minimax_h3_bucket_geometry_v1.sha256)
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$source_file" -o "$binary"
"$binary"
