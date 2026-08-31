#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

output="output/bin/minimax_h3_endless"
mkdir -p "$(dirname "$output")"
pixi run mojo build --optimization-level 2 --disable-warnings -j 1 \
  -I . -I vendor/mojo-libs \
  serenitymojo/pipeline/minimax_h3_endless.mojo -o "$output"
echo "built: $output"
