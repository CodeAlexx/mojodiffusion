#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
env_root=${CONDA_PREFIX:?Run this build through pixi: pixi run bash serenitymojo/models/dit/cshim/build_minimax_h3_int8.sh}
output_path=${1:-"$repo_root/output/lib/libserenity_minimax_h3_int8.so"}

mkdir -p "$(dirname -- "$output_path")"
g++ -O3 -std=c++17 -fPIC -shared \
  "$script_dir/minimax_h3_int8_gemm.cpp" \
  -I"$env_root/targets/x86_64-linux/include" \
  -L"$env_root/lib" \
  -Wl,-rpath,'$ORIGIN/../../.pixi/envs/default/lib' \
  -lcublas \
  -o "$output_path"
