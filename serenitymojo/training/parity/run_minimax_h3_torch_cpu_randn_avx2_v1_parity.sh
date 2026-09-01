#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${H3_ORACLE_PYTHON:-${H3_TORCH_ORACLE_PYTHON:-python3}}"
fixture="$repo/serenitymojo/training/parity/fixtures/minimax_h3_torch_cpu_randn_avx2_v1.json"
gate="$repo/serenitymojo/training/parity/minimax_h3_torch_cpu_randn_avx2_v1_parity.mojo"
binary=/tmp/minimax_h3_torch_cpu_randn_avx2_v1_parity

cleanup() {
  rm -f "$binary"
}
trap cleanup EXIT

test -x "$python_bin"
cd "$repo"
ATEN_CPU_CAPABILITY=avx2 CUDA_VISIBLE_DEVICES='' \
  "$python_bin" scripts/minimax_h3_torch_cpu_randn_avx2_v1_oracle.py \
  --check "$fixture"
sha256sum -c \
  serenitymojo/training/parity/fixtures/minimax_h3_torch_cpu_randn_avx2_v1.sha256
pixi run mojo build --optimization-level 2 -j 1 -I . -I vendor/mojo-libs \
  "$gate" -o "$binary" -Xlinker -lm
"$binary"
