#!/usr/bin/env bash
set -euo pipefail

# Mandatory evidence runner. The caller must hold the single Mojo compile slot.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
fixture_sha="${fixture_dir}/minimax_h3_training_stack50_bf16_flash.sha256"
oracle="${script_dir}/minimax_h3_training_stack50_bf16_flash_oracle.py"
gate="${script_dir}/minimax_h3_training_stack50_bf16_flash_parity.mojo"
binary="/tmp/minimax_h3_training_stack50_bf16_flash_parity"

cd "${repo_root}"
/home/alex/OneTrainer/venv/bin/python "${oracle}" --check
(
    cd "${fixture_dir}"
    sha256sum -c minimax_h3_training_stack50_bf16_flash.sha256
)
sidecar_sha="$(awk 'NR == 1 { print $1 }' "${fixture_sha}")"
source_sha="$(awk -F '"' '/comptime FIXTURE_SHA256/ { print $2; exit }' "${gate}")"
if [[ -z "${source_sha}" || "${source_sha}" != "${sidecar_sha}" ]]; then
    echo "stack50 gate FIXTURE_SHA256 does not match validated sidecar" >&2
    exit 1
fi
pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
    -Xlinker -lm -Xlinker -lcuda \
    -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -rpath \
    -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
    "${gate}" -o "${binary}"
"${binary}"
