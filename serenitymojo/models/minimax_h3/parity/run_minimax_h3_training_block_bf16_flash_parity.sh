#!/usr/bin/env bash
set -euo pipefail

# Mandatory evidence runner for the reduced mixed-dtype MiniMax-H3 flash core.
# The caller must hold the repository's single Mojo compile slot.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
fixture_sha="${fixture_dir}/minimax_h3_training_block_bf16_flash.sha256"
oracle="${script_dir}/minimax_h3_training_block_bf16_flash_oracle.py"
gate="${script_dir}/minimax_h3_training_block_bf16_flash_parity.mojo"
binary="/tmp/minimax_h3_training_block_bf16_flash_parity"
oracle_python="${H3_TORCH_ORACLE_PYTHON:-python3}"

cd "${repo_root}"
"${oracle_python}" "${oracle}" --check
(
    cd "${fixture_dir}"
    sha256sum -c minimax_h3_training_block_bf16_flash.sha256
)
sidecar_sha="$(awk 'NR == 1 { print $1 }' "${fixture_sha}")"
source_sha="$(awk -F '"' '/comptime REF_SHA256/ { print $2; exit }' "${gate}")"
if [[ -z "${source_sha}" || "${source_sha}" != "${sidecar_sha}" ]]; then
    echo "gate REF_SHA256 does not match the validated fixture sidecar" >&2
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
