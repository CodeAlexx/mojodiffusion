#!/usr/bin/env bash
set -euo pipefail

# Mandatory evidence runner for the bounded product-core structural smoke.
# The caller must hold the repository's single Mojo compile slot.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
fixture_sha="${fixture_dir}/minimax_h3_training_stack50_bf16_flash.sha256"
oracle="${script_dir}/minimax_h3_training_stack50_bf16_flash_oracle.py"
gate="${script_dir}/minimax_h3_training_product_stack_smoke.mojo"
binary="/tmp/minimax_h3_training_product_stack_smoke"
oracle_python="${H3_TRAIN_ORACLE_PYTHON:-}"
target_accelerator="${H3_TARGET_ACCELERATOR:-}"

if [[ -z "${oracle_python}" ]]; then
    if [[ -x "${repo_root}/../OneTrainer/venv/bin/python" ]]; then
        oracle_python="${repo_root}/../OneTrainer/venv/bin/python"
    else
        oracle_python=python3
    fi
fi

if [[ -z "${target_accelerator}" ]]; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')"
    if [[ ! "${compute_cap}" =~ ^[0-9]+$ ]]; then
        echo "Unable to detect the NVIDIA compute capability; set H3_TARGET_ACCELERATOR." >&2
        exit 1
    fi
    target_accelerator="sm_${compute_cap}"
fi

cd "${repo_root}"
"${oracle_python}" "${oracle}" --check
(
    cd "${fixture_dir}"
    sha256sum -c minimax_h3_training_stack50_bf16_flash.sha256
)
sidecar_sha="$(awk 'NR == 1 { print $1 }' "${fixture_sha}")"
source_sha="$(awk -F '"' '/comptime FIXTURE_SHA256/ { print $2; exit }' "${gate}")"
if [[ -z "${source_sha}" || "${source_sha}" != "${sidecar_sha}" ]]; then
    echo "product stack smoke FIXTURE_SHA256 does not match validated sidecar" >&2
    exit 1
fi
pixi run mojo build --target-accelerator "${target_accelerator}" \
    -O2 -j 1 -I . -I vendor/mojo-libs \
    -Xlinker -lm -Xlinker -lcuda \
    -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
    -Xlinker -Lserenitymojo/ops/cshim/lib \
    -Xlinker -lserenity_cudnn_sdpa \
    -Xlinker -rpath \
    -Xlinker "${repo_root}/serenitymojo/ops/cshim/lib" \
    "${gate}" -o "${binary}"
"${binary}"
