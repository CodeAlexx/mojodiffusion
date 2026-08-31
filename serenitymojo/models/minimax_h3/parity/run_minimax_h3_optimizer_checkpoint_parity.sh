#!/usr/bin/env bash
set -euo pipefail

# Mandatory fail-fast runner for the 200-target inventory plus four reduced
# optimizer/private-state component gate. Musubi weight-file I/O is excluded.
# The caller must hold the repository's single Mojo compile slot.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
fixture_sha="${fixture_dir}/minimax_h3_optimizer_checkpoint_v1.sha256"
oracle="${script_dir}/minimax_h3_optimizer_checkpoint_oracle.py"
gate="${script_dir}/minimax_h3_optimizer_checkpoint_parity.mojo"
binary="/tmp/minimax_h3_inventory_reduced_optimizer_private_state_gate"
one_trainer_python="${H3_ONETRAINER_ORACLE_PYTHON:-${repo_root}/../OneTrainer/venv/bin/python}"
ltx_python="${H3_LTX_ORACLE_PYTHON:-${repo_root}/../LTX-2/.venv/bin/python}"
target_accelerator="${H3_TARGET_ACCELERATOR:-}"

if [[ -z "${target_accelerator}" ]]; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')"
    if [[ ! "${compute_cap}" =~ ^[0-9]+$ ]]; then
        echo "Unable to detect the NVIDIA compute capability; set H3_TARGET_ACCELERATOR." >&2
        exit 1
    fi
    target_accelerator="sm_${compute_cap}"
fi

for oracle_python in "${one_trainer_python}" "${ltx_python}"; do
    if [[ ! -x "${oracle_python}" ]]; then
        echo "Missing required oracle Python: ${oracle_python}" >&2
        exit 1
    fi
done

cd "${repo_root}"

# Both supported development environments must independently regenerate the
# same canonical bytes before compilation is allowed to start.
"${one_trainer_python}" "${oracle}" --check
"${ltx_python}" "${oracle}" --check
(
    cd "${fixture_dir}"
    sha256sum -c minimax_h3_optimizer_checkpoint_v1.sha256
)
sidecar_sha="$(awk 'NR == 1 { print $1 }' "${fixture_sha}")"
gate_sha="$(awk -F '"' '/comptime FIXTURE_DIGEST/ { print $2; exit }' "${gate}")"
if [[ -z "${gate_sha}" || "${gate_sha}" != "${sidecar_sha}" ]]; then
    echo "gate FIXTURE_DIGEST does not match the validated fixture sidecar" >&2
    exit 1
fi

pixi run mojo build --target-accelerator "${target_accelerator}" \
    -O2 -j 1 -I . -I vendor/mojo-libs \
    -Xlinker -lm -Xlinker -lcuda \
    -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
    "${gate}" -o "${binary}"
"${binary}"
