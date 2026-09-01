#!/usr/bin/env bash
set -euo pipefail

# Mandatory evidence runner. The caller must own the single Mojo compile slot.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
oracle="${script_dir}/minimax_h3_training_device_av_head_oracle.py"
gate="${script_dir}/minimax_h3_training_device_av_head_parity.mojo"
binary="/tmp/minimax_h3_training_device_av_head_parity"
oracle_python="${H3_TORCH_ORACLE_PYTHON:-python3}"

cd "${repo_root}"
"${oracle_python}" "${oracle}" --check
generation_dir="$(mktemp -d /tmp/minimax-h3-av-head-generations.XXXXXX)"
run_dir="$(mktemp -d /tmp/minimax-h3-av-head-runs.XXXXXX)"
trap 'rm -rf "${generation_dir}" "${run_dir}"' EXIT
for run_id in 1 2 3; do
    "${oracle_python}" "${oracle}" \
        --output "${generation_dir}/fixture_${run_id}.safetensors" >/dev/null
done
canonical_sha="$(sha256sum "${fixture_dir}/minimax_h3_training_device_av_head.safetensors" | awk '{print $1}')"
for run_id in 1 2 3; do
    generated_sha="$(sha256sum "${generation_dir}/fixture_${run_id}.safetensors" | awk '{print $1}')"
    if [[ "${generated_sha}" != "${canonical_sha}" ]]; then
        echo "H3 AV/head fresh oracle generation digest mismatch" >&2
        exit 1
    fi
done
echo "PASS: three fresh oracle processes reproduced ${canonical_sha}"
(
    cd "${fixture_dir}"
    sha256sum -c minimax_h3_training_device_av_head.sha256
)
sidecar_sha="$(awk 'NR == 1 { print $1 }' "${fixture_dir}/minimax_h3_training_device_av_head.sha256")"
source_sha="$(awk -F '"' '/comptime FIXTURE_SHA256/ { print $2; exit }' "${gate}")"
if [[ -z "${source_sha}" || "${source_sha}" != "${sidecar_sha}" ]]; then
    echo "H3 AV/head gate FIXTURE_SHA256 does not match validated sidecar" >&2
    exit 1
fi
pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
    "${gate}" -o "${binary}"
for run_id in 1 2 3; do
    "${binary}" >"${run_dir}/run_${run_id}.txt"
done
cmp "${run_dir}/run_1.txt" "${run_dir}/run_2.txt"
cmp "${run_dir}/run_1.txt" "${run_dir}/run_3.txt"
cat "${run_dir}/run_1.txt"
echo "PASS: three fresh device processes produced an identical metric envelope"
