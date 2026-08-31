#!/usr/bin/env bash
set -euo pipefail

# Mandatory real-artifact intake/layout runner.  This never modifies the
# checkpoint.  Three independent Python processes must reproduce the canonical
# receipt before the one compiled Mojo gate is allowed to touch a DeviceContext.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
fixture="${fixture_dir}/minimax_h3_fl2va_convrot_artifact_v1.json"
sidecar="${fixture_dir}/minimax_h3_fl2va_convrot_artifact_v1.sha256"
oracle="${repo_root}/scripts/minimax_h3_fl2va_convrot_artifact_oracle.py"
gate="${script_dir}/minimax_h3_fl2va_convrot_artifact_intake.mojo"
artifact=/home/alex/SwarmUI/Models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
comfy_repo="${H3_FL2VA_COMFY_REPO:-/home/alex/SwarmUI/dlbackend/ComfyUI}"
binary=/tmp/minimax_h3_fl2va_convrot_artifact_intake
temp_dir="$(mktemp -d /tmp/h3-fl2va-convrot-intake.XXXXXX)"

cleanup() {
  rm -f "${binary}"
  rm -rf "${temp_dir}"
}
trap cleanup EXIT

cd "${repo_root}"
for run in 1 2 3; do
  python3 "${oracle}" \
    --artifact "${artifact}" \
    --comfy-repo "${comfy_repo}" \
    --output "${temp_dir}/fixture-${run}.json"
done
cmp "${temp_dir}/fixture-1.json" "${temp_dir}/fixture-2.json"
cmp "${temp_dir}/fixture-1.json" "${temp_dir}/fixture-3.json"
cmp "${temp_dir}/fixture-1.json" "${fixture}"
python3 "${oracle}" \
  --artifact "${artifact}" \
  --comfy-repo "${comfy_repo}" \
  --check
(
  cd "${fixture_dir}"
  sha256sum -c "$(basename "${sidecar}")"
)

sidecar_sha="$(awk 'NR == 1 { print $1 }' "${sidecar}")"
gate_sha="$(awk -F '"' '/comptime FIXTURE_SHA256/ { getline; print $2; exit }' "${gate}")"
if [[ -z "${gate_sha}" || "${gate_sha}" != "${sidecar_sha}" ]]; then
  echo "FL2VA ConvRot gate FIXTURE_SHA256 does not match sidecar" >&2
  exit 1
fi

pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
  -Xlinker -lm -Xlinker -lcuda \
  "${gate}" -o "${binary}"

for run in 1 2 3; do
  if ! "${binary}" >"${temp_dir}/device-${run}.txt" 2>&1; then
    cat "${temp_dir}/device-${run}.txt" >&2
    echo "FL2VA ConvRot device run ${run} failed" >&2
    exit 1
  fi
done
cmp "${temp_dir}/device-1.txt" "${temp_dir}/device-2.txt"
cmp "${temp_dir}/device-1.txt" "${temp_dir}/device-3.txt"
cat "${temp_dir}/device-1.txt"
echo "FL2VA ConvRot intake mandatory runner PASS: 3 fresh receipts + 3 device runs"
