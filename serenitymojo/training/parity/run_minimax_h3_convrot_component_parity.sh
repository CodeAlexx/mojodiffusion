#!/usr/bin/env bash
set -euo pipefail

# Mandatory bounded ConvRot component evidence runner. The caller must own the
# repository's single Mojo compile slot. Three independent oracle processes must
# emit byte-identical fixtures before the canonical fixture is accepted.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
fixture_dir="${script_dir}/fixtures"
fixture="${fixture_dir}/minimax_h3_convrot_component_v1.json"
sidecar="${fixture_dir}/minimax_h3_convrot_component_v1.sha256"
oracle="${repo_root}/scripts/minimax_h3_convrot_component_oracle.py"
gate="${script_dir}/minimax_h3_convrot_component_parity.mojo"
python="${H3_CONVROT_ORACLE_PYTHON:-/home/alex/OneTrainer/venv/bin/python}"
source_repo="${H3_CONVROT_MUSUBI_REPO:-/home/alex/musubi-tuner}"
binary=/tmp/minimax_h3_convrot_component_parity
temp_dir="$(mktemp -d /tmp/h3-convrot-oracle.XXXXXX)"

cleanup() {
  rm -f "${binary}"
  rm -rf "${temp_dir}"
}
trap cleanup EXIT

cd "${repo_root}"
for run in 1 2 3; do
  "${python}" "${oracle}" \
    --source-repo "${source_repo}" \
    --output "${temp_dir}/fixture-${run}.json"
done
cmp "${temp_dir}/fixture-1.json" "${temp_dir}/fixture-2.json"
cmp "${temp_dir}/fixture-1.json" "${temp_dir}/fixture-3.json"
cmp "${temp_dir}/fixture-1.json" "${fixture}"
"${python}" "${oracle}" --source-repo "${source_repo}" --check
(
  cd "${fixture_dir}"
  sha256sum -c "$(basename "${sidecar}")"
)

sidecar_sha="$(awk 'NR == 1 { print $1 }' "${sidecar}")"
gate_sha="$(awk -F '"' '/comptime FIXTURE_SHA256/ { getline; print $2; exit }' "${gate}")"
if [[ -z "${gate_sha}" || "${gate_sha}" != "${sidecar_sha}" ]]; then
  echo "ConvRot gate FIXTURE_SHA256 does not match validated sidecar" >&2
  exit 1
fi

pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
  -Xlinker -lm -Xlinker -lcuda \
  "${gate}" -o "${binary}"

for run in 1 2 3; do
  if ! "${binary}" >"${temp_dir}/device-${run}.txt" 2>&1; then
    cat "${temp_dir}/device-${run}.txt" >&2
    echo "ConvRot device run ${run} failed" >&2
    exit 1
  fi
done
cmp "${temp_dir}/device-1.txt" "${temp_dir}/device-2.txt"
cmp "${temp_dir}/device-1.txt" "${temp_dir}/device-3.txt"
cat "${temp_dir}/device-1.txt"
echo "ConvRot mandatory runner PASS: 3 fresh oracle + 3 fresh device runs"
