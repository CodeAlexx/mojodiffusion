#!/usr/bin/env bash
# Run a high-memory command in a transient SYSTEM service, outside
# user@1000.service. This is the required boundary for image/model runtimes:
# systemd-oomd must be able to kill this unit without destroying GNOME.
#
# By default the script never prompts for a password. Prime sudo explicitly in
# a trusted terminal (`sudo -v`) before invoking it. An interactive desktop
# Polkit prompt is available only when MEM_SAFE_SYSTEM_AUTH=polkit is explicit;
# the password stays inside the system authentication dialog.
set -euo pipefail

MEM_MAX="${MEM_MAX:-24G}"
MEM_HIGH="${MEM_HIGH:-infinity}"
SWAP_MAX="${SWAP_MAX:-2G}"

if [[ $# -lt 1 ]]; then
  echo "mem_safe_system: usage: $0 <program> [args...]" >&2
  exit 64
fi

prog="$1"
shift
prog_path="$(command -v "$prog")" || {
  echo "mem_safe_system: '$prog' not on PATH" >&2
  exit 127
}
if [[ "$prog_path" != /* ]]; then
  prog_path="$(realpath "$prog_path")"
fi

task_uid="$(id -u)"
task_gid="$(id -g)"
task_dir="$PWD"
unit_name="serenity-memory-$(date +%Y%m%d-%H%M%S)-$$"
auth_mode="${MEM_SAFE_SYSTEM_AUTH:-cached-sudo}"
extra_env=()
[[ -n "${CONDA_PREFIX:-}" ]] && extra_env+=(--setenv="CONDA_PREFIX=$CONDA_PREFIX")
[[ -n "${MODULAR_HOME:-}" ]] && extra_env+=(--setenv="MODULAR_HOME=$MODULAR_HOME")
[[ -n "${LD_LIBRARY_PATH:-}" ]] && extra_env+=(--setenv="LD_LIBRARY_PATH=$LD_LIBRARY_PATH")

if (( EUID == 0 )); then
  systemd_runner=(systemd-run)
elif sudo -n /usr/bin/true 2>/dev/null; then
  systemd_runner=(sudo -n systemd-run)
elif [[ "$auth_mode" == "polkit" ]]; then
  systemd_runner=(pkexec /usr/bin/systemd-run)
else
  echo "mem_safe_system: system-manager authorization is not cached" >&2
  echo "mem_safe_system: run 'sudo -v' in a trusted terminal, or explicitly use" >&2
  echo "mem_safe_system: MEM_SAFE_SYSTEM_AUTH=polkit for a secure desktop prompt" >&2
  exit 77
fi

exec "${systemd_runner[@]}" \
  --quiet --wait --collect --pipe --service-type=exec \
  --unit="$unit_name" \
  --uid="$task_uid" --gid="$task_gid" \
  --working-directory="$task_dir" \
  --property="MemoryHigh=$MEM_HIGH" \
  --property="MemoryMax=$MEM_MAX" \
  --property="MemorySwapMax=$SWAP_MAX" \
  --property=OOMPolicy=kill \
  --setenv="PATH=$PATH" \
  "${extra_env[@]}" \
  -- "$prog_path" "$@"
