#!/usr/bin/env bash
# Run a command and print final cgroup-v2 memory evidence from inside its scope.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "run_with_cgroup_metrics: usage: $0 <program> [args...]" >&2
  exit 64
fi

scope_rel="$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)"
scope_path="/sys/fs/cgroup${scope_rel}"

printf '[cgroup] scope=%s max=%s high=%s swap_max=%s\n' \
  "$scope_rel" \
  "$(<"${scope_path}/memory.max")" \
  "$(<"${scope_path}/memory.high")" \
  "$(<"${scope_path}/memory.swap.max")"

set +e
/usr/bin/time -v "$@"
run_rc=$?
set -e

printf '[cgroup] memory.current='; cat "${scope_path}/memory.current"
printf '[cgroup] memory.peak='; cat "${scope_path}/memory.peak"
printf '[cgroup] memory.swap.peak='; cat "${scope_path}/memory.swap.peak"
printf '[cgroup] memory.events\n'; cat "${scope_path}/memory.events"
printf '[cgroup] memory.pressure\n'; cat "${scope_path}/memory.pressure"

exit "$run_rc"
