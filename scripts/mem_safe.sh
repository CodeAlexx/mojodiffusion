#!/usr/bin/env bash
# Run a command inside a memory-capped transient systemd user scope so that an
# out-of-memory blowup kills ONLY this job's cgroup — never the GNOME session.
#
# Why this exists: the user@1000.service slice runs ManagedOOMMemoryPressure=kill.
# A `mojo build -j 0` of serenity_daemon.mojo peaks past 60 GB on a 62 GB box and
# drives the whole user slice over the 50% memory-pressure limit, so systemd-oomd
# kills a scope in the slice and the desktop drops to the login screen (measured
# 2026-06-13, 10:07). A per-job scope with MemoryMax confines the blowup: the
# cgroup-local OOM killer reaps inside this scope before global pressure spikes.
#
# Usage:   scripts/mem_safe.sh <program> [args...]
# Tunable: MEM_MAX (hard cap) MEM_HIGH (reclaim throttle) SWAP_MAX  e.g.
#          MEM_MAX=52G scripts/mem_safe.sh mojo build ...
set -euo pipefail

MEM_MAX="${MEM_MAX:-44G}"     # hard ceiling: scope is OOM-killed if it exceeds this
MEM_HIGH="${MEM_HIGH:-40G}"   # soft: kernel reclaims aggressively above this
SWAP_MAX="${SWAP_MAX:-4G}"    # small spill, not minutes of thrash

if [[ $# -lt 1 ]]; then
  echo "mem_safe: usage: $0 <program> [args...]" >&2
  exit 64
fi

# Resolve the program on the CURRENT PATH (pixi-activated env) so the scope, which
# may run under the user manager's leaner PATH, still finds it. --scope inherits
# our environment and cwd, so MODULAR_* / conda vars carry through.
prog="$1"; shift
prog_path="$(command -v "$prog")" || { echo "mem_safe: '$prog' not on PATH" >&2; exit 127; }

exec systemd-run --user --scope --quiet \
  -p MemoryHigh="$MEM_HIGH" \
  -p MemoryMax="$MEM_MAX" \
  -p MemorySwapMax="$SWAP_MAX" \
  -- "$prog_path" "$@"
