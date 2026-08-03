#!/usr/bin/env bash
# Run a SMALL command inside a memory-capped transient systemd user scope.
#
# This is intentionally limited to 12 GiB. A user scope is still nested below
# user@1000.service, whose systemd-oomd pressure policy can kill the entire
# desktop session before a large child scope reaches MemoryMax. Use
# scripts/mem_safe_system.sh for image/model runtimes or any larger job.
#
# Why this exists: the user@1000.service slice runs ManagedOOMMemoryPressure=kill.
# A `mojo build -j 0` of serenity_daemon.mojo peaks past 60 GB on a 62 GB box and
# drives the whole user slice over the 50% memory-pressure limit. On 2026-07-31,
# a 24G/20G user scope running Klein reached 20.0 GiB and 87.94% Avg10 pressure;
# systemd-oomd destroyed the user session before the child scope was OOM-killed.
#
# Usage:   scripts/mem_safe.sh <program> [args...]
# Tunable within the small-job ceiling: MEM_MAX, MEM_HIGH, SWAP_MAX.
set -euo pipefail

MEM_MAX="${MEM_MAX:-12G}"     # hard ceiling for a user-manager child scope
MEM_HIGH="${MEM_HIGH:-10G}"   # soft reclaim threshold for bounded builds
SWAP_MAX="${SWAP_MAX:-2G}"    # small spill, not minutes of thrash
SAFE_USER_MAX="12G"

if [[ $# -lt 1 ]]; then
  echo "mem_safe: usage: $0 <program> [args...]" >&2
  exit 64
fi

max_bytes="$(numfmt --from=iec "$MEM_MAX")" || {
  echo "mem_safe: invalid MEM_MAX: $MEM_MAX" >&2
  exit 64
}
safe_user_bytes="$(numfmt --from=iec "$SAFE_USER_MAX")"
if (( max_bytes > safe_user_bytes )); then
  echo "mem_safe: refusing unsafe $MEM_MAX user scope (maximum $SAFE_USER_MAX)" >&2
  echo "mem_safe: large runtimes must use scripts/mem_safe_system.sh" >&2
  exit 78
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
