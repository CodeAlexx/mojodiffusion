#!/usr/bin/env bash
# Prove that mem_safe_system.sh fails a sacrificial 64 MiB job outside the
# desktop's user@UID.service and leaves that user manager alive and unchanged.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user_unit="user@$(id -u).service"
before_pid="$(systemctl show "$user_unit" --property=MainPID --value)"

if [[ -z "$before_pid" || "$before_pid" == "0" ]]; then
  echo "isolation-check: $user_unit is not active" >&2
  exit 1
fi

set +e
probe_output="$({
  MEM_MAX=64M MEM_HIGH=infinity SWAP_MAX=0 \
    "$repo_root/scripts/mem_safe_system.sh" python3 -c '
from pathlib import Path
import time

print(Path("/proc/self/cgroup").read_text().strip(), flush=True)
blocks = []
for index in range(16):
    blocks.append(bytearray(16 * 1024 * 1024))
    print(f"allocated_mib={(index + 1) * 16}", flush=True)
    time.sleep(0.05)
';
} 2>&1)"
probe_rc=$?
set -e

printf '%s\n' "$probe_output"

after_pid="$(systemctl show "$user_unit" --property=MainPID --value)"
if [[ "$probe_output" != *"0::/system.slice/serenity-memory-"*".service"* ]]; then
  echo "isolation-check: probe did not enter the dedicated system slice" >&2
  exit 1
fi
if (( probe_rc == 0 )); then
  echo "isolation-check: sacrificial allocation unexpectedly survived" >&2
  exit 1
fi
if [[ "$after_pid" != "$before_pid" ]]; then
  echo "isolation-check: $user_unit PID changed ($before_pid -> $after_pid)" >&2
  exit 1
fi
if ! systemctl is-active --quiet "$user_unit"; then
  echo "isolation-check: $user_unit is not active after the probe" >&2
  exit 1
fi

echo "isolation-check: PASS; system-scope job failed locally and $user_unit PID $after_pid survived"
