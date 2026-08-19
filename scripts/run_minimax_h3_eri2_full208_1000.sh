#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$repo_root/scripts/run_minimax_h3_eri2_mojo_4000.sh" \
  "$repo_root/serenitymojo/configs/minimax_h3_eri2_full208_1000.json"
