#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

roots=(serenitymojo serenity-server trainer scripts docs)
printf '%-20s %12s %12s\n' root files occurrences
for root in "${roots[@]}"; do
  files=$(rg -l '/home/[A-Za-z0-9._-]+/|/mnt/disk[0-9]+/' "$root" 2>/dev/null | wc -l)
  occurrences=$(rg -o '/home/[A-Za-z0-9._-]+/|/mnt/disk[0-9]+/' "$root" 2>/dev/null | wc -l)
  printf '%-20s %12d %12d\n' "$root" "$files" "$occurrences"
done
