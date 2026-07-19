#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  echo "usage: $0 output/bin/<binary> [...]" >&2
  exit 2
fi
if ! command -v patchelf >/dev/null 2>&1; then
  echo "patchelf is required; run this build through the locked Pixi environment" >&2
  exit 1
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

for binary in "$@"; do
  case "$binary" in
    output/bin/*)
      runtime_rpath='$ORIGIN/../../.pixi/envs/default/lib:$ORIGIN/../../serenitymojo/ops/cshim/lib'
      ;;
    output/verification/bin/*)
      runtime_rpath='$ORIGIN/../../../.pixi/envs/default/lib:$ORIGIN/../../../serenitymojo/ops/cshim/lib'
      ;;
    *)
      echo "refusing to patch a binary outside Serenity output directories: $binary" >&2
      exit 1
      ;;
  esac
  if [[ ! -f "$binary" || ! -x "$binary" ]]; then
    echo "built Mojo executable is missing: $binary" >&2
    exit 1
  fi
  patchelf --set-rpath "$runtime_rpath" "$binary"
  actual=$(patchelf --print-rpath "$binary")
  if [[ "$actual" != "$runtime_rpath" ]]; then
    echo "failed to set portable RUNPATH on $binary: $actual" >&2
    exit 1
  fi
  echo "portable RUNPATH: $binary"
done
