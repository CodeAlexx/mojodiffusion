#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

if ! command -v pixi >/dev/null 2>&1; then
  echo "Pixi is required. Install it from https://pixi.sh and rerun scripts/install.sh." >&2
  exit 1
fi

archive_path() {
  local path=$1
  local reason=$2
  local stamp archive_root archive_name destination
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  archive_root="$repo_root/output/build-artifact-archives"
  archive_name=${path//\//-}
  destination="$archive_root/${archive_name}-relocated-${stamp}"
  if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite relocation archive: $destination" >&2
    exit 1
  fi
  mkdir -p -- "$archive_root"
  mv -- "$path" "$destination"
  echo "Archived non-relocatable $reason at $destination"
}

# Pixi/MAX records an absolute prefix. Reusing that environment after moving a
# checkout makes Mojo locate the old standard library. Preserve it and rebuild.
modular_cfg=.pixi/envs/default/share/max/modular.cfg
if [[ -f "$modular_cfg" ]] && ! grep -Fq -- "$repo_root" "$modular_cfg"; then
  archive_path .pixi "Pixi environment"
fi

# Rust env! values and dep-info also carry the checkout path. A moved Cargo
# target can contain a mix of newly rebuilt crates and stale dependencies, so
# checking one dep-info file is insufficient. Inspect every absolute Cargo
# target prefix and preserve the whole target if any prefix belongs elsewhere.
cargo_target_is_stale() {
  local target=$1
  local prefix
  while IFS= read -r prefix; do
    [[ -n "$prefix" ]] || continue
    if [[ "$prefix" != "$repo_root/"* ]]; then
      return 0
    fi
  done < <(
    rg --no-filename -o -P '/[^ :]+/target/(?=(?:debug|release)/)' "$target" --glob '*.d' 2>/dev/null \
      | sort -u || true
  )
  return 1
}

for target in trainer/webui/target serenity-server/target target; do
  [[ -d "$target" ]] || continue
  if cargo_target_is_stale "$target"; then
    archive_path "$target" "Cargo target"
  fi
done

pixi install --locked
pixi run setup
pixi run repository-check
pixi run build-trainer
pixi run build-inference
pixi run check

echo "Serenity installation verified at $repo_root"
echo "Install model assets under $repo_root/models or SERENITY_MODEL_ROOT as documented in models/README.md."
