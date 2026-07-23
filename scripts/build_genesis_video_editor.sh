#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

# Genesis is a deliberately separate Rust/C/FFmpeg/OpenCL subsystem. Building
# this target does not compile or call Mojo and never launches the native egui UI.
bash scripts/mem_safe.sh cargo build \
  --manifest-path vendor/genesis/Cargo.toml \
  -p gcompose \
  --release

install -d -- output/bin
install -m 0755 -- \
  vendor/genesis/target/release/gcompose \
  output/bin/genesis-gcompose

echo "Genesis video worker installed at $repo_root/output/bin/genesis-gcompose"
