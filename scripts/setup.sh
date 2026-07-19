#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

mkdir -p output/bin \
  models/anima models/chroma models/flux1-dev models/hidream-o1 \
  models/ideogram4 models/klein models/klein4b models/klein9b \
  models/checkpoints models/loras models/wildcards \
  models/krea2 models/qwen-image models/qwen3-4b models/qwen3-8b \
  models/qwen3-vl-4b models/sd3.5 models/sdxl models/sensenova-u1 \
  models/text-encoders models/vae models/zimage

echo "Serenity workspace is ready at $repo_root"
echo "Model files belong under $repo_root/models; see models/README.md."
