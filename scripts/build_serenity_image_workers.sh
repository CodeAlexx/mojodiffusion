#!/usr/bin/env bash
set -euo pipefail

# Build production image workers with the measured-safe compiler settings.
# Passing names builds a subset, for example:
#   pixi run build-image-workers-safe flux chroma sd3

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if (( $# )); then
  worker_names=("$@")
else
  worker_names=(
    zimage qwenimage ideogram4 sdxl anima
    sd3 flux klein krea2 chroma sensenova lens
  )
fi

mkdir -p output/bin
for worker_name in "${worker_names[@]}"; do
  case "$worker_name" in
    zimage|qwenimage|ideogram4|sdxl|anima|sd3|flux|klein|krea2|chroma|sensenova|lens) ;;
    *)
      echo "unsupported image worker: $worker_name" >&2
      exit 2
      ;;
  esac

  source_path="serenitymojo/serve/serenity_worker_${worker_name}.mojo"
  final_path="output/bin/serenity_worker_${worker_name}"
  scratch_path="${final_path}.optimizing"
  echo "[image-worker-build] ${worker_name}: pinned -O2, one compiler job"
  MEM_MAX="${MEM_MAX:-12G}" \
  MEM_HIGH="${MEM_HIGH:-10G}" \
  SWAP_MAX="${SWAP_MAX:-2G}" \
    pixi run scripts/mem_safe.sh mojo build \
      --optimization-level 2 -j 1 \
      -I . -I trainer/src -I vendor/mojo-libs \
      -Xlinker -L.pixi/envs/default/lib \
      -Xlinker -rpath-link -Xlinker .pixi/envs/default/lib \
      -Xlinker -L.pixi/envs/default/targets/x86_64-linux/lib/stubs \
      -Xlinker -lcuda -Xlinker -lcublas -Xlinker -lm -Xlinker -ldl \
      -Xlinker -lpng16 -Xlinker -lturbojpeg -Xlinker -lsqlite3 \
      -Xlinker -Lserenitymojo/ops/cshim/lib \
      -Xlinker -lserenity_cudnn_sdpa \
      -Xlinker -rpath \
      -Xlinker "$repo_root/.pixi/envs/default/lib" \
      "$source_path" -o "$scratch_path"

  LD_LIBRARY_PATH="$repo_root/serenitymojo/ops/cshim/lib:$repo_root/.pixi/envs/default/lib:$repo_root/.pixi/envs/default/targets/x86_64-linux/lib/stubs" \
    "$scratch_path" >/dev/null
  mv "$scratch_path" "$final_path"
  stat -c '[image-worker-build] installed %n (%s bytes)' "$final_path"
done
